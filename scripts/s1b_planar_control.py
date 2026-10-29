#!/usr/bin/env python3
"""Read-only probe by default; calibrated planar tool server with explicit --execute."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pressb.planar_control.runtime import LiveBackend, validate_config
from pressb.planar_control.sensing import Sensors
from pressb.planar_control.api import make_server
from pressb.s1b_deployment.async_log import AsyncLog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/s1b_planar_v1.json')
    parser.add_argument('--guard-config', default='configs/s1b_execution_p4_v1.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--operator-present', action='store_true')
    parser.add_argument('--estop-ready', action='store_true')
    parser.add_argument('--port', type=int, default=6210)
    parser.add_argument('--probe-seconds', type=float, default=5)
    parser.add_argument('--air-probe-mm', type=float, help='local-only finite normal roundtrip in a verified empty workspace')
    parser.add_argument('--keyboard', action='store_true', help='WASD planar jog, F forward 100 mm, R retract, space stop, Q quit')
    parser.add_argument('--step-mm', type=float, default=2., help='keyboard planar displacement per keypress')
    args = parser.parse_args()
    if args.air_probe_mm is not None and not args.execute:
        parser.error('--air-probe-mm requires --execute')
    if args.keyboard and (not args.execute or args.air_probe_mm is not None):
        parser.error('--keyboard requires --execute and cannot combine with --air-probe-mm')
    if args.keyboard and not sys.stdin.isatty():
        parser.error('--keyboard requires a terminal (use ssh -t for remote access)')
    config = json.loads(Path(args.config).read_text())
    if args.execute:
        validate_config(config)  # fail before any SDK initialization
        if args.keyboard:
            import math
            if not math.isfinite(args.step_mm) or not 0 < args.step_mm/1000 <= config['control']['max_move_m']:
                parser.error('--step-mm exceeds the calibrated planar step limit')
        if not args.operator_present or not args.estop_ready:
            parser.error('--execute requires --operator-present and --estop-ready')
    if not 0 < args.probe_seconds <= 60:
        parser.error('--probe-seconds must be in (0, 60]')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    events = (out/'events.jsonl').open('w')
    log = AsyncLog({'events':events}, 1024**3)
    emit = lambda row: log.log('events', row)
    sensors = backend = controller = server = None
    try:
        sensors = Sensors(config['sensors'], emit)
        if not args.execute:
            until = time.monotonic()+args.probe_seconds
            samples = []
            while time.monotonic() < until:
                try:
                    samples.append(sensors.probe())
                except ValueError as exc:
                    samples.append(dict(error=str(exc)))
                time.sleep(.1)
            (out/'probe.json').write_text(json.dumps(samples, indent=2)+'\n')
            print(json.dumps(dict(mode='read_only', samples=len(samples), output=str(out))))
            return
        deadline = time.monotonic()+5
        while True:
            try:
                sensors.monitor.snapshot()
                sensors.images()
                break
            except ValueError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.05)
        guard_config = json.loads(Path(args.guard_config).read_text())
        (out/'guard_config.json').write_text(json.dumps(guard_config, indent=2)+'\n')
        backend = LiveBackend(config, guard_config, sensors, emit)
        controller = backend.acquire(args.operator_present, args.estop_ready)
        def stop_signal(signum, frame):
            controller.stop('operator_signal')
        signal.signal(signal.SIGINT, stop_signal)
        signal.signal(signal.SIGTERM, stop_signal)
        if args.keyboard:
            from pressb.planar_control.keyboard import run_keyboard
            run_keyboard(controller, step_mm=args.step_mm, check=log.check)
            return
        if args.air_probe_mm is not None:
            deadline = time.monotonic()+3
            while True:
                try:
                    observation = controller.observe()
                    break
                except ValueError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.05)
            controller.submit('air_probe', observation['observation_id'], 'local-air-probe', normal_mm=args.air_probe_mm)
            while controller.get_state()['phase'] not in ('READY', 'FAULT', 'STOPPED'):
                log.check()
                time.sleep(.05)
            result = controller.get_state()
            (out/'air_probe.json').write_text(json.dumps(result, indent=2)+'\n')
            if not result['result'] or not result['result']['action_completed']:
                raise ValueError('air_probe_failed')
            return
        server = make_server(controller, args.port)
        server.timeout = .1
        print(json.dumps(dict(mode='execute', address=server.server_address)), flush=True)
        while not controller.cancel.is_set():
            log.check()
            server.handle_request()
    finally:
        cleanup = []
        if server:
            cleanup.append(server.server_close)
        if controller:
            cleanup.append(controller.close)
        elif backend:
            cleanup.append(backend.close)
        if sensors:
            cleanup.append(sensors.close)
        cleanup.extend([log.close, events.close])
        failures = []
        for close in cleanup:
            try:
                close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError('cleanup_failure:'+str(failures[0])) from failures[0]


if __name__ == '__main__':
    main()
