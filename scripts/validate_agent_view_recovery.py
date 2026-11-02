#!/usr/bin/env python3
"""Three saved-image localization checks; model access only, no robot client."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pressb.agent_loop.common import atomic_json, request
from pressb.agent_loop.conversation import image_count, live_message
from pressb.agent_loop import view_acquisition


def load(path):
    return json.loads(Path(path).read_text())


@contextmanager
def model_tunnel(host, port):
    """Open only the model service. No robot/service startup or robot HTTP calls."""
    if not host:
        yield
        return
    process = subprocess.Popen([
        'ssh', '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
        '-L', f'127.0.0.1:{port}:127.0.0.1:8000', host])
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('model_tunnel_exited')
            try:
                request(f'http://127.0.0.1:{port}', '/v1/models', timeout=1)
                break
            except (OSError, ValueError):
                time.sleep(.2)
        else:
            raise TimeoutError('model_tunnel_unavailable')
        yield
    finally:
        process.terminate()
        try:
            process.wait(3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def overlay(source, destination, assessment, label):
    """Display model-provided points without modifying model input images."""
    with Image.open(source) as raw:
        image = raw.convert('RGB')
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for key, color in (('button_box', '#00ff00'), ('arrow_box', '#ff9900')):
        value = assessment.get(key)
        if value is not None:
            coordinates = [value[0]*width, value[1]*height, value[2]*width, value[3]*height]
            draw.rectangle(coordinates, outline=color, width=2)
            draw.text((coordinates[0], max(20, coordinates[1]-14)), key, fill=color)
    tip = assessment.get('contact_tip')
    if tip is not None:
        x, y = tip[0]*width, tip[1]*height
        draw.line((x-8, y, x+8, y), fill='#00ffff', width=3)
        draw.line((x, y-8, x, y+8), fill='#00ffff', width=3)
        draw.text((max(0, x-60), max(20, y-20)), 'physical_tip', fill='#00ffff')
    relation = assessment.get('derived_relation', {})
    title = f"{label}: MODEL localization | {relation.get('vertical', 'unknown')}/{relation.get('horizontal', 'unknown')}"
    # Keep the original frame's top edge visible: the first saved button is near it.
    canvas = Image.new('RGB', (width, height+22), 'black')
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((3, 3), title, fill='white')
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=ROOT/'outputs/agent_web/20261102_120741_0ca36af5/agent')
    parser.add_argument('--model-base')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--thor-port', type=int, default=18021)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/report'/(
        'view_recovery_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    cancel = threading.Event()

    def stop(signum, frame):
        cancel.set()
        raise KeyboardInterrupt('operator_stopped_model_replay')

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    base = args.model_base or f'http://127.0.0.1:{args.thor_port}/v1'
    summary = dict(model_only=True, robot_connected=False, commands_sent=0,
        source_run=str(args.run.resolve()), rounds=[1, 4, 8], thinking=False,
        source_code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (Path(__file__).resolve(), Path(view_acquisition.__file__).resolve())},
        cases=[], status='running',
        limitations=[
            'Three independent saved observations, one requested check each; no closed-loop execution.',
            'Annotations visualize model localization, not calibrated contact or human confirmation.',
            'A valid schema is not proof of correct visual localization.',
            'Original and overlay images are distinct; only original images are sent to the model.'])
    atomic_json(args.output/'validation.json', summary)
    try:
        with model_tunnel(None if args.model_base else args.thor_host, args.thor_port):
            for round_number in (1, 4, 8):
                if cancel.is_set():
                    raise InterruptedError('operator_stopped')
                context = load(args.run/'rounds'/f'{round_number-1:04d}'/'decision_context.json')
                oid = context['observation_id']
                original = args.run/'observations'/oid
                obs = load(original/'manifest.json')
                if obs['observation_id'] != oid:
                    raise ValueError('saved_observation_id_mismatch')
                head, wrist = (original/'head.jpg').read_bytes(), (original/'right.jpg').read_bytes()
                directory = args.output/f'round_{round_number:02d}'
                directory.mkdir()
                (directory/'head.jpg').write_bytes(head)
                (directory/'right.jpg').write_bytes(wrist)
                source = dict(round=round_number, observation_id=oid,
                    source_directory=str(original.resolve()),
                    sha256={name: hashlib.sha256(data).hexdigest()
                            for name, data in (('head.jpg', head), ('right.jpg', wrist))})
                atomic_json(directory/'source.json', source)
                body = view_acquisition.build_request('按下电梯外呼向下按钮并撤回',
                    live_message(obs, head, wrist), contact_reference='white_arrow')
                if image_count(body['messages']) != 2:
                    raise ValueError('exactly_two_current_original_images_required')
                if body.get('chat_template_kwargs', {}).get('enable_thinking') is not False:
                    raise ValueError('this_check_requires_thinking_disabled')
                print(f'Checking saved round {round_number}, observation {oid}, 2 original images', flush=True)
                assessment = view_acquisition.inspect(base, body, oid, directory/'localization', cancel)
                overlay(directory/'head.jpg', directory/'head_localization.png', assessment['head'], 'HEAD')
                overlay(directory/'right.jpg', directory/'right_localization.png', assessment['wrist'], 'WRIST')
                record = dict(round=round_number, observation_id=oid, assessment=assessment,
                              metrics=load(directory/'localization'/'metrics.json'))
                summary['cases'].append(record)
                atomic_json(args.output/'validation.json', summary)
                print(json.dumps(record, ensure_ascii=False), flush=True)
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'error',
                       error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic_json(args.output/'validation.json', summary)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        print('Model-only report: '+str(args.output), flush=True)
    return summary


if __name__ == '__main__':
    main()
