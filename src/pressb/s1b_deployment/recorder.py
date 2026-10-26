"""Append-only JSONL evidence; errors propagate to the caller."""
import json
from pathlib import Path
import time
from .contract import canonical

REQUIRED = ('session_id', 'observation_id', 'seed', 'model_sha256', 'data_contract_sha256', 'source_time_ns', 'mapped_time_ns', 'observation_monotonic_ns', 'state10', 'action_raw', 'gripper_binary', 'sdk_target', 'fault')

class Recorder:
    def __init__(self, directory, config, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'resolved_config.json').write_text(json.dumps(dict(config=config,sha256=canonical(config)),indent=2)+'\n')
        (self.directory / 'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    def append(self, event, **fields):
        if event == 'prediction':
            missing = [k for k in REQUIRED if k not in fields]
            if missing: raise ValueError('Missing log fields: '+', '.join(missing))
        row=dict(schema='s1b-deployment-log-v1',event=event,monotonic_ns=time.monotonic_ns(),wall_time_ns=time.time_ns(),**fields)
        line=json.dumps(row,allow_nan=False)
        with (self.directory / 'events.jsonl').open('a') as f: f.write(line+'\n')
