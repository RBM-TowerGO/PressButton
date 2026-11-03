"""Replay two saved localization requests against the model, never a robot."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from diagnose_localized_feedback import model_tunnel, digest, image_hashes
from pressb.agent_loop import model_localization as loc
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.view_acquisition import CONTACT_TEXT


def run():
    source = ROOT / 'outputs/agent_web/20261102_184803_299e7957/agent/rounds'
    output = Path(__file__).resolve().parent / 'results'
    output.mkdir(exist_ok=False)
    (output/'model_localization_used.py').write_bytes(Path(loc.__file__).read_bytes())
    report = dict(robot_connected=False, robot_commands_sent=0,
                  method='Same archived images and input messages; replace only localization system prompt and tool description. Compare with original recorded metrics, not a randomized benchmark.',
                  results=[])
    with model_tunnel('Thor', 18117, None) as base:
        for folder in ('0001','0002'):
            directory=source/folder/'localization'
            original_bytes=(directory/'request.json').read_bytes()
            original=json.loads(original_bytes)
            body=copy.deepcopy(original)
            body['messages'][0]['content']=loc.PROMPT.format(contact=CONTACT_TEXT['white_arrow'])
            body['tools']=[copy.deepcopy(loc.TOOL)]
            assert image_hashes(body)==image_hashes(original)
            oid=json.loads((source/folder/'decision_context.json').read_text())['observation_id']
            before=json.loads((directory/'metrics.json').read_text())
            row=dict(round=int(folder)+1,observation_id=oid,source=str(directory.relative_to(ROOT)),
                     source_sha256=digest(original_bytes),images_unchanged=True,
                     original={k:before[k] for k in ('elapsed_s','prompt_tokens','completion_tokens')})
            try:
                result=loc.inspect(base,body,oid,output/folder)
                metrics=json.loads((output/folder/'metrics.json').read_text())
                row['compact']={k:metrics[k] for k in ('elapsed_s','prompt_tokens','completion_tokens','attempts')}
                row['views']={view:{k:result[view][k] for k in (
                    'button_visibility','contact_visibility','button_box','contact_tip','note')}
                    for view in ('head','wrist')}
                row['parse_notes']=result['parse_notes']
            except Exception as exc:
                row['error']=str(exc)
            row['source_unchanged']=(directory/'request.json').read_bytes()==original_bytes
            report['results'].append(row)
            atomic_json(output/'report.json',report)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    report['completed']=True
    atomic_json(output/'report.json',report)


if __name__=='__main__':
    run()
