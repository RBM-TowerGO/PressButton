"""Local-only, lossless NumPy RPC for the frozen Astribot model."""
import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint-step', type=int, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--port', type=int, default=8794)
    a = p.parse_args()
    sys.path.insert(0, str(a.run / 'runtime'))
    from starVLA.inference.astribot_policy import AstribotPolicy
    import torch
    started = time.perf_counter()
    policy = AstribotPolicy(a.run / f'checkpoints/step_{a.checkpoint_step:06d}', a.device)
    health = dict(ready=True, checkpoint_step=a.checkpoint_step, loaded_tensors=policy.loaded_tensors,
                  model_sha256=policy.checked['manifest']['model_sha256'],
                  torch=torch.__version__, device=a.device,
                  gpu=torch.cuda.get_device_name(policy.device),
                  load_seconds=time.perf_counter()-started)
    print(json.dumps(health), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, payload, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.respond(200 if self.path == '/health' else 404,
                         json.dumps(health).encode(), 'application/json')

        def do_POST(self):
            try:
                if self.path != '/predict':
                    raise ValueError('Unknown endpoint')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 8_000_000:
                    raise ValueError('Invalid payload length')
                with np.load(io.BytesIO(self.rfile.read(size)), allow_pickle=False) as data:
                    head, wrist, state = data['head'], data['wrist_right'], data['state']
                    if head.shape != (720,1280,3) or wrist.shape != (480,640,3):
                        raise ValueError('Wrong camera resolution/order')
                    if head.dtype != np.uint8 or wrist.dtype != np.uint8 or state.shape != (32,):
                        raise ValueError('Wrong image or state type')
                    started = time.perf_counter()
                    result = policy.predict(head_image=Image.fromarray(head),
                        wrist_right_image=Image.fromarray(wrist), state=state,
                        seed=int(data['seed']))
                    torch.cuda.synchronize(policy.device)
                    result['inference_seconds'] = np.array(time.perf_counter()-started)
                buffer = io.BytesIO()
                np.savez(buffer, **result)
                self.respond(200, buffer.getvalue(), 'application/octet-stream')
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True)
                self.respond(500, json.dumps(dict(error=error)).encode(), 'application/json')

    HTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
