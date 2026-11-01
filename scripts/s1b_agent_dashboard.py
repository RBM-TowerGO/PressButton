"""Loopback-only live camera and recorded Agent output dashboard."""
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pressb.agent_loop.common import request


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>S1B · Agent 实验</title>
<style>
*{box-sizing:border-box}body{margin:0;padding:24px;background:#101720;color:#e7eff7;font:15px system-ui,sans-serif}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:20px}h1{font-size:24px;margin:0 0 6px}h2{font-size:17px;margin:0 0 12px}p{line-height:1.6}.muted{color:#a9bacc}.status{color:#77d9c3}button{padding:10px 18px;background:#a23838;color:white;border:1px solid #df7777;border-radius:8px;font:inherit;cursor:pointer}button:disabled{opacity:.5;cursor:default}
main{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(340px,1fr);gap:20px}article,.panel{padding:18px;background:#192331;border:1px solid #314154;border-radius:12px;margin-bottom:16px}.screen{background:#080d13;aspect-ratio:16/9;display:grid;place-items:center;overflow:hidden}.screen img{width:100%;height:100%;object-fit:contain}.screen img:not([src]){visibility:hidden}footer{padding-top:10px;color:#a9bacc}.agent{max-height:85vh;overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace;margin:10px 0}summary{cursor:pointer;color:#a9bacc}.round{border-top:1px solid #314154;padding:15px 0}.round:first-child{border-top:0}.reason{white-space:pre-wrap}.badge{color:#77d9c3;margin-left:12px}#error{color:#ffb4a2}#goal{overflow-wrap:anywhere}#terminal{max-height:260px;overflow:auto}@media(max-width:950px){main{grid-template-columns:1fr}.agent{max-height:none}body{padding:14px}header{align-items:flex-start}}
</style></head><body>
<header><div><h1>S1B / Agent 闭环实验</h1><div id="status" class="status" aria-live="polite">等待启动</div></div><button id="stop">停止执行</button></header>
<p id="goal" class="muted"></p><p id="error" role="alert"></p>
<main><section><article><h2>头部相机</h2><div class="screen"><img id="head" alt="头部相机画面"></div><footer id="headStatus">等待画面</footer></article>
<article><h2>右腕相机</h2><div class="screen"><img id="right" alt="右腕相机画面"></div><footer id="rightStatus">等待画面</footer></article>
<div class="panel"><div id="depth">固定点深度：—</div><p class="muted">原始画面。深度无效不代表按钮已触发；只有看到目标按钮亮起才能确认触发。</p></div></section>
<section class="panel agent"><h2>Agent 输出</h2><p class="muted">每轮显示动作参数、决策理由和执行结果。任务结束后保留记录。</p><div id="rounds"></div><details><summary>启动与运行日志</summary><pre id="terminal"></pre></details><details><summary>本次结果</summary><pre id="result">尚未结束</pre></details></section></main>
<script>
const $=id=>document.getElementById(id);
const phases={starting:'正在启动机器人服务',waiting_for_service:'等待服务就绪 / 终端接管控制权',running:'Agent 自动闭环运行中',finished:'本次任务已结束',error:'本次任务已停止：发生错误',stopping:'正在停止'};
let rendered='';
function renderRounds(rows){const signature=JSON.stringify(rows);if(signature===rendered)return;rendered=signature;const root=$('rounds');root.replaceChildren();if(!rows.length){root.textContent='等待第一轮模型请求…';return;}for(const row of [...rows].reverse()){const card=document.createElement('div');card.className='round';const title=document.createElement('strong');title.textContent='第 '+row.round+' 轮';const badge=document.createElement('span');badge.className='badge';badge.textContent=row.phase;card.append(title,badge);if(row.proposal){const p=row.proposal;const pre=document.createElement('pre');const args={...p.arguments};delete args.reason;delete args.evidence;pre.textContent=p.name+'\n'+JSON.stringify(args,null,2);const reason=document.createElement('p');reason.className='reason';reason.textContent=p.arguments.reason||p.arguments.evidence||'';card.append(pre,reason);}if(row.elapsed_s!==null){const timing=document.createElement('small');timing.className='muted';timing.textContent='模型用时 '+row.elapsed_s.toFixed(1)+' 秒';card.append(timing);}if(row.error){const e=document.createElement('pre');e.textContent=JSON.stringify(row.error,null,2);card.append(e);}if(row.response){const detail=document.createElement('details');const label=document.createElement('summary');label.textContent='模型原始输出';const pre=document.createElement('pre');pre.textContent=JSON.stringify(row.response,null,2);detail.append(label,pre);card.append(detail);}root.append(card);}}
async function stateTick(){try{const r=await fetch('/api/agent',{cache:'no-store',signal:AbortSignal.timeout(3000)});if(!r.ok)throw Error('HTTP '+r.status);const s=await r.json();$('status').textContent=phases[s.launcher.phase]||s.launcher.phase||'等待启动';$('goal').textContent=s.goal||'';$('error').textContent=s.launcher.error||s.result?.error||'';$('stop').disabled=['finished','error','stopping'].includes(s.launcher.phase);renderRounds(s.rounds);$('terminal').textContent=s.log;$('result').textContent=s.result?JSON.stringify(s.result,null,2):'尚未结束';}catch(e){$('status').textContent='网页服务连接中断，状态已停止更新';}finally{setTimeout(stateTick,700);}}
async function cameraTick(){try{const r=await fetch('/api/cameras',{cache:'no-store',signal:AbortSignal.timeout(3000)});if(!r.ok)throw Error('HTTP '+r.status);const s=await r.json();for(const v of ['head','right']){const f=s.frames?.[v];if(f?.jpeg)$(v).src='data:image/jpeg;base64,'+f.jpeg;$(v+'Status').textContent=s.archived?'已停止实时更新 · 最后保存的观测':f?(f.stale?'画面已过期':'实时画面')+' · '+Math.round(f.age_ms)+' ms':'等待画面';}const d=s.depth?.median_m;$('depth').textContent='固定点深度：'+(typeof d==='number'?(d*1000).toFixed(1)+' mm':'无有效读数');}catch(e){for(const v of ['head','right'])$(v+'Status').textContent='连接中断 · 显示的画面已过期';$('depth').textContent='固定点深度：—';}finally{setTimeout(cameraTick,400);}}
$('stop').onclick=async()=>{$('stop').disabled=true;try{const r=await fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}',signal:AbortSignal.timeout(3000)});if(!r.ok)throw Error('HTTP '+r.status);$('status').textContent='正在停止';}catch(e){$('error').textContent='停止请求未确认，请在终端按 Ctrl-C。';$('stop').disabled=false;}};
stateTick();cameraTick();
</script></body></html>'''


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


class Dashboard:
    def __init__(self, directory, camera_base, stop):
        self.root = Path(directory)
        self.run = self.root / 'agent'
        self.camera_base = camera_base
        self.stop = stop

    def state(self):
        rows = []
        outcome = read_json(self.run/'result.json')
        for folder in sorted((self.run/'rounds').glob('*')):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            proposal = read_json(folder/'proposal.json')
            # Completed round files are atomic and contain large state traces.
            # Their existence is enough here; never reload those traces on a UI poll.
            completed = (folder/'result.json').exists()
            error = read_json(folder/'error.json')
            response = read_json(folder/'response-1.json') or read_json(folder/'response-0.json')
            phase = '请求模型中'
            if proposal:
                phase = {'report_done': '模型报告完成', 'report_unable': '模型无法继续',
                         'request_observation': '请求新观测'}.get(proposal['name'], '已生成建议')
            if (folder/'execution.json').exists():
                phase = '执行 / 采集新观测中'
            if completed:
                phase = '执行完成'
            elif outcome and outcome.get('status') in ('error', 'operator_stopped'):
                phase = '已停止'
            if error:
                phase = '请求失败'
            rows.append(dict(round=int(folder.name)+1, phase=phase, proposal=proposal, error=error,
                             response=response.get('response') if response else None,
                             elapsed_s=response.get('elapsed_s') if response else None))
        try:
            with (self.root/'terminal.log').open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-24000))
                log = stream.read().decode(errors='replace')
        except OSError:
            log = ''
        task = read_json(self.run/'task.json', {})
        launcher = read_json(self.root/'launcher.json', {})
        return dict(launcher=launcher, goal=task.get('goal', launcher.get('goal')), rounds=rows,
                    result=outcome, log=log)

    def cameras(self):
        try:
            snapshot = request(self.camera_base, '/api/snapshot', timeout=1)
            snapshot['archived'] = False
            return snapshot
        except (OSError, ValueError):
            manifests = sorted((self.run/'observations').glob('*/manifest.json'),
                               key=lambda p: p.stat().st_mtime_ns, reverse=True)
            frames = {}
            metadata = {}
            if manifests:
                metadata = read_json(manifests[0], {})
                for view in ('head', 'right'):
                    path = manifests[0].parent / (view+'.jpg')
                    if path.exists():
                        frames[view] = dict(jpeg=base64.b64encode(path.read_bytes()).decode(), stale=True)
            return dict(frames=frames, depth={}, archived=True,
                        observation_id=metadata.get('observation_id'))


def serve(dashboard, port=8766):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data, html=False):
            body = data.encode() if html else json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == '/':
                    self.reply(200, HTML, html=True)
                elif self.path == '/api/agent':
                    self.reply(200, dashboard.state())
                elif self.path == '/api/cameras':
                    self.reply(200, dashboard.cameras())
                else:
                    self.reply(404, dict(error='not_found'))
            except Exception as exc:
                self.reply(503, dict(error=str(exc)))

        def do_POST(self):
            self.connection.settimeout(2)
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                self.reply(403, dict(error='cross_origin_request'))
                return
            if self.path != '/api/stop':
                self.reply(404, dict(error='not_found'))
                return
            try:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('json_required')
                if self.headers.get('Transfer-Encoding') or int(self.headers.get('Content-Length', '0')) != 2:
                    raise ValueError('empty_object_required')
                if self.rfile.read(2) != b'{}':
                    raise ValueError('empty_object_required')
                dashboard.stop.set()
                self.reply(200, dict(status='stop_requested'))
            except (ValueError, OSError) as exc:
                self.reply(400, dict(error=str(exc)))

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
