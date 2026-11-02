import base64,hashlib,json,sys,time,shutil
from pathlib import Path
from PIL import Image
ROOT=Path.cwd()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from pressb.agent_loop.common import atomic_json,request
from validate_agent_view_recovery import model_tunnel
OUT=ROOT/'docs/report/20261102_view_loop_audit/pixel_probe'
source=ROOT/'outputs/agent_web/20261102_142505_13668097/agent/observations/981630409f1349a2a93a53a23eb19399'
for name in ('head','right'):shutil.copy2(source/(name+'.jpg'),OUT/(name+'.jpg'))
Image.new('RGB',(640,360),(128,128,128)).save(OUT/'blank.jpg','JPEG')
question='请仅根据附图返回简短JSON：image_available（是否收到可观察的图像）、main_subject（主要可见内容）、button（是否可见按钮）、gripper（是否可见夹爪）、marker（是否可见标记及外观）、positions（上述可见物的大致位置）。不可见就写不可见，不要根据问题假设物体存在。'
summary=dict(robot_connected=False,robot_commands_sent=0,independent_requests=True,shared_question=question,cases=[])
atomic_json(OUT/'summary.json',summary)
try:
 with model_tunnel('Thor',18022):
  base='http://127.0.0.1:18022'
  atomic_json(OUT/'models.json',request(base,'/v1/models'))
  for name in ('head','right','blank'):
   data=(OUT/(name+'.jpg')).read_bytes()
   body=dict(model='qwen38',temperature=0,max_tokens=256,stream=False,chat_template_kwargs=dict(enable_thinking=False),messages=[dict(role='user',content=[dict(type='text',text=question),dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(data).decode()))])])
   folder=OUT/name;atomic_json(folder/'request.json',body)
   started=time.monotonic()
   response=request(base,'/v1/chat/completions',body,timeout=120)
   elapsed=time.monotonic()-started
   atomic_json(folder/'response.json',response)
   row=dict(case=name,elapsed_s=elapsed,usage=response.get('usage'),image_sha256=hashlib.sha256(data).hexdigest(),response_id=response.get('id'),finish_reason=response['choices'][0].get('finish_reason'),content=response['choices'][0]['message'].get('content'))
   summary['cases'].append(row);atomic_json(OUT/'summary.json',summary)
   print(json.dumps(row,ensure_ascii=False),flush=True)
except Exception as exc:
 summary.update(error=str(exc),error_type=type(exc).__name__);raise
finally:
 atomic_json(OUT/'summary.json',summary)
