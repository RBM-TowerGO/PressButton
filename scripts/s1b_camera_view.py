#!/usr/bin/env python3
"""Local camera webpage with a persistent measurement point; no robot commands."""
import argparse
import base64
from collections import deque
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import tempfile

import cv2
import numpy as np

TOPICS = {'head':'/astribot_camera/head_rgbd/color_compress/compressed',
          'right':'/astribot_camera/right_wrist_rgbd/color_compress/compressed',
          'depth':'/astribot_camera/right_wrist_rgbd/depth_compress'}


def validate_point(point):
    if point is None:return None
    if (not isinstance(point,list) or len(point)!=2
            or any(type(v) not in (float,int) or not np.isfinite(v) or not 0<=v<=1 for v in point)):
        raise ValueError('测距点必须是 0–1 范围内的二维坐标')
    return [float(v) for v in point]


def point_pixel(point,width,height):
    if point is None:return width//2,height//2
    return tuple(int(np.floor(v*(size-1)+.5)) for v,size in zip(validate_point(point),(width,height)))


def matrix(i):
    return np.array([[i['fx'],0,i['ppx']],[0,i['fy'],i['ppy']],[0,0,1.]],float)


def rays(i):
    yy,xx=np.indices((i['height'],i['width']))
    pixels=np.stack([xx,yy],axis=-1).astype(float).reshape(-1,1,2)
    k=matrix(i);coeff=np.asarray(i['coeffs'],float)
    if i['model'] in (0,4):
        xy=cv2.undistortPoints(pixels,k,coeff if i['model']==4 else np.zeros(5)).reshape(-1,2)
    elif i['model']==2:  # inverse Brown: observed normalized coordinates -> ideal ray
        xy=(pixels.reshape(-1,2)-[i['ppx'],i['ppy']])/[i['fx'],i['fy']]
        xyz=np.column_stack([xy,np.ones(len(xy))])
        xy=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),np.eye(3),coeff)[0].reshape(-1,2)
    else:
        raise ValueError('unsupported_factory_distortion_model')
    return np.column_stack([xy,np.ones(len(xy))])


def project(points,i):
    if i['model'] in (0,4):
        return cv2.projectPoints(points,np.zeros(3),np.zeros(3),matrix(i),
                                 np.asarray(i['coeffs']) if i['model']==4 else np.zeros(5))[0].reshape(-1,2)
    if i['model']==2:
        ideal=(points[:,:2]/points[:,2:3]).reshape(-1,1,2)
        observed=cv2.undistortPoints(ideal,np.eye(3),np.asarray(i['coeffs'])).reshape(-1,2)
        return observed*[i['fx'],i['fy']]+[i['ppx'],i['ppy']]
    raise ValueError('unsupported_factory_distortion_model')


class CenterDepth:
    def __init__(self, metadata):
        self.m=metadata
        self.rays=rays(metadata['depth'])
        self.rotation=np.asarray(metadata['rotation_column_major']).reshape(3,3,order='F')
        self.translation=np.asarray(metadata['translation_m'])
        self.scale=float(metadata['scale_m'])
        if not np.isfinite(self.scale) or self.scale<=0:
            raise ValueError('invalid_device_depth_units')

    def measure(self,data,color_shape,point=None):
        d,c=self.m['depth'],self.m['color']
        if tuple(color_shape[:2]) != (c['height'],c['width']):
            raise ValueError('RGB_resolution_does_not_match_factory_profile')
        if len(data)!=d['width']*d['height']*2:
            raise ValueError('raw_depth_size_mismatch')
        z=np.frombuffer(data,dtype='<u2').astype(float)*self.scale
        valid=(z>0)&np.isfinite(z)
        points=(self.rays[valid]*z[valid,None])@self.rotation.T+self.translation
        points=points[np.isfinite(points).all(1)&(points[:,2]>0)]
        image=np.full((c['height'],c['width']),np.inf)
        if len(points):
            uv=project(points,c)
            inside=np.isfinite(uv).all(1)&(uv[:,0]>-.5)&(uv[:,0]<c['width']-.5)&(uv[:,1]>-.5)&(uv[:,1]<c['height']-.5)
            pixels=np.rint(uv[inside]).astype(int)
            np.minimum.at(image,(pixels[:,1],pixels[:,0]),points[inside,2])
        x,y=point_pixel(point,c['width'],c['height'])
        patch=image[max(0,y-2):y+3,max(0,x-2):x+3]
        samples=patch[np.isfinite(patch)]
        exact=image[y,x]
        return dict(center_px=[x,y],center_m=float(exact) if np.isfinite(exact) else None,
                    median_m=float(np.median(samples)) if len(samples) else None,
                    valid_pixels=int(len(samples)),window=5,window_pixels=int(patch.size),scale_m=self.scale,
                    status='ok' if len(samples) else '测距区域无有效深度',
                    serial=self.m['serial'])


def factory_metadata(source):
    source=Path(source)
    target=source.with_suffix('.bin')
    headers='/opt/astribot_ros/software/astribot_camera_sdk/include/realsense/include'
    if not target.exists() or target.stat().st_mtime<source.stat().st_mtime:
        subprocess.run(['g++','-std=c++11',str(source),'-I'+headers,
                        '-L/usr/lib/aarch64-linux-gnu','-l:librealsense2.so.2.56','-o',str(target)],
                       check=True,capture_output=True,timeout=30)
    serial=json.loads(Path('/etc/config/camera_info.json').read_text())['right_wrist_rgbd']
    result=subprocess.run([str(target),serial],check=True,capture_output=True,text=True,timeout=10)
    return json.loads(result.stdout)


class Frames:
    def __init__(self,config_path=None):
        self.lock=threading.Lock()
        self.latest={}
        self.right=deque(maxlen=8)
        self.head=deque(maxlen=8)
        self.depth=deque(maxlen=8)
        self.result=None
        self.error=None
        self.measurement_error='正在读取相机出厂标定'
        self.measurement=None
        self.activation='等待相机'
        self.config_path=Path(config_path).expanduser() if config_path is not None else None
        self.point=None
        self.saved_point=None
        self.has_saved=False
        self.revision=0
        self.config_error=None
        if self.config_path and self.config_path.exists():
            try:
                data=json.loads(self.config_path.read_text())
                if not isinstance(data,dict) or data.get('schema')!='s1b-view-point-v1' or data.get('camera')!='right_wrist_rgbd':
                    raise ValueError('配置版本或相机不匹配')
                self.point=validate_point(data['point_normalized'])
                self.saved_point=self.point
                self.has_saved=True
            except (ValueError,KeyError,OSError,TypeError) as exc:
                self.config_error='未加载测距点配置：'+str(exc)

    def selection(self):
        return dict(point_normalized=self.point,revision=self.revision,
                    saved=self.has_saved and self.point==self.saved_point,
                    config_path=str(self.config_path) if self.config_path else None,error=self.config_error)

    def select_point(self,point):
        point=validate_point(point)
        with self.lock:
            self.point=point
            self.revision+=1
            self.result=None  # Never display the previous point's distance under a new marker.
            return self.selection()

    def save_point(self,revision):
        with self.lock:
            if type(revision) is not int or revision!=self.revision:
                raise ValueError('测距点已变更，请刷新后重新保存')
            if self.config_path is None:raise ValueError('未配置保存路径')
            self.config_path.parent.mkdir(parents=True,exist_ok=True)
            data=dict(schema='s1b-view-point-v1',camera='right_wrist_rgbd',point_normalized=self.point)
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(mode='w',dir=self.config_path.parent,delete=False) as out:
                    temporary=out.name
                    json.dump(data,out,allow_nan=False,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
                os.replace(temporary,self.config_path)
            finally:
                if temporary and os.path.exists(temporary):os.unlink(temporary)
            self.saved_point=self.point
            self.has_saved=True
            self.config_error=None
            return self.selection()

    def receive(self,key,message):
        try:
            fmt=message.format.lower()
            if key=='depth' and fmt!='raw':raise ValueError('unsupported_depth_format:'+fmt)
            if key!='depth' and 'jpeg' not in fmt:raise ValueError('unsupported_color_format:'+fmt)
            stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec
            if stamp<=0:raise ValueError('missing_camera_timestamp')
            frame=dict(data=bytes(message.data),source_ns=stamp,receipt_ns=time.monotonic_ns())
            with self.lock:
                old=self.latest.get(key)
                if old and stamp<=old['source_ns']:return
                self.latest[key]=frame
                self.error=None
                if key=='right':self.right.append(frame)
                if key=='head':self.head.append(frame)
                if key=='depth':self.depth.append(frame)
        except Exception as exc:
            self.error=str(exc)

    def update(self):
        with self.lock:
            if not self.right or not self.depth:return
            d=self.depth[-1]
            rgb=min(self.right,key=lambda f:abs(f['source_ns']-d['source_ns']))
            point,revision=self.point,self.revision
        skew=abs(rgb['source_ns']-d['source_ns'])
        if skew>80_000_000:
            self.measurement_error='右腕彩色与深度时间差过大'
            return
        if self.measurement is None:return
        if self.result and self.result['depth_source_ns']==d['source_ns']:return
        try:
            image=cv2.imdecode(np.frombuffer(rgb['data'],np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('invalid_right_jpeg')
            value=self.measurement.measure(d['data'],image.shape,point)
            value.update(depth_source_ns=d['source_ns'],rgb_source_ns=rgb['source_ns'],skew_ms=skew/1e6)
            with self.lock:
                if revision!=self.revision:return
                self.result=value
                self.paired=rgb
                self.paired_depth=d
                self.depth_receipt_ns=d['receipt_ns']
                self.measurement_error=None
        except Exception as exc:
            self.measurement_error=str(exc)

    def snapshot(self):
        now=time.monotonic_ns()
        with self.lock:
            result=dict(self.result) if self.result else dict(median_m=None,center_m=None,valid_pixels=0)
            selected={k:v for k,v in self.latest.items() if k in ('head','right')}
            if (self.result and not self.measurement_error
                    and now-self.depth_receipt_ns<=1_000_000_000):selected['right']=self.paired
            frames={k:dict(jpeg=base64.b64encode(v['data']).decode(),age_ms=(now-v['receipt_ns'])/1e6,
                           stale=now-v['receipt_ns']>1_000_000_000,source_ns=v['source_ns']) for k,v in selected.items()}
            stale=(not self.result or now-self.depth_receipt_ns>1_000_000_000
                   or 'right' not in frames or frames['right']['stale'])
            if stale or self.measurement_error:
                result.update(median_m=None,center_m=None,status=self.measurement_error or '深度已过期')
            return dict(frames=frames,depth=result,error=self.error,activation=self.activation,selection=self.selection())

    def capture(self):
        """Atomic, timestamped export for persisted observations; never substitutes old depth."""
        with self.lock:
            now=time.monotonic_ns()
            if not self.result or self.measurement_error or self.measurement is None:
                raise ValueError('depth_not_ready')
            selected={'right':self.paired, 'depth':self.paired_depth}
            if not self.head:raise ValueError('head_not_ready')
            # Match the retained head history to the paired wrist exposure,
            # rather than combining delayed wrist processing with latest head.
            selected['head']=min(self.head,key=lambda f:abs(f['source_ns']-self.paired_depth['source_ns']))
            if any(not 0 <= now-f['receipt_ns'] <= 1_000_000_000 for f in selected.values()):
                raise ValueError('stale_capture')
            if any(abs(time.time_ns()-f['source_ns'])>1_000_000_000 for f in selected.values()):
                raise ValueError('stale_capture_source')
            if max(f['source_ns'] for f in selected.values())-min(f['source_ns'] for f in selected.values())>80_000_000:
                raise ValueError('capture_skew')
            return dict(frames={k:dict(data_base64=base64.b64encode(f['data']).decode(),
                source_ns=f['source_ns'],receipt_ns=f['receipt_ns']) for k,f in selected.items()},
                depth=dict(self.result),metadata=self.measurement.m,selection=self.selection())


HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S1B · 双相机与定点测距</title><style>
*{box-sizing:border-box}body{margin:0;background:#101720;color:#e7eff7;font:16px system-ui,sans-serif;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;gap:20px}h1{font-size:23px;margin:0}small,.muted{color:#9cadbf}#status{color:#77d9c3}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:24px}article{background:#192331;border:1px solid #314154;border-radius:14px;overflow:hidden}h2{font-size:17px;padding:16px;margin:0}
.screen{position:relative;aspect-ratio:16/9;background:#080d13;display:flex;align-items:center;justify-content:center}.screen canvas{display:block;width:100%;height:100%;object-fit:contain}.empty{position:absolute;color:#8f9eb0}
footer{padding:14px 16px;color:#9cadbf}.depth{margin-top:20px;padding:22px;border-radius:14px;background:#192331;border:1px solid #314154;display:flex;align-items:center;gap:35px;flex-wrap:wrap}.value{font-size:44px;color:#65e6ce;font-variant-numeric:tabular-nums}.label{font-size:14px;color:#9cadbf}.bad{color:#ffbc80!important}p{line-height:1.7}.keyboard{margin-top:20px;color:#9cadbf;font-size:14px}kbd{background:#263547;padding:2px 6px;border-radius:4px}@media(max-width:800px){.grid{grid-template-columns:1fr}body{padding:14px}header{display:block}#status{margin-top:10px}}
.tools{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:16px}button{border:1px solid #456073;background:#233748;color:#e7eff7;border-radius:8px;padding:10px 16px;font:inherit;cursor:pointer}button.primary{background:#207465;border-color:#409d8c}button:disabled{opacity:.5;cursor:wait}#right{cursor:crosshair}.empty{pointer-events:none}
</style><header><div><h1>S1B / 实时视觉</h1><small>头部 · 右腕 · 定点测距</small></div><span id="status">连接中…</span></header>
<div class="grid"><article><h2>头部相机</h2><div class="screen"><canvas id="head"></canvas><span class="empty" id="headEmpty">等待画面</span></div><footer id="headAge">—</footer></article>
<article><h2>右腕相机 <small>点击或用方向键调整测距点</small></h2><div class="screen"><canvas id="right"></canvas><span class="empty" id="rightEmpty">等待画面</span></div><footer id="rightAge">—</footer></article></div>
<div class="tools"><button id="savePoint" class="primary" disabled>保存测距点</button><button id="resetPoint" disabled>恢复画面中心</button><span id="pointState" class="muted">等待右腕画面</span></div>
<p class="muted" id="pointKeys">测距点微调：网页获得焦点后，按 ↑ ↓ ← → 移动 1 像素；Shift＋方向键移动 10 像素。调整后点击“保存测距点”。</p>
<div class="depth"><div><div class="label">选定点周围 5 × 5 区域中值</div><div class="value" id="distance">—</div></div><div><div id="exact">选定像素：—</div><p class="muted" id="detail">等待深度</p></div></div>
<p class="muted">显示相机光轴方向的深度 Z，不是夹爪到目标的间隙。深度经设备出厂内外参映射到右腕彩色图；空洞和过期数据不作为有效距离。</p>
<div class="keyboard">键盘仍在终端操作：<kbd>W A S D</kbd> 平移　<kbd>F</kbd> 前伸　<kbd>R</kbd> 返回　<kbd>C</kbd> 闭合　<kbd>O</kbd> 打开　<kbd>空格</kbd> 停止。网页选点只改变测距位置，不控制机械臂。</div>
<script>
const $=id=>document.getElementById(id);let generation=0,selection={point_normalized:null,revision:-1},requesting=false,rightReady=false,rightImage=null,previewPoint=undefined,pendingPoint=undefined,actionError='';
function selectedPixel(width,height){const p=previewPoint===undefined?selection.point_normalized:previewPoint;return p?p.map((v,i)=>Math.round(v*([width,height][i]-1))):[Math.floor(width/2),Math.floor(height/2)];}
function paintRight(){if(!rightImage)return;const c=$('right');c.width=rightImage.width;c.height=rightImage.height;const ctx=c.getContext('2d');ctx.drawImage(rightImage,0,0);const [x,y]=selectedPixel(c.width,c.height);ctx.strokeStyle='#56ffe2';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x-15,y);ctx.lineTo(x+15,y);ctx.moveTo(x,y-15);ctx.lineTo(x,y+15);ctx.stroke();ctx.strokeRect(x-2,y-2,5,5);}
function selectionState(s){if(s.revision<selection.revision)return;selection=s;paintRight();let pos=rightImage?selectedPixel(rightImage.width,rightImage.height).join(', '):'—';$('pointState').textContent=actionError||s.error||('像素 ('+pos+') · '+(previewPoint!==undefined?'正在更新测距点…':s.saved?'已保存，下次启动自动恢复':'未保存'));$('pointState').className=actionError?'bad':'muted';$('savePoint').disabled=requesting||previewPoint!==undefined||!rightReady;$('resetPoint').disabled=!rightReady;}
function clearDistance(message){$('distance').textContent='—';$('exact').textContent='选定像素：—';$('detail').textContent=message;}
function draw(key,f,gen){if(!f)return;const im=new Image();im.onload=()=>{if(gen!==generation)return;if(key==='right'){rightImage=im;rightReady=true;paintRight();selectionState(selection);}else{const c=$(key);c.width=im.width;c.height=im.height;c.getContext('2d').drawImage(im,0,0);}$(key+'Empty').style.display='none';};im.src='data:image/jpeg;base64,'+f.jpeg;$(key+'Age').textContent=(f.stale?'画面已过期 · ':'')+'接收距今 '+Math.round(f.age_ms)+' ms';$(key+'Age').className=f.stale?'bad':'';}
async function pointRequest(path,body){if(requesting)return;requesting=true;++generation;actionError='';selectionState(selection);clearDistance('更新测距点…');try{const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(3000)});const data=await response.json();if(!response.ok)throw Error(data.error||'保存失败');selectionState(data);clearDistance('等待该测距点的新读数');}catch(e){actionError='操作失败：'+e.message+'；请再次选点或保存重试';clearDistance(actionError);}finally{requesting=false;if(pendingPoint===undefined)previewPoint=undefined;selectionState(selection);flushPoint();}}
function flushPoint(){if(requesting||pendingPoint===undefined)return;const point=pendingPoint;pendingPoint=undefined;pointRequest('/api/selection',{point_normalized:point});}
function choosePoint(point){previewPoint=point;pendingPoint=point;actionError='';++generation;paintRight();clearDistance('已选择新位置，等待深度更新');selectionState(selection);flushPoint();}
$('right').addEventListener('pointerdown',event=>{if(event.button!==0)return;event.preventDefault();if(!rightImage){actionError='等待右腕画面加载后再选点';selectionState(selection);return;}const c=$('right'),r=c.getBoundingClientRect(),scale=Math.min(r.width/c.width,r.height/c.height);const x=(event.clientX-r.left-(r.width-c.width*scale)/2)/scale,y=(event.clientY-r.top-(r.height-c.height*scale)/2)/scale;if(x<0||y<0||x>=c.width||y>=c.height){actionError='请点击实际画面区域，而非留黑边';selectionState(selection);return;}choosePoint([Math.floor(x)/Math.max(1,c.width-1),Math.floor(y)/Math.max(1,c.height-1)]);});
document.addEventListener('keydown',event=>{const offsets={ArrowUp:[0,-1],ArrowDown:[0,1],ArrowLeft:[-1,0],ArrowRight:[1,0]};const offset=offsets[event.key];if(!offset||event.ctrlKey||event.metaKey||event.altKey||event.isComposing)return;if(event.target instanceof Element && event.target.closest('input,textarea,select,[contenteditable="true"]'))return;event.preventDefault();if(!rightImage){actionError='等待右腕画面加载后再使用方向键';selectionState(selection);return;}const w=rightImage.width,h=rightImage.height,step=event.shiftKey?10:1;let [x,y]=selectedPixel(w,h);x=Math.max(0,Math.min(w-1,x+offset[0]*step));y=Math.max(0,Math.min(h-1,y+offset[1]*step));choosePoint([x/Math.max(1,w-1),y/Math.max(1,h-1)]);});
$('savePoint').onclick=()=>pointRequest('/api/selection/save',{revision:selection.revision});$('resetPoint').onclick=()=>choosePoint(null);
async function tick(){const requestedGeneration=generation;try{const response=await fetch('/api/snapshot',{cache:'no-store',signal:AbortSignal.timeout(2000)});if(!response.ok)throw Error('HTTP '+response.status);const s=await response.json();if(requesting||requestedGeneration!==generation||s.selection.revision<selection.revision)return;selectionState(s.selection);const gen=++generation;draw('head',s.frames.head,gen);draw('right',s.frames.right,gen);const d=s.depth,good=d.median_m!==null;$('distance').textContent=good?(d.median_m*1000).toFixed(1)+' mm':'—';$('exact').textContent='选定像素：'+(d.center_m===null?'无有效深度':(d.center_m*1000).toFixed(1)+' mm');$('detail').textContent=good?'有效像素 '+d.valid_pixels+'/'+d.window_pixels+' · 彩色/深度时间差 '+d.skew_ms.toFixed(1)+' ms':(d.status||'等待深度');$('status').textContent=s.error||(!s.frames.head||!s.frames.right?s.activation:good?'实时接收':'画面接收中 · 深度待就绪');$('status').className=good?'':'bad';}catch(e){++generation;rightReady=!!rightImage;selectionState(selection);$('status').textContent='连接中断，画面已停止更新';$('status').className='bad';clearDistance('等待重新连接');for(const key of ['head','right']){$(key+'Age').textContent='连接中断 · 画面已过期';$(key+'Age').className='bad';}}finally{setTimeout(tick,200);}}tick();
</script></html>'''


def serve(frames,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def setup(self):
            super().setup();self.connection.settimeout(2)
        def reply(self,status,payload):
            body=json.dumps(payload,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(body)
        def do_POST(self):
            try:
                host=self.headers.get('Host','')
                origin=self.headers.get('Origin')
                if urlsplit('http://'+host).hostname not in ('127.0.0.1','localhost','::1') or (origin and urlsplit(origin).netloc!=host):
                    self.reply(403,dict(error='仅允许本地网页修改测距点'));return
                if self.headers.get('Content-Type','').split(';')[0]!='application/json' or self.headers.get('Transfer-Encoding'):
                    raise ValueError('需要 JSON 请求')
                length=int(self.headers.get('Content-Length','-1'))
                if not 0<length<=1024:raise ValueError('请求长度无效')
                data=json.loads(self.rfile.read(length))
                if self.path=='/api/selection' and isinstance(data,dict) and set(data)=={'point_normalized'}:
                    self.reply(200,frames.select_point(data['point_normalized']))
                elif self.path=='/api/selection/save' and isinstance(data,dict) and set(data)=={'revision'}:
                    self.reply(200,frames.save_point(data['revision']))
                else:raise ValueError('请求路径或参数无效')
            except (ValueError,TypeError) as exc:self.reply(400,dict(error=str(exc)))
            except OSError as exc:self.reply(500,dict(error='配置保存失败：'+str(exc)))
        def do_GET(self):
            if self.path=='/api/capture':
                try:self.reply(200,frames.capture())
                except ValueError as exc:self.reply(503,dict(error=str(exc)))
                return
            if self.path=='/':body=HTML.encode();kind='text/html; charset=utf-8'
            elif self.path=='/health':body=json.dumps(dict(service='s1b-camera-view-v1',pid=os.getpid())).encode();kind='application/json'
            elif self.path=='/api/snapshot':body=json.dumps(frames.snapshot(),allow_nan=False).encode();kind='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server,thread


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--point-config',default='~/.config/pressb/right_wrist_depth_point.json')
    args=parser.parse_args()
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from std_srvs.srv import SetBool
    rclpy.init();node=rclpy.create_node('pressb_keyboard_camera_view')
    frames=Frames(args.point_config);done=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *a:done.set())
    server,thread=serve(frames,args.port)
    for key,topic in TOPICS.items():
        node.create_subscription(CompressedImage,topic,lambda m,k=key:frames.receive(k,m),qos_profile_sensor_data)
    def metadata():
        try:
            frames.measurement=CenterDepth(factory_metadata(Path(__file__).with_name('s1b_depth_metadata.cpp')))
            frames.measurement_error=None
        except Exception as exc:frames.measurement_error='深度标定读取失败：'+str(exc)
    meta=threading.Thread(target=metadata,daemon=True);meta.start()
    client=node.create_client(SetBool,'/astribot_camera_control');future=None
    started=time.monotonic();activation_tried=False;next_update=0
    try:
        while not done.is_set():
            rclpy.spin_once(node,timeout_sec=.02)
            if time.monotonic()-started>2 and not activation_tried:
                activation_tried=True
                if not frames.latest and client.service_is_ready():
                    request=SetBool.Request();request.data=True;future=client.call_async(request)
                    frames.activation='正在启动相机'
                else:frames.activation='等待相机画面（复用现有驱动）'
            if future is not None and future.done():
                response=future.result();frames.activation=response.message if response else '相机启动失败';future=None
            if time.monotonic()>=next_update:
                frames.update();next_update=time.monotonic()+.2
    finally:
        server.shutdown();server.server_close();thread.join(2)
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
