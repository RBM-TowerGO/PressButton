"""One SimulationApp / World per GPU, reusable cloned Astribot scenes."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import traceback
import numpy as np
from .contract import CAMERAS, JOINT_NAMES, atomic_json, sha256
from .video import VideoWriter
from .episode import episode

RENDER_SIZE=(1280,720)

def remap(value,prefix):
    if isinstance(value,str): return prefix+value[6:] if value.startswith('/World/') else value
    if isinstance(value,list): return [remap(v,prefix) for v in value]
    if isinstance(value,dict): return {k:remap(v,prefix) for k,v in value.items()}
    return value

def array(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)

class BatchRobot:
    """Named single-environment interface over shared batched state/control."""
    def __init__(self,runtime,index,single): self.runtime,self.index,self.single=runtime,index,single
    @property
    def dof_names(self): return self.single.dof_names
    @property
    def dof_properties(self):
        limits=array(self.runtime.view.get_dof_limits())[self.index]
        return dict(lower=limits[:,0],upper=limits[:,1])
    def get_joint_positions(self): return self.runtime.q[self.index]
    def get_joint_velocities(self): return self.runtime.qd[self.index]
    def set_joint_positions(self,q):
        self.single.set_joint_positions(self.runtime.tensor(q));self.runtime.q[self.index]=q
    def set_joint_velocities(self,q):
        self.single.set_joint_velocities(self.runtime.tensor(q));self.runtime.qd[self.index]=q
    def set_joints_default_state(self,positions,velocities):
        self.single.set_joints_default_state(positions=self.runtime.tensor(positions),velocities=self.runtime.tensor(velocities))
        self.runtime.defaults[self.index]=positions
    def get_articulation_controller(self): return self
    def apply_action(self,action): self.runtime.targets[self.index]=array(action.joint_positions)

class NumpyBody:
    def __init__(self,body): self.body=body;self.runtime=None;self.index=None
    def get_world_pose(self):
        if self.runtime is not None and self.runtime.body_positions is not None:
            return self.runtime.body_positions[self.index],self.runtime.body_orientations[self.index]
        return tuple(array(v) for v in self.body.get_world_pose())

class Runtime:
    def __init__(self,app,config,shared,num_envs,gpu,backend='cpu',render_subframes=1,render_layout='individual',asset_cache=None,render_cameras=None):
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, Articulation, RigidPrim
        from isaacsim.core.cloner import Cloner
        from pressb.astribot_asset import add_astribot
        from pressb.astribot_press import prepare_hand, complete_collision_geometry
        from pressb.astribot_sensors import create_robot_cameras, DEFAULT_CONFIG
        from pressb.vib820_scene import build_scene
        from .lights import prepare_feedback_materials
        from pressb.dataset_scene import build_tiled_rgb
        import omni.replicator.core as rep
        import omni.client
        omni.client.bypass_list_cache(True)
        self.app,self.gpu,self.backend,self.rep=app,gpu,backend,rep
        self.render_subframes=render_subframes
        self.render_layout=render_layout
        self.render_cameras=tuple(CAMERAS if render_cameras is None else render_cameras)
        if not self.render_cameras or len(set(self.render_cameras))!=len(self.render_cameras) or any(n not in CAMERAS for n in self.render_cameras):
            raise ValueError('render_cameras must contain unique known camera names')
        self.config=deepcopy(config);self.shared=Path(shared);self.shared.mkdir(parents=True,exist_ok=True)
        import carb
        settings=carb.settings.get_settings()
        render_controls={
            '/rtx-transient/post/aa/limitedOps':True,
            '/rtx/post/aa/op':3,
            # Auto mode chooses a lower per-camera quality as atlas size grows.
            '/rtx/post/dlss/execMode':2,
            '/rtx/post/histogram/enabled':False,
            '/rtx/post/tonemap/filmIso':100.,'/rtx/post/tonemap/exposureTime':.02,
            '/rtx/post/tonemap/fNumber':5.,'/rtx/post/tonemap/op':6,
        }
        previous={key:settings.get(key) for key in render_controls}
        for key,value in render_controls.items():settings.set(key,value)
        atomic_json(self.shared/'render_controls.json',dict(previous=previous,applied=render_controls,
            reason='Fixed exposure and DLSS Quality across atlas sizes; invalidate DLSS history at lamp transitions and resets.'))
        self.timings={};self.ik_pool=ThreadPoolExecutor(max_workers=min(num_envs,4),thread_name_prefix='astribot-ik')
        start=time.perf_counter()
        self.world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/30,
                         backend='torch' if backend=='gpu' else 'numpy',device=f'cuda:{gpu}' if backend=='gpu' else 'cpu')
        if backend=='gpu':
            # Suppressed readback makes legacy rigid-contact impulses zero in
            # this installation. Keep the CUDA solver and GPU broadphase, but
            # read back batched state/contact data so the unchanged audit works.
            settings.set_bool('/physics/suppressReadback',False)
        world=self.world; stage=world.stage
        # Build once using the original scene/geometry functions, then reference
        # the complete frozen /World; USD remaps internal joints/material links.
        scene=build_scene(stage,config);meta=add_astribot(stage)
        hand=prepare_hand(stage,meta);collision=complete_collision_geometry(stage,meta)
        sensor_cfg=json.loads(DEFAULT_CONFIG.read_text())
        camera_wrappers,robot_paths=create_robot_cameras(stage,meta['robot_path'],sensor_cfg)
        camera_paths={n:robot_paths[n+'_camera'] for n in camera_wrappers}
        camera_paths.update(overview=scene['camera_path'],**{n:scene[n+'_camera_path'] for n in ('panel','front','lobby')})
        # Kit's mixed-resolution tiled products do not preserve the individual
        # camera projection reliably. Render a single padded atlas and crop at
        # unchanged pixel density: fx=W*f/Ax remains the verified sensor fx.
        for name,path in camera_paths.items():
            camera=UsdGeom.Camera(stage.GetPrimAtPath(path));width,height=CAMERAS[name]
            original_horizontal=float(camera.GetHorizontalApertureAttr().Get())
            render_width,render_height=RENDER_SIZE if render_layout=='padded' else (width,height)
            camera.GetHorizontalApertureAttr().Set(original_horizontal*render_width/width)
            camera.GetVerticalApertureAttr().Set(original_horizontal*render_height/width)
        snapshot=self.shared/'scene.usda'
        stage.GetRootLayer().Export(str(snapshot))
        assets={}
        if asset_cache:
            from .assets import share_asset
            assets['scene.usda']=share_asset(snapshot,asset_cache)
        atomic_json(self.shared/'metadata.json',dict(meta=meta,scene=scene,hand=hand,collision=collision,
                    camera_paths=camera_paths,sensor_config=sensor_cfg,scene_sha256=sha256(snapshot)))
        if asset_cache:
            assets['metadata.json']=share_asset(self.shared/'metadata.json',asset_cache)
            atomic_json(self.shared/'assets.json',assets)
        else:
            assets={name:dict(sha256=sha256(self.shared/name),canonical=str((self.shared/name).resolve()),
                             bytes=(self.shared/name).stat().st_size) for name in ('scene.usda','metadata.json')}
        self.asset_references=assets
        stage.RemovePrim('/World');UsdGeom.Xform.Define(stage,'/World');UsdGeom.Scope.Define(stage,'/World/envs')
        if render_layout=='isolated':
            # The shared sky must remain visible while room roots are hidden.
            dome=stage.DefinePrim('/World/SharedDome','DomeLight')
            dome.GetReferences().AddReference(str(snapshot),'/World/Lights/Ambient')
        self.isolated_render_state={}
        self.envs=[]
        for index in range(num_envs):
            prefix=f'/World/envs/env_{index}'
            # A one-column grid keeps neighbouring walls out of each room's
            # forward view and avoids placing another wall behind the light rig.
            offset=np.array([0.,index*6.,0.])
            prim=UsdGeom.Xform.Define(stage,prefix)
            prim.GetPrim().GetReferences().AddReference(str(snapshot),'/World')
            prim.ClearXformOpOrder();prim.SetResetXformStack(False);prim.AddTranslateOp().Set(Gf.Vec3d(*offset))
            if render_layout=='isolated' and index:
                UsdGeom.Imageable(prim).MakeInvisible()
            for child in list(Usd.PrimRange(prim.GetPrim())):
                if child.IsA(UsdLux.DomeLight) and (index>0 or render_layout=='isolated'): child.SetActive(False)
                if child.IsA(UsdLux.RectLight):
                    light=UsdLux.LightAPI(child)
                    for collection in (light.GetLightLinkCollectionAPI(),light.GetShadowLinkCollectionAPI()):
                        collection.CreateIncludeRootAttr(False)
                        collection.CreateIncludesRel().SetTargets([prefix])
                if child.IsA(UsdPhysics.Joint):
                    joint=UsdPhysics.Joint(child)
                    for relation,attr in ((joint.GetBody0Rel(),joint.GetLocalPos0Attr()),(joint.GetBody1Rel(),joint.GetLocalPos1Attr())):
                        if not relation.GetTargets(): attr.Set(Gf.Vec3f(*(np.asarray(attr.Get())+offset)))
                        elif any(not str(p).startswith(prefix+'/') for p in relation.GetTargets()):
                            raise RuntimeError('Joint escaped cloned environment')
            emeta,escene=remap(meta,prefix),remap(scene,prefix)
            for info in escene['buttons'].values():
                for field in ('center','body_center'): info[field]=(np.asarray(info[field])+offset).tolist()
                info['rest_x']+=float(offset[0])
                prepare_feedback_materials(stage,info,prefix)
            single=world.scene.add(SingleArticulation(prim_path=emeta['robot_path'],name=f'robot_{index}'))
            def rigid(path,name): return NumpyBody(world.scene.add(SingleRigidPrim(prim_path=path,name=f'{name}_{index}')))
            env=SimpleNamespace(index=index,prefix=prefix,offset=offset,world=world,meta=emeta,scene=escene,hand=hand,
                single=single,wrist=rigid(emeta['wrist_links']['right'],'wrist'),base_body=rigid(emeta['base_link'],'base'),
                buttons={f:rigid(b['body_path'],f'button_{f}') for f,b in escene['buttons'].items()},
                finger_bodies={s:rigid(emeta['robot_path']+'/astribot_gripper_right_Link_'+s,'finger_'+s) for s in ('L11','R11')},
                camera_paths=remap(camera_paths,prefix),ik_pool=self.ik_pool)
            self.envs.append(env)
        self.all_envs=tuple(self.envs)
        Cloner(stage=stage).filter_collisions(world.get_physics_context().prim_path,'/World/CollisionGroups',[e.prefix for e in self.envs],[])
        self.view=world.scene.add(Articulation([e.meta['robot_path'] for e in self.envs],name='batch',reset_xform_properties=False))
        bodies=[b for e in self.envs for b in [e.wrist,e.base_body,*e.buttons.values(),*e.finger_bodies.values()]]
        self.body_view=RigidPrim([b.body.prim_path for b in bodies],name='batch_body_readback',reset_xform_properties=False)
        self.body_positions=None;self.body_orientations=None
        body_indices={path:index for index,path in enumerate(self.body_view.prim_paths)}
        for body in bodies:body.runtime=self;body.index=body_indices[body.body.prim_path]
        from .contacts import ContactRouter
        from pxr import PhysicsSchemaTools
        import omni.physx
        self.contact_router=ContactRouter([e.prefix for e in self.envs],PhysicsSchemaTools.intToSdfPath)
        self.contact_subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.contact_router.dispatch)
        for e in self.envs:
            e.subscribe_contacts=lambda callback,index=e.index:self.contact_router.subscribe(index,callback)
        self.products=[]
        self.clock_annotators=[]
        self.camera_param_annotators=[]
        self.readback_buffers={}
        self.rgb_device_buffers={}
        self.last_light_states=None
        rep.orchestrator.set_capture_on_play(False)
        entries=[(e,n) for e in self.envs for n in self.render_cameras]
        if render_layout=='padded':
            product,annotator,split=build_tiled_rgb([e.camera_paths[n] for e,n in entries],resolution=RENDER_SIZE,device=f'cuda:{gpu}')
            def crop_tiles(buffer):
                result=[]
                for (_,name),tile in zip(entries,split(buffer)):
                    width,height=CAMERAS[name];x=(RENDER_SIZE[0]-width)//2;y=(RENDER_SIZE[1]-height)//2
                    result.append(np.ascontiguousarray(tile[y:y+height,x:x+width]))
                return result
            self.products.append((product,annotator,crop_tiles,entries))
        elif render_layout=='homogeneous':
            groups={}
            for e,name in entries:
                camera=UsdGeom.Camera(stage.GetPrimAtPath(e.camera_paths[name]))
                signature=(*CAMERAS[name],float(camera.GetFocalLengthAttr().Get()),
                    float(camera.GetHorizontalApertureAttr().Get()),float(camera.GetVerticalApertureAttr().Get()),
                    *map(float,camera.GetClippingRangeAttr().Get()))
                groups.setdefault(signature,[]).append((e,name))
            for signature,group in groups.items():
                product,annotator,split=build_tiled_rgb([e.camera_paths[n] for e,n in group],
                    resolution=signature[:2],device=f'cuda:{gpu}')
                self.products.append((product,annotator,split,group))
        else:
            # Persistent native products retain the verified camera projections.
            # All products render in one GPU orchestrator step at the same time.
            for e,name in entries:
                product=rep.create.render_product(e.camera_paths[name],CAMERAS[name],force_new=True)
                if render_layout=='isolated':product.hydra_texture.set_updates_enabled(e.index==0)
                annotator=rep.AnnotatorRegistry.get_annotator('rgb',device=f'cuda:{gpu}',do_array_copy=False)
                annotator.attach([product.path])
                self.products.append((product,annotator,lambda a:[np.ascontiguousarray(a[:,:,:3])],[(e,name)]))
        for product,_,_,_ in self.products:
            clock=rep.AnnotatorRegistry.get_annotator('ReferenceTime')
            clock.attach([product.path]);self.clock_annotators.append(clock)
            if render_layout in ('individual','homogeneous','isolated'):
                params=rep.AnnotatorRegistry.get_annotator('CameraParams')
                params.attach([product.path]);self.camera_param_annotators.append(params)
        self.q=np.zeros((num_envs,32),np.float32);self.qd=self.q.copy();self.targets=self.q.copy();self.defaults=self.q.copy()
        for e in self.envs:e.robot=BatchRobot(self,e.index,e.single)
        world.reset();self.refresh()
        for e in self.envs:
            if tuple(e.single.dof_names)!=JOINT_NAMES: raise RuntimeError('Model joint order changed; remap before collecting')
        world.get_physics_context().set_physx_update_transformations_settings(update_to_usd=False,update_velocities_to_usd=False)
        self.timings['startup_s']=time.perf_counter()-start
        phys=world.get_physics_context()
        self.actual_backend=dict(requested=backend,device=str(world.device),tensor_backend=world.backend,
            gpu_dynamics=phys.is_gpu_dynamics_enabled(),broadphase=phys.get_broadphase_type(),
            suppress_readback=settings.get('/physics/suppressReadback'))
        if backend=='gpu' and (not self.actual_backend['gpu_dynamics'] or self.actual_backend['broadphase']!='GPU'):
            raise RuntimeError(f'GPU backend did not take effect: {self.actual_backend}')
        atomic_json(self.shared/'backend.json',self.actual_backend)
        atomic_json(self.shared/'render_layout.json',dict(mode=render_layout,environments=num_envs,
            render_cameras=list(self.render_cameras),
            products=len(self.products),environment_offsets_m=[e.offset.tolist() for e in self.envs],
            visibility_policy='one room visible, seven active native products, frozen physics' if render_layout=='isolated' else 'all rooms visible',
            rgb_transfer='owned CUDA arrays, GPU RGB packing, pinned host batches, bounded NVENC queues'))
        self.render()

    def tensor(self,value):
        if self.backend=='cpu': return np.asarray(value,dtype=np.float32)
        import torch
        return torch.as_tensor(np.asarray(value),dtype=torch.float32,device=self.world.device)
    def refresh(self):
        self.q=array(self.view.get_joint_positions()).copy();self.qd=array(self.view.get_joint_velocities()).copy()
        if not self.body_view.is_physics_handle_valid():self.body_view.initialize()
        positions,orientations=self.body_view.get_world_poses()
        self.body_positions=array(positions).copy();self.body_orientations=array(orientations).copy()
    def render(self):
        if self.render_layout!='isolated':return self._render_current()
        from pxr import UsdGeom
        # Experimental: physics stays frozen across all room renders. Each room
        # has seven products and owned buffers; continuous-image/renderer-history
        # acceptance is required separately and has not yet passed.
        envs,products,clocks=self.envs,self.products,self.clock_annotators
        cache_names=('readback_buffers','rgb_device_buffers','last_light_states','recorded_buffer_diagnostic')
        original={name:getattr(self,name,False if name=='recorded_buffer_diagnostic' else None) for name in cache_names}
        force=getattr(self,'force_history_flush',False);images={};times={}
        try:
            for env in envs:
                for other in envs:
                    visibility='inherited' if other.index==env.index else 'invisible'
                    UsdGeom.Imageable(self.world.stage.GetPrimAtPath(other.prefix)).GetVisibilityAttr().Set(visibility)
                visible=[other.index for other in envs if
                    UsdGeom.Imageable(self.world.stage.GetPrimAtPath(other.prefix)).ComputeVisibility()!='invisible']
                if visible!=[env.index]:raise RuntimeError(f'Environment visibility isolation failed: {visible}')
                indices=[]
                for index,(product,_,_,entries) in enumerate(products):
                    active=entries[0][0].index==env.index
                    product.hydra_texture.set_updates_enabled(active)
                    if active:indices.append(index)
                self.envs=[env];self.products=[products[i] for i in indices]
                self.clock_annotators=[clocks[i] for i in indices]
                state=self.isolated_render_state.setdefault(env.index,dict(readback_buffers={},rgb_device_buffers={},
                    last_light_states=None,recorded_buffer_diagnostic=False))
                for name in cache_names:setattr(self,name,state[name])
                self.force_history_flush=force
                images.update(self._render_current());times.update(self.camera_times)
                for name in cache_names:state[name]=getattr(self,name)
        finally:
            self.envs,self.products,self.clock_annotators=envs,products,clocks
            for name,value in original.items():setattr(self,name,value)
        self.force_history_flush=False
        self.camera_times=times;self.last_images=images
        return images

    def _render_current(self):
        import omni.physx
        world=self.world;t=float(world.current_time);started=time.perf_counter()
        world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False,True,True)
        states=tuple(bool(world.stage.GetPrimAtPath(e.scene['buttons'][0]['body_path']).GetAttribute('pressb:illuminated').Get()) for e in self.envs)
        subframes=self.render_subframes
        if self.last_light_states is None or states!=self.last_light_states or getattr(self,'force_history_flush',False):
            import omni.usd
            import carb
            omni.usd.get_context().reset_renderer_accumulation()
            # Kit 107 requires DLSS Ray Reconstruction. Temporarily invalidate
            # its history, then restore the frozen quality mode before capture.
            # The invalidation image is never recorded and physics stays frozen.
            settings=carb.settings.get_settings()
            settings.set('/rtx-transient/post/aa/limitedOps',False)
            settings.set('/rtx/post/aa/op',0)
            for _ in range(3):self.rep.orchestrator.step(delta_time=0.,pause_timeline=False,rt_subframes=2)
            settings.set('/rtx/post/aa/op',3)
            settings.set('/rtx-transient/post/aa/limitedOps',True)
            for _ in range(4):self.rep.orchestrator.step(delta_time=0.,pause_timeline=False,rt_subframes=1)
            self.force_history_flush=False
            subframes=1
        self.last_light_states=states
        # The orchestrator waits for Synthetic Data products, including their
        # device buffers. World.render alone can expose stale/recycled pointers
        # even when ReferenceTime already reflects the current physics time.
        self.rep.orchestrator.step(delta_time=0.,pause_timeline=False,rt_subframes=subframes)
        if not getattr(self,'recorded_effective_render_controls',False):
            import carb
            controls=json.loads((self.shared/'render_controls.json').read_text())
            controls['effective_after_render']={key:carb.settings.get_settings().get(key) for key in controls['applied']}
            atomic_json(self.shared/'render_controls.json',controls)
            from .render_controls import setting_matches
            if any(not setting_matches(controls['effective_after_render'][key],value) for key,value in controls['applied'].items()):
                raise RuntimeError('Renderer overrode frozen collection controls; see render_controls.json')
            self.recorded_effective_render_controls=True
        if abs(float(world.current_time)-t)>1e-7: raise RuntimeError('Render advanced physics')
        images={e.index:{} for e in self.envs}
        self.camera_times={e.index:{} for e in self.envs}
        import torch
        import warp as wp
        torch.cuda.set_device(self.gpu)
        pending=[];readback_start=time.perf_counter()
        from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
        for attempt in range(3):
            pending=[]
            try:
                for group,(_,annotator,split,entries) in enumerate(self.products):
                    reference=self.clock_annotators[group].get_data()
                    timestamp=float(_isaacsim_core_nodes.acquire_interface().get_sim_time_at_time(
                        (reference['referenceTimeNumerator'],reference['referenceTimeDenominator'])))
                    if abs(timestamp-t)>1e-6:raise RuntimeError('Capture product not ready')
                    # Keep an owned CUDA array while other products are read.
                    # Borrowed Synthetic Data buffers can be recycled between
                    # products (observed as panel/front stream aliasing).
                    buffer=annotator.get_data(device=f'cuda:{self.gpu}',do_array_copy=True)
                    if isinstance(buffer,dict):buffer=buffer['data']
                    if not buffer.size or not buffer.ptr:raise RuntimeError('Capture product not ready')
                    pending.append(wp.to_torch(buffer))
                break
            except RuntimeError as exc:
                if attempt==2 or not any(s in str(exc) for s in
                    ('Capture product not ready','pointer resides on host memory')):raise
                self.timings['render_readiness_retries']=self.timings.get('render_readiness_retries',0)+1
                # Products can finish on different updates. Freeze physics and
                # wait for a fresh complete capture; never reuse an old image.
                self.rep.orchestrator.step(delta_time=0.,pause_timeline=False,rt_subframes=1)
                if abs(float(world.current_time)-t)>1e-7:raise RuntimeError('Capture retry advanced physics')
        # Synthetic Data produces CUDA buffers on a different stream from
        # PyTorch's pinned-memory copy. A fence only AFTER the copy does not
        # establish producer-before-consumer ordering and can read an old image.
        for device in {t.device for t in pending}:torch.cuda.synchronize(device)
        for group,device_tensor in enumerate(pending):
            if self.render_layout in ('individual','homogeneous','isolated'):
                from .gpu_pixels import pack_rgb
                entries=self.products[group][3];width,height=CAMERAS[entries[0][1]]
                if group not in self.rgb_device_buffers:
                    self.rgb_device_buffers[group]=torch.empty((len(entries),height,width,3),
                        dtype=torch.uint8,device=device_tensor.device)
                columns=math.ceil(math.sqrt(len(entries))) if self.render_layout=='homogeneous' else 1
                pack_rgb(device_tensor,self.rgb_device_buffers[group],columns)
                transfer=self.rgb_device_buffers[group]
            else:transfer=device_tensor
            if group not in self.readback_buffers:
                self.readback_buffers[group]=torch.empty(transfer.shape,dtype=transfer.dtype,pin_memory=True)
            self.readback_buffers[group].copy_(transfer,non_blocking=True)
        for device in {t.device for t in pending}:torch.cuda.synchronize(device)
        self.timings['readback_s']=self.timings.get('readback_s',0.)+time.perf_counter()-readback_start
        for group,((_,annotator,split,entries),clock) in enumerate(zip(self.products,self.clock_annotators)):
            from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
            reference=clock.get_data()
            timestamp=float(_isaacsim_core_nodes.acquire_interface().get_sim_time_at_time(
                (reference['referenceTimeNumerator'],reference['referenceTimeDenominator'])))
            if abs(timestamp-t)>1e-6:raise RuntimeError(f'Stale rendered clock {timestamp}, physics {t}')
            host=self.readback_buffers[group].numpy()
            tiles=host if self.render_layout in ('individual','homogeneous','isolated') else split(host)
            # Views live until the next render. Writers copy into owned bounded
            # queue buffers before returning; snapshots are saved synchronously.
            for (e,name),rgb in zip(entries,tiles):
                sample=rgb[::16,::16]
                if float(sample.mean())<5 or float(sample.std())<1:
                    raise RuntimeError(f'Invalid blank image: environment {e.index}, camera {name}')
                images[e.index][name]=rgb
                self.camera_times[e.index][name]=timestamp
        if not getattr(self,'recorded_buffer_diagnostic',False):
            from PIL import Image
            diagnostic=[]
            for group,(_,annotator,split,entries) in enumerate(self.products):
                cpu=annotator.get_data(device='cpu')
                if isinstance(cpu,dict):cpu=cpu['data']
                host=self.readback_buffers[group].numpy()
                cpu_tiles=split(cpu)
                host_tiles=host if self.render_layout in ('individual','homogeneous','isolated') else split(host)
                difference=max(int(np.max(np.abs(a.astype(int)-b.astype(int)))) for a,b in zip(cpu_tiles,host_tiles))
                if difference:raise RuntimeError('GPU RGB packing changed camera pixels')
                diagnostic.append(dict(group=group,device=str(pending[group].device),shape=list(host.shape),
                    cpu_mean=float(np.mean([a.mean() for a in cpu_tiles])),cuda_mean=float(np.mean(host)),
                    max_cpu_cuda_difference=difference))
                for (e,name),rgb in zip(entries,cpu_tiles):
                    Image.fromarray(rgb).save(self.shared/f'initial_cpu_env{e.index}_{name}.png')
            diagnostic_name=f'buffer_diagnostic_env{self.envs[0].index}.json' if self.render_layout=='isolated' else 'buffer_diagnostic.json'
            atomic_json(self.shared/diagnostic_name,diagnostic)
            self.recorded_buffer_diagnostic=True
        self.timings['render_s']=self.timings.get('render_s',0.)+time.perf_counter()-started
        self.last_images=images
        return images
    def prepare_wave(self,configs,outputs):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from .lights import set_button_light
        self.world.stop()
        base_positions=[array(e.single.get_default_state().position) for e in self.envs]
        base_orientations=[array(e.single.get_default_state().orientation) for e in self.envs]
        for e,cfg,out in zip(self.envs,configs,outputs):
            e.config=deepcopy(cfg);e.output=Path(out);e.output.mkdir(parents=True,exist_ok=False)
            e.report=dict(status='running',backend=self.actual_backend,shared=str(self.shared),
                shared_assets=self.asset_references,
                render_configuration=dict(layout=self.render_layout,environments=len(self.envs),
                    visibility_isolated=self.render_layout=='isolated' or len(self.envs)==1))
            e.video_frames={};e.camera_timing=[];e.writers={};e.started=time.perf_counter()
            local=np.array([cfg['robot_base_x'],cfg['robot_base_y'],e.meta['base_lift_m']]);yaw=cfg['robot_base_yaw_rad']
            xf=UsdGeom.Xformable(self.world.stage.GetPrimAtPath(e.prefix+'/Astribot'))
            xf.ClearXformOpOrder()
            # Existing ops are reused; no accumulating transform op suffixes.
            xf.AddTranslateOp().Set(Gf.Vec3d(*local));xf.AddRotateZOp().Set(float(np.degrees(yaw)))
            root=UsdPhysics.FixedJoint(self.world.stage.GetPrimAtPath(e.meta['articulation_path']))
            root.GetLocalPos0Attr().Set(Gf.Vec3f(*(local+e.offset)))
            root.GetLocalRot0Attr().Set(Gf.Quatf(float(np.cos(yaw/2)),Gf.Vec3f(0,0,float(np.sin(yaw/2)))))
            base_positions[e.index]=local+e.offset
            base_orientations[e.index]=[np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
            e.single.set_default_state(position=self.tensor(base_positions[e.index]),orientation=self.tensor(base_orientations[e.index]))
            e.base_body.body.set_default_state(position=self.tensor(base_positions[e.index]),orientation=self.tensor(base_orientations[e.index]))
            for f,b in e.buttons.items():
                info=e.scene['buttons'][f]
                b.body.set_default_state(position=self.tensor(info['body_center']),orientation=self.tensor([1,0,0,0]),
                    linear_velocity=self.tensor([0,0,0]),angular_velocity=self.tensor([0,0,0]))
                set_button_light(self.world.stage,info,False)
            q=np.array([cfg['initial_arm_positions'].get(n,0.) for n in JOINT_NAMES],np.float32)
            q[JOINT_NAMES.index('astribot_head_joint_2')]=np.deg2rad(30.)
            e.robot.set_joints_default_state(q,np.zeros_like(q));self.targets[e.index]=q
        self.view.set_default_state(positions=self.tensor(base_positions),orientations=self.tensor(base_orientations))
        self.view.set_joints_default_state(positions=self.tensor(self.defaults),velocities=self.tensor(np.zeros_like(self.defaults)))
        start=time.perf_counter();self.world.reset();self.refresh()
        self.view.set_joint_positions(self.tensor(self.targets));self.view.set_joint_velocities(self.tensor(np.zeros_like(self.targets)))
        self.view.set_joint_position_targets(self.tensor(self.targets));self.refresh()
        for e in self.envs[:len(configs)]:
            calibration={}
            for name,path in e.camera_paths.items():
                camera=UsdGeom.Camera(self.world.stage.GetPrimAtPath(path));width,height=CAMERAS[name]
                focal=float(camera.GetFocalLengthAttr().Get())
                render_width,render_height=RENDER_SIZE if self.render_layout=='padded' else (width,height)
                fx=render_width*focal/float(camera.GetHorizontalApertureAttr().Get())
                fy=render_height*focal/float(camera.GetVerticalApertureAttr().Get())
                transform=np.asarray(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                calibration[name]=dict(resolution=[width,height],render_resolution=[render_width,render_height],
                    crop_origin=[(render_width-width)//2,(render_height-height)//2],
                    intrinsics=[[fx,0,width/2],[0,fy,height/2],[0,0,1]],
                    camera_path=path,parent_path=str(camera.GetPrim().GetParent().GetPath()),
                    mount_transform=np.asarray(UsdGeom.Xformable(camera).GetLocalTransformation()).T.tolist(),
                    initial_world_transform=transform.tolist(),coordinate_convention='USD camera: +X right, +Y up, -Z forward')
                if name=='panel':
                    center=np.array(e.scene['buttons'][0]['center']);radius=e.config['button_outer_diameter']/2-.0008
                    angles=np.linspace(0,2*np.pi,128,endpoint=False)
                    points=np.c_[np.full(128,center[0]-.0001),center[1]+radius*np.cos(angles),center[2]+radius*np.sin(angles),np.ones(128)]
                    local=points@np.linalg.inv(transform).T
                    uv=np.c_[width/2+fx*local[:,0]/-local[:,2],height/2-fy*local[:,1]/-local[:,2]]
                    calibration[name]['light_ring_pixels']=np.rint(uv).astype(int).tolist()
            atomic_json(e.output/'calibration.json',dict(cameras=calibration,environment_offset_m=e.offset.tolist()))
        self.timings['reset_s']=self.timings.get('reset_s',0.)+time.perf_counter()-start
        import omni.usd
        omni.usd.get_context().reset_renderer_accumulation()
        self.force_history_flush=True
        self.render()
        if self.render_layout in ('individual','homogeneous','isolated'):
            # Validate against renderer-provided projection, not USD assumptions.
            calibrations={e.index:json.loads((e.output/'calibration.json').read_text()) for e in self.envs[:len(configs)]}
            for (_,_,_,entries),annotator in zip(self.products,self.camera_param_annotators):
                native=annotator.get_data();projection=np.asarray(native['cameraProjection']).reshape(4,4)
                for e,name in entries:
                    if e.index not in calibrations:continue
                    width,height=CAMERAS[name]
                    intrinsic=np.array([[width*projection[0,0]/2,0,width*(1-projection[2,0])/2],
                                        [0,height*projection[1,1]/2,height*(1+projection[2,1])/2],[0,0,1]])
                    c=calibrations[e.index]['cameras'][name]
                    if not np.allclose(intrinsic,c['intrinsics'],atol=.001,rtol=0):
                        raise RuntimeError(f'Rendered camera projection mismatch: environment {e.index}, {name}')
                    transform=np.asarray(UsdGeom.Xformable(self.world.stage.GetPrimAtPath(e.camera_paths[name])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                    c.update(native_projection_matrix=projection.tolist(),native_projection_verified=True,
                        projection_group_camera_count=len(entries),initial_world_transform=transform.tolist())
            for e in self.envs[:len(configs)]:atomic_json(e.output/'calibration.json',calibrations[e.index])

    def run_wave(self,configs,outputs,*,physics_only=False,replay_sources=None):
        self.prepare_wave(configs,outputs)
        for e in self.envs[:len(configs)]:
            e.physics_only=physics_only
            e.replay_source=Path(replay_sources[e.index]) if replay_sources and replay_sources[e.index] else None
            e.report['capture_scope']='physics_only_preflight' if physics_only else 'seven_continuous_videos'
        generators={e.index:episode(e) for e in self.envs[:len(configs)]}; requests={}
        def advance(index,reset_completed=None):
            try: requests[index]=generators[index].send(reset_completed)
            except StopIteration: requests.pop(index,None);generators.pop(index,None)
            except BaseException as exc:
                e=self.envs[index];e.report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                requests.pop(index,None);generators.pop(index,None)
        for i in list(generators):advance(i)
        while generators:
            # Nonphysics barriers are all processed while the world is frozen.
            others=[i for i,r in requests.items() if r[0] not in ('step','reset')]
            if others:
                need_render=any(requests[i][0] in ('capture','snapshot') for i in others)
                images=self.render() if need_render else None
                for i in others:
                    kind,value=requests[i];e=self.envs[i]
                    if kind=='ik':
                        t=time.perf_counter();wait([value]);self.timings['ik_wait_s']=self.timings.get('ik_wait_s',0.)+time.perf_counter()-t
                    elif kind=='snapshot':
                        from PIL import Image
                        for n,rgb in images[i].items():Image.fromarray(rgb).save(e.output/f'{value}_{n}.png')
                    elif kind=='capture':
                        for n,size in CAMERAS.items():
                            if n not in e.writers:e.writers[n]=VideoWriter(e.output/f'{n}.mp4',size,self.gpu)
                            e.writers[n].write(images[i][n]);e.video_frames[n]=e.writers[n].frames
                        value['cameras']=self.camera_times[i].copy()
                        e.camera_timing.append(value)
                    else:raise RuntimeError(f'Unknown barrier: {kind}')
                    advance(i)
                continue
            if all(r[0]=='reset' for r in requests.values()):
                self.view.set_joints_default_state(positions=self.tensor(self.defaults),velocities=self.tensor(np.zeros_like(self.defaults)))
                self.world.reset();self.refresh()
                self.view.set_joint_position_targets(self.tensor(self.targets))
                for i in list(generators):advance(i,True)
                continue
            # Finished environments still record every physical tick and camera
            # sample while peers finish. The wave reset is one shared barrier.
            for i in [i for i,r in requests.items() if r[0]=='reset']:advance(i,False)
            for i,(kind,value) in requests.items():
                if kind=='step':self.targets[i]=value
            t=time.perf_counter()
            self.view.set_joint_position_targets(self.tensor(self.targets))
            self.world.step(render=False);self.refresh()
            self.timings['physics_s']=self.timings.get('physics_s',0.)+time.perf_counter()-t
            for i in [i for i,r in requests.items() if r[0]=='step']:advance(i)
        results=[]
        for e in self.envs[:len(configs)]:
            try:
                for writer in e.writers.values():writer.close()
            except Exception as exc:e.report.update(status='failed',video_error=str(exc))
            e.report['elapsed_s']=time.perf_counter()-e.started
            e.report['encoding']={n:dict(frames=w.frames,backpressure_s=w.wait_s) for n,w in e.writers.items()}
            atomic_json(e.output/'report.json',e.report);results.append(e.report)
        atomic_json(self.shared/'timings.json',self.timings)
        return results
    def close(self):
        self.contact_subscription=None
        self.contact_router.callbacks.clear()
        self.ik_pool.shutdown(wait=True,cancel_futures=True)
        for e in self.envs:
            for writer in getattr(e,'writers',{}).values():writer.close()

    def probe_resets(self,config,count,output):
        """Exercise reset with deliberately dirty velocities and latched lamps."""
        from .lights import set_button_light
        from pressb.astribot_validation import initial_overlap_check
        output=Path(output);rows=[];dark_reference={}
        for iteration in range(count):
            if iteration:
                self.view.set_joint_velocities(self.tensor(np.full_like(self.q,.1)))
                for e in self.envs:set_button_light(self.world.stage,e.scene['buttons'][0],True)
                self.render()
            self.prepare_wave([config]*len(self.envs),[output/f'round_{iteration:02d}'/f'env_{e.index}' for e in self.envs])
            self.refresh()
            q_error=float(np.max(np.abs(self.q-self.targets)));qd=float(np.max(np.abs(self.qd)))
            clear=all(initial_overlap_check(self.world.stage,e.meta,floor_path=e.prefix+'/Environment/Floor')['status']=='clear' for e in self.envs)
            released=all(abs(e.buttons[0].get_world_pose()[0][0]-e.scene['buttons'][0]['rest_x'])<.0003 for e in self.envs)
            dark=all(not self.world.stage.GetPrimAtPath(e.scene['buttons'][0]['body_path']).GetAttribute('pressb:illuminated').Get() for e in self.envs)
            rendered_dark=True;ring_deltas={};ring_bright_counts={}
            for e in self.envs:
                c=json.loads((e.output/'calibration.json').read_text())['cameras']['panel']
                uv=np.unique(np.asarray(c['light_ring_pixels'],int),axis=0)
                values=self.last_images[e.index]['panel'][uv[:,1],uv[:,0]].astype(float).mean(axis=1)
                panel=self.last_images[e.index]['panel']
                if float(panel.mean())<5 or float(panel.std())<1:
                    raise RuntimeError(f'Invalid blank panel image in environment {e.index}')
                if e.index not in dark_reference:dark_reference[e.index]=values.copy()
                delta=values-dark_reference[e.index]
                ring_deltas[str(e.index)]=float(np.mean(delta))
                ring_bright_counts[str(e.index)]=int((delta>=20).sum())
                rendered_dark=rendered_dark and int((delta>=20).sum())<8 and abs(float(delta.mean()))<10
                from PIL import Image
                Image.fromarray(self.last_images[e.index]['panel']).save(e.output/'reset_panel.png')
            row=dict(round=iteration,joint_error_rad=q_error,joint_velocity_max_rad_s=qd,initial_overlap_clear=clear,
                     buttons_released=bool(released),lights_cleared=dark,rendered_lights_cleared=bool(rendered_dark),ring_mean_delta=ring_deltas,
                     ring_bright_counts=ring_bright_counts,
                     passed=bool(q_error<1e-5 and qd<1e-5 and clear and released and dark and rendered_dark))
            rows.append(row);atomic_json(output/'reset_audit.json',dict(status='passed' if all(r['passed'] for r in rows) and len(rows)>=20 else 'incomplete',rounds=rows))
            if not row['passed']:raise RuntimeError(f'Reset contamination: {row}')
        return rows
