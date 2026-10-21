"""Explicit PhysX-to-Fabric synchronization for live closed-loop observations."""
import numpy as np
from scipy.spatial.transform import Rotation
from pressb.astribot_collection.runtime import Runtime, array


class LiveRuntime(Runtime):
    def __init__(self,*args,**kwargs):
        self.pose_guard=None
        self.pose_sync_retries=0
        super().__init__(*args,**kwargs)
        from pxr import UsdGeom,UsdPhysics
        from isaacsim.core.prims import RigidPrim
        self.camera_mounts={};paths=[]
        for env in self.envs:
            cache=UsdGeom.XformCache()
            for name,path in env.camera_paths.items():
                camera=self.world.stage.GetPrimAtPath(path);body=camera.GetParent()
                while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
                if not body or body.IsPseudoRoot():continue
                body_path=str(body.GetPath())
                if body_path not in paths:paths.append(body_path)
                world_body=np.asarray(cache.GetLocalToWorldTransform(body)).T
                world_camera=np.asarray(cache.GetLocalToWorldTransform(camera)).T
                self.camera_mounts[(env.index,name)]=(body_path,np.linalg.inv(world_body)@world_camera)
        self.guard_paths=paths
        self.pose_guard=RigidPrim(paths,name='live_camera_pose_guard',reset_xform_properties=False)
        self.pose_guard.initialize()
        self.max_pose_sync_error=0.

    def _render_current(self):
        from omni.physxfabric import get_physx_fabric_interface
        import torch
        import json
        frozen_time=float(self.world.current_time)
        torch.cuda.synchronize(self.gpu)
        frozen_q=array(self.view.get_joint_positions()).copy()
        rejected=[]
        for attempt in range(3):
            # Publish current physical poses, then fence before the capture.
            # Retrying a stale render must never advance physical time/state.
            self.world.physics_sim_view.update_articulations_kinematic()
            get_physx_fabric_interface().force_update(1/120,frozen_time)
            torch.cuda.synchronize(self.gpu)
            images=super()._render_current()
            if float(self.world.current_time)!=frozen_time or not np.array_equal(
                    array(self.view.get_joint_positions()),frozen_q):
                raise RuntimeError('Physics advanced during pose-checked render')
            if self.pose_guard is None:return images
            errors=self._camera_pose_errors()
            self.last_pose_sync=errors
            if all(v<=1e-4 for v in errors.values()):
                self.max_pose_sync_error=max(self.max_pose_sync_error,max(errors.values(),default=0.))
                self.last_verified_camera_transforms={key:value.copy() for key,value in self.last_camera_transforms.items()}
                if rejected:
                    with (self.shared/'pose_sync_retries.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(physics_time=frozen_time,attempt=attempt+1,
                            errors=errors,recovered=True,discarded_captures=len(rejected)))+'\n')
                return images
            rejected.append(errors)
            self.pose_sync_retries+=1
            diagnostic=dict(physics_time=frozen_time,attempt=attempt+1,errors=errors,
                            recovered=False,rejected_frame_discarded=True,
                            joints=frozen_q.tolist(),camera_poses=getattr(self,'last_camera_pose_diagnostic',{}))
            with (self.shared/'pose_sync_retries.jsonl').open('a') as stream:
                stream.write(json.dumps(diagnostic)+'\n')
        raise RuntimeError(f'Rendered camera/body pose differs from current physics after 3 frozen captures: {rejected}')

    def _camera_pose_errors(self):
        if not self.pose_guard.is_physics_handle_valid():self.pose_guard.initialize()
        position,orientation=self.pose_guard.get_world_poses()
        position,orientation=array(position),array(orientation)
        transforms={};errors={}
        for path,p,q in zip(self.pose_guard.prim_paths,position,orientation):
            physical=np.eye(4);physical[:3,3]=p
            physical[:3,:3]=Rotation.from_quat(np.roll(q,-1)).as_matrix()
            transforms[path]=physical
        self.last_camera_transforms={}
        for key,(path,relative) in self.camera_mounts.items():
            self.last_camera_transforms[key]=transforms[path]@relative
        self.last_camera_pose_diagnostic={}
        for (_,_,_,entries),ann in zip(self.products,self.camera_param_annotators):
            # The evaluator uses individual products so each camera is checked.
            if len(entries)!=1:raise RuntimeError('Pose guard requires individual camera products')
            env,name=entries[0]
            if (env.index,name) not in self.camera_mounts:continue
            expected=self.last_camera_transforms[(env.index,name)]
            native_view=np.asarray(ann.get_data()['cameraViewTransform']).reshape(4,4).T
            errors['native_'+name]=float(np.max(np.abs(native_view@expected-np.eye(4))))
            previous=getattr(self,'last_verified_camera_transforms',{}).get((env.index,name))
            self.last_camera_pose_diagnostic[name]=dict(expected=expected.tolist(),
                rendered_view=native_view.tolist(),
                previous_verified_pose_error=float(np.max(np.abs(native_view@previous-np.eye(4))))
                    if previous is not None else None)
        return errors

    def prepare_wave(self,configs,outputs):
        super().prepare_wave(configs,outputs)
        import json
        from pressb.astribot_collection.contract import atomic_json
        for env in self.envs:
            path=env.output/'calibration.json'
            data=json.loads(path.read_text())
            for (index,name),matrix in self.last_camera_transforms.items():
                if index!=env.index:continue
                camera=data['cameras'][name]
                camera['usd_authored_rest_world_transform']=camera['initial_world_transform']
                camera['initial_world_transform']=matrix.tolist()
                camera['world_transform_source']='current PhysX rigid body and fixed camera mount; independently checked against rendered CameraParams'
            atomic_json(path,data)
