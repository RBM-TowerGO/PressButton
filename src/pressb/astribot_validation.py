"""Initial geometric checks, distinct from dynamic contact-force monitoring."""
def initial_overlap_check(stage, meta):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysicsSchemaTools
    import omni.physx
    robot = meta['robot_path']
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
                 and p.IsA(UsdGeom.Gprim) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    def body(path):
        p = stage.GetPrimAtPath(path)
        while p and not p.IsPseudoRoot():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(p.GetPath())
            p = p.GetParent()
        return path
    filtered = set()
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.FilteredPairsAPI):
            for other in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets():
                filtered.add(tuple(sorted((str(p.GetPath()), str(other)))))
        if p.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(p)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if a and b and not joint.GetCollisionEnabledAttr().Get():
                filtered.add(tuple(sorted((str(a[0]), str(b[0])))))
    tips = {robot + '/astribot_gripper_right_Link_' + s for s in ('L11','R11')}
    result = dict(status='unknown', method='PhysX overlap_shape for every enabled robot Gprim before motion; meshes use convex approximation',
                  robot_colliders=[str(p.GetPath()) for p in colliders if str(p.GetPath()).startswith(robot+'/')],
                  scene_colliders=[str(p.GetPath()) for p in colliders], filtered_body_pairs=[list(p) for p in sorted(filtered)],
                  support_allowance='base link /World/Environment/Floor only', queried=0, events=[], ignored_hits=[])
    pairs = set()
    try:
        interface = omni.physx.get_physx_scene_query_interface()
        for source in result['robot_colliders']:
            a = body(source)
            def hit(h):
                destination = str(h.collision)
                b = body(destination)
                pair = tuple(sorted((a,b)))
                allowed = a == b or pair in filtered or {a,b} == tips or (a == meta['base_link'] and destination == '/World/Environment/Floor')
                row = dict(collider_pair=[source,destination], body_pair=[a,b])
                if allowed:
                    result['ignored_hits'].append(row)
                elif pair not in pairs:
                    pairs.add(pair)
                    result['events'].append(row)
                return True
            encoded = PhysicsSchemaTools.encodeSdfPath(source)
            interface.overlap_shape(encoded[0], encoded[1], hit, False)
            result['queried'] += 1
        result['status'] = 'collision' if result['events'] else 'clear'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if result['events']:
            result['status'] = 'collision'
    return result
