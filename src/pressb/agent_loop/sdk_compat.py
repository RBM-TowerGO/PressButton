"""Instance-local workaround for the S1B mixed-command remote-WBC routing bug.

The public mixed API does not expose remote_wbc_option. Its native Cartesian
sub-call defaults to remote WBC even with use_wbc=False. A four-joint torso
entry then leaks into the WBC Cartesian pose dictionary. Keep joint dispatch
unchanged, and explicitly select the existing SDK local Cartesian path.
"""
import inspect


def install_native_local_route(native):
    original = native.cartesian_pose_control
    if 'remote_wbc_option' not in inspect.signature(original).parameters:
        raise RuntimeError('unsupported_SDK_cartesian_contract: remote_wbc_option missing')

    def local_cartesian(name_list, command_dict, use_wbc=False, **kwargs):
        if use_wbc:
            raise ValueError('agent_requires_local_cartesian_control')
        # Mixed commands share a dictionary; only Cartesian entries belong here.
        poses = {name: command_dict[name] for name in name_list}
        if any(len(pose) != 7 for pose in poses.values()):
            raise ValueError('cartesian_pose_requires_7_values')
        kwargs['remote_wbc_option'] = False
        return original(name_list, poses, use_wbc=False, **kwargs)

    native.cartesian_pose_control = local_cartesian

    def restore():
        if native.cartesian_pose_control is local_cartesian:
            native.cartesian_pose_control = original
    return restore


def install_local_route(robot):
    interface = robot.astribot_interface
    native = getattr(interface, '_AstribotInterface__astribot_class', None)
    if native is None:
        raise RuntimeError('unsupported_SDK_interface: native mixed-command instance unavailable')
    return install_native_local_route(native)
