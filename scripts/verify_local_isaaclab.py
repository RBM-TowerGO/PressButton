import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
from isaaclab.app import AppLauncher
class LocalAppLauncher(AppLauncher):
    _SIM_APP_CFG_TYPES = {**AppLauncher._SIM_APP_CFG_TYPES, "create_new_stage": [bool]}

launcher = LocalAppLauncher(headless=True, enable_cameras=True, device='cuda:0', kit_args='--/validate/p2p/enabled=false --/validate/iommu/enabled=false')
app = launcher.app
try:
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaacsim.core.utils.stage import create_new_stage
    create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=1/120, device='cuda:0'))
    sim.reset()
    for _ in range(10):
        sim.step(render=True)
    print('PASS: Isaac Lab reset and 10 physics steps on GPU 0', flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    app.app.post_quit(1)
    raise
finally:
    SimulationContext.clear_instance()
    app.close(wait_for_replicator=False)
