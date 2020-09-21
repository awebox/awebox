import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.mdl.aero.tether_dir.frames as frames

def test_aero_comp():

    biot_savart.test()

    frames.test_transforms()

    return None
