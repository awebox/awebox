import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import awebox.mdl.aero.induction_dir.vortex_dir.flow as vortex_flow
import awebox.mdl.aero.tether_dir.frames as frames

def test_aero_comp():

    biot_savart.test()
    test_list = vortex_filament_list.test()
    vortex_flow.test(test_list)

    frames.test_transforms()

    return None
