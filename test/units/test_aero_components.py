import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex
import awebox.mdl.aero.geometry_dir.geometry as geometry

def test_vortex_components():
    vortex.test()
    return None

def test_geometry_components():
    geometry.test()
    return None


if __name__ == "__main__":
    test_vortex_components()
    test_geometry_components()