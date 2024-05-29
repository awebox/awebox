#!/usr/bin/python3
"""Test to check options functionality

@author: Jochem De Schutter,
"""

import awebox as awe
#
def test_seed_fill_in():
    """Test options seed routines
    """

    seed = {}
    seed['nlp.n_k'] = 5
    seed['user_options.system_model.architecture'] = {1:0, 2:1, 3:1, 4:1, 5:1}
    seed['model.aero.actuator.a_ref'] = 0.1
    options = awe.Options()
    options.fill_in_seed(seed)

    assert options['nlp']['n_k'] == 5
    assert options['user_options']['system_model']['architecture'] == {1:0, 2:1, 3:1, 4:1, 5:1}
    assert options['model']['aero']['actuator']['a_ref'] == 0.1

    return None

# test_seed_fill_in()