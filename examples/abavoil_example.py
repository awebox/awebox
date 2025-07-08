"""
TODO:
 - [x] add 'system_type" = 'rocking_mode'
 - [x] add arm_length, amr_inertia parameters  `opts/defaults.py`
 - what is sweep_type ???

 1. Torque proportional to the angular velocity of the arm
  - [x] add torque_slope
 2. Torque Control : choose how much the generator should resist the rotation
  - [x] add control `torque`                                https://vscode.dev/github/abavoil/awebox/blob/abavoil/awebox/mdl/system.py#L173
  - [x] don't add control bounds : let the tension on the lines handle that
  - [x] add bounds [0, 0] if disabled & normalize           https://vscode.dev/github/abavoil/awebox/blob/abavoil/awebox/opts/model_funcs.py#L906

 - [x] add state α and dα                                   https://vscode.dev/github/abavoil/awebox/blob/abavoil/awebox/mdl/system.py#L173
 - [ ] attach tether to arm                                 https://vscode.dev/github/abavoil/awebox/blob/abavoil/awebox/mdl/dynamics.py#L962
  - algebraic equation : x0 = l_a * [cos, sin, 0] = 0 et dx0 = ... ou x0 = dx0 = 0
  - si premier noeud, q1 = x0 et dq1 = dx0
 - dynamics of the arm `dynamics.py`
 - add kinetic energy of the arm `dynamics.py`              https://vscode.dev/github/abavoil/awebox/blob/abavoil/awebox/mdl/dynamics.py#L416
 - add power generation                                     https://vscode.dev/github/abavoil/awebox/blob/abavoil/awebox/mdl/dynamics.py#L327

 - Initialize path:
  - kite flies at v = 1/2 * L/D * v_wind * 2Δphi, over a distance d = 2 * (l_t + l_a) * Δφ -> Period T = v / d
  - kite(t) = (l_t + l_a) * [cos(φ) cos(θ), sin(φ) cos(θ), sin(θ)]], where θ = Θ0 + Δθ sin(2τ), φ = Δφ sin(τ), and τ = 2π t / T
  - arm(t) = l_a * [cos(α), sin(α), 0], where α = 2π t / T - π/2

Investigate:
 - sweep type
 - model.tether.attachment
 - how is the winch modeled ?
 - consider aerodynamic drag of the arm ? -> C_D_arm, α, dα, arm_diameter -> torque

Questions :
 - Add a tether ground attachment either at [0, 0, 0] or at the tip of the arm
"""