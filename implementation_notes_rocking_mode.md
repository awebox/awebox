# Defining the rocking mode
## Parameters
  - `arm_length`
  - `arm_inertia`
  - `torque_slope` > 0: passive torque, `torque(darm_angle) = torque_slope * darm_angle`
  - `enable_arm_control`: active torque

## Variables:
  - x:
    - `arm_angle` in `[-π, π]`  # There shouldn't be any singularity since going around is 1. forbidden and 2. non-optimal
    - `darm_angle` in `[-4π, 4π]`  # Doing more than 2 full turns in 1 second seems unreasonable
    - `arm_control_torque` in `[-inf, inf]` (or `[0, 0]` if `enable_arm_control` is `False`), can be refined using maximum acceleration or maximum tether tension 
  - u:
    - `darm_control_torque` in `[-inf, inf]` (or `[0, 0]` if `enable_arm_control` is `False`)

## Constraints:
  - `q_0 = [cos(arm_angle), sin(arm_angle)]`

  - z:
    - A(arm_angle) in `[[-inf]*3, [inf]*3]`
    - dA(arm_angle, darm_angle) in `[[-inf]*3, [inf]*3]`

## Optimizable parameters
  - `torque_slope`
  - `arm_inertia`

## Energies
  - Kinetic: `0.5 * arm_inertia * darm_angle^2`
  - revise tether velocity at the ground station ? p.69

## Power output
  - `P_rocking = (torque_slope * darm_angle + arm_control_torque) * arm_angle` (positive is power is being extracted from the device)

## Dynamics:
  - (`d(arm_angle) = darm_angle`)
  - `arm_inertia * d(darm_angle) = (A - O) * base_tether_tension - control_torque - torque_slope * darm_angle (- arm_aero_torque)`
  - (`d(arm_control_torque) = darm_control_torque`)
  - besoin de quelque chose comme eq:3.5 ?

## Phase invariance
  - `psy_rocking(x_rocking(0)) = darm_angle(0) == 0`, also imposes oscillation of the arm instead of full rotations ?

## Initialization
### Fly in eights
 parameters `π^0 = (dq0, θ0, Δθ, Δφ, τ0, Δα)`, `dq0` speed of kite along the trajectory, `θ0 = 30°` elevation angle of the center of the eight, `Δθ = 10°` elevation amplitude of the eight, `Δφ = 40°` azimuth amplitude of the eight, `τ0 = π/4` initial phase of the eight (when arm angle `α` is maximized, value through my julia code), `Δα = 90°` amplitude of the arm angle `α`
  - arm(t) = l_a * [cos(α), sin(α), 0], where α = Δα * cos(τ - τ0) and τ = 2π t / T + τ0
  - kite flies at dq0 = 1/2 * L/D * v_wind, over a distance d = 2 * (l_t + l_a) * Δφ (very bad approximation) -> Period T = v / d
  - kite(t) = (l_t + l_a) * [cos(φ) cos(θ), sin(φ) cos(θ), sin(θ)], where θ = Θ0 + Δθ sin(2τ), φ = Δφ sin(τ)

### Fly in ellipses
 parameters `π^0 = (dq0, θ0, Δθ, Δφ, τ0, Δα)`, same except for circle, no idea for `τ0` and `Δα`, also, `α` will be way further to a sinusoid because of the symmetry breaking
  - same arm
  - d = 2π * (l_t + l_a) * sin(Δθ)
  - kite(t) = (l_t + l_a) * [cos(τ), sin(τ), 0], where θ = Θ0 + Δθ sin(τ), φ = Δφ cos(τ)
  

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


Investigate:
 - sweep type
 - model.tether.attachment
 - how is the winch modeled ?
 - consider aerodynamic drag of the arm ? -> C_D_arm, α, dα, arm_diameter -> torque

Questions :
 - Add a tether ground attachment either at [0, 0, 0] or at the tip of the arm
 - p.78: "by translating a priori knowledge on the optimal value of T into variable bounds" -> things like estimating the tension, kite speed etc.?
 - Homotopy: is it automatic or do I have to implement anything myself ?
 - Possible to simulate for half of the eight to use and impose symmetry ?