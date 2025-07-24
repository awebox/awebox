'''
Rocking arm functions
- author: Antonin Bavoil 2025
'''

import casadi as cas
import numpy as np

def get_q_arm_tip(arm_angle, arm_length):
    return arm_length * cas.vertcat(np.cos(arm_angle), np.sin(arm_angle), 0.0)

def get_dq_arm_tip(arm_angle, darm_angle, arm_length):
    return darm_angle * arm_length * cas.vertcat(-np.sin(arm_angle), np.cos(arm_angle), 0.0)

def get_arm_passive_and_active_torques(variables_si, parameters):
    passive_torque = -variables_si['theta']['torque_slope'] * variables_si['x']['darm_angle']
    active_torque = variables_si['x']['active_torque']
    return passive_torque, active_torque
