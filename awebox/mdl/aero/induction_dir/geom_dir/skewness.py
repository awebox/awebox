



def extend_system_variables(model_options, system_lifted, system_states, architecture):

    system_lifted, system_states = extend_velocity_variables(model_options, system_lifted, system_states, architecture)

    return system_lifted, system_states
