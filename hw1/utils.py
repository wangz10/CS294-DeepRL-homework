def describe_env(env):
    print('Observations:', env.observation_space.dtype, env.observation_space.shape)
    print('Actions:', env.action_space.dtype, env.action_space.shape)
    print('Action space high:', env.action_space.high)
    print('Action space low:', env.action_space.low)
    print('Time step limit:', env.spec.timestep_limit)
    return
