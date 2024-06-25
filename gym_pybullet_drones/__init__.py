from gym.envs.registration import register

register(
    id='ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:CtrlAviary',
)

register(
    id='dyn-aviary-v0',
    entry_point='gym_pybullet_drones.envs:DynAviary',
)

register(
    id='velocity-aviary-v0',
    entry_point='gym_pybullet_drones.envs:VelocityAviary',
)

register(
    id='vision-aviary-v0',
    entry_point='gym_pybullet_drones.envs:VisionAviary',
)



# New--------------------------------------------------


register(
    id='routing-aviary-v0',
    entry_point='gym_pybullet_drones.envs:RoutingAviary',
)

register(
    id='extendedsingleagent-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:ExtendedSingleAgentAviary',
)

register(
    id='ca_static-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:AutoroutingAviary',
)

# -----------------------------------------------------

register(
    id='takeoff-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:TakeoffAviary',
)

register(
    id='hover-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:HoverAviary',
)

register(
    id='flythrugate-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:FlyThruGateAviary',
)

register(
    id='tune-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:TuneAviary',
)




register(
    id='flock-aviary-v0',
    entry_point='gym_pybullet_drones.envs.multi_agent_rl:FlockAviary',
)

register(
    id='leaderfollower-aviary-v0',
    entry_point='gym_pybullet_drones.envs.multi_agent_rl:LeaderFollowerAviary',
)

register(
    id='meetup-aviary-v0',
    entry_point='gym_pybullet_drones.envs.multi_agent_rl:MeetupAviary',
)