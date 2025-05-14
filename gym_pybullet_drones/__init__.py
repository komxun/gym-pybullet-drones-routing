from gymnasium.envs.registration import register

register(
    id='ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:CtrlAviary',
)

register(
    id='velocity-aviary-v0',
    entry_point='gym_pybullet_drones.envs:VelocityAviary',
)

register(
    id='hover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:HoverAviary',
)

register(
    id='multihover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:MultiHoverAviary',
)

register(
    id='autorouting-aviary-v0',
    entry_point='gym_pybullet_drones.envs:AutoroutingRLAviary',
)

register(
    id='autorouting-sa-aviary-v0',
    entry_point='gym_pybullet_drones.envs:AutoroutingSARLAviary',
)

register(
    id='autorouting-marl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:AutoroutingMARLAviary',
)

register(
    id='routing-aviary-v0',
    entry_point='gym_pybullet_drones.envs:RoutingAviary',
)
