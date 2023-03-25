from gym.envs.registration import register

register(
    id='gym_examples/GridWorld-v1',
    entry_point='gym_examples.envs:GridWorldEnv',
    max_episode_steps=300,
)