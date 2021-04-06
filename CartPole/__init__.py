from gym.envs.registration import register

register(
    id='CartPole-C-v0',
    entry_point='CartPole.CartPoleEnvs:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)