import gym
import CartPole
import numpy as np
env = gym.make('CartPole-C-v0')
state = env.reset()
env.render()

for i in range(500):
    angle = state[2]
    angle_d = state[3]
    control = 100 * angle + 10*angle_d

    state, reward, done, _ = env.step(control.astype(np.float32))
    # state, reward, done, _ = env.step(np.array([10]))

    print(i, state[1][0])
    env.render()
    if done:
        break