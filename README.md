# MAC-CBF-RL

安装CartPole：在CartPole目录下运行pip install -e .

Deom
```
import gym
import CartPole
import numpy as np
env = gym.make('CartPole-C-v0')	# 新环境名字是CartPole-C-v0
state = env.reset()
env.render()

for i in range(500):
    angle = state[2]
    angle_d = state[3]
    control = 100 * angle + 10*angle_d

    state, reward, done, _ = env.step(control.astype(np.float32))		# state 是一个4x1 numpy [x, x_dot, angle, angle_dot]
																		# reward 是 0 or 1
    # state, reward, done, _ = env.step(np.array([10]))

    print(i, state[1][0])
    env.render()
    if done:
        break
```