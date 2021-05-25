import gym
from pathlib import Path
import numpy as np
import torch
import time
import random

import CartPole
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from algos.algos.ppo import PPO
from algos.memory.memory import Memory
from algos.utils.logger import Logger, get_logger
from algos.utils.smarts_utils import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from tensorboardX import SummaryWriter

from CBF.solvecbf2 import solvecbf

import warnings

warnings.filterwarnings("ignore")


def state_wrapper(state):
    state = state['AGENT-008']
    cbf_obs = state['cbf_obs']
    a = []
    for k in state.keys():
        a.extend(list(state[k])) if k != 'cbf_obs' else True
    return np.array(a), cbf_obs


class PIDController:
    def __init__(self, P=1, I=0, D=0):
        self.P = P
        self.D = D
        self.I = I
        self.err_int = 0
        self.err_last = 0
        self.dt = 0.1

    def __call__(self, target, current):
        err = target - current
        output = self.P * err + self.D * (err - self.err_last) + self.I * self.err_int
        self.err_int += err * self.dt
        self.err_int = np.clip(self.err_int, -10, 10)
        self.err_last = err
        return output


def main():
    ############## Hyperparameters ##############
    render = False
    writer = SummaryWriter('logs/ppo4')  # 参数为指定存储路径
    env_name = 'smarts'
    scenario_path = '/home/llx/code/SMARTS/scenarios/myself'
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 1  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 500  # max timesteps in one episode

    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    alpha = 1
    random_seed = None
    #############################################
    # creating environment

    agent_specs = {"AGENT-008": agent_spec}
    scenario_path = Path(scenario_path).absolute()

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario_path],
        agent_specs={"AGENT-008": agent_spec},
        # set headless to false if u want to use envision
        headless=False,
        visdom=False,
        seed=42,
    )
    # env.close()
    state_dim = 79
    action_dim = 2

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(int(random.random() * 1000))

    memory = Memory()

    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    total_step = 0

    # logger = Logger(get_logger())
    # logger.setup_tb('./resource/' + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) + '/')

    # training loop
    for i_episode in range(1, max_episodes + 1):
        try:

            print('episode:', i_episode)
            stats = None
            state = env.reset()
            state, cbf_obs = state_wrapper(state)
            speed_control = PIDController(P=2, I=1, D=0)
            angle_control = PIDController(P=1, I=0.1, D=0.1)
            u_v = 0

            for t in range(max_timesteps):
                time_step += 1
                total_step += 1

                # Running policy_old:
                action = ppo.select_action(state, memory)

                # u_a, u_w = solvecbf(action[0], action[1], cbf_obs)
                # u_a = action[0] if u_a is None else u_a
                # u_w = action[1] if u_w is None else u_w
                #
                # u_v += u_a * 0.1
                # u_v = np.clip(u_v, 0, 20)
                # action[1] = -np.arctan2(u_w * 10 + angle_control(u_w, cbf_obs.ego.w),
                #                         cbf_obs.ego.speed) / np.pi * 2
                # action[0] = speed_control(u_v, cbf_obs.ego.speed)
                #
                # action[0] = speed_control(u_a, cbf_obs.ego.acc)
                # action[1] = angle_control(cbf_obs.ego.w, u_w)

                take_action = {"AGENT-008": action}
                state, reward, done, _ = env.step(take_action)
                state, cbf_obs = state_wrapper(state)
                reward = reward["AGENT-008"]
                done = done["__all__"]

                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if time_step % update_timestep == 0:
                    stats = ppo.update(memory)
                    memory.clear_memory()
                    time_step = 0
                running_reward += reward

                if render:
                    env.render()
                if done:
                    break

            avg_length += t

            # save every 50 episodes
            if i_episode % 50 == 0:
                torch.save(ppo.policy.state_dict(), './resource/{}.pth'.format(env_name))

            # logging
            if i_episode % log_interval == 0:
                avg_length = int(avg_length / log_interval)
                running_reward = (running_reward / log_interval)
                # if stats is not None:
                #     logger.log_stat("ppo/actor_loss", stats["actor_loss"], total_step)
                #     logger.log_stat("ppo/critic_loss", stats["critic_loss"], total_step)
                #     logger.log_stat("ppo/entropy", stats["entropy"], total_step)
                #     logger.log_stat("reward", running_reward, total_step)
                #     logger.log_stat("avg_length", avg_length, total_step)

                print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
                writer.add_scalar("reward", running_reward, i_episode)
                running_reward = 0
                avg_length = 0
        except:
            pass


if __name__ == '__main__':
    main()