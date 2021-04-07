import gym
import numpy as np
import torch
import time

import CartPole
from algos.algos.ppo import PPO
from algos.memory.memory import Memory
from algos.utils.logger import Logger, get_logger
from CBF.solvecbf import solvecbf

def main():
    ############## Hyperparameters ##############
    env_name = "CartPole-C-v0"
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 2            # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    alpha = 1
    random_seed = None
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    total_step = 0
    
    logger = Logger(get_logger())
    logger.setup_tb('./resource/' + time.strftime("%a_%b_%d_%H%M%S_%Y", time.localtime()) + '/')

    # training loop
    for i_episode in range(1, max_episodes+1):
        stats = None
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            total_step += 1

            # Running policy_old:
            action_ref = ppo.select_action(state, memory)
            action, punish = solvecbf(state, action_ref, threshold=[-0.2617, 0.2617])
            state, reward, done, _ = env.step(action)
            reward += alpha * punish

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
                
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './resource/PPO_{}.pth'.format(env_name))
            
        # logging       
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            if stats is not None:
                logger.log_stat("ppo/actor_loss", stats["actor_loss"], total_step)
                logger.log_stat("ppo/critic_loss", stats["critic_loss"], total_step)
                logger.log_stat("ppo/entropy", stats["entropy"], total_step)
                logger.log_stat("reward", running_reward, total_step)
                logger.log_stat("avg_length", avg_length, total_step)

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()