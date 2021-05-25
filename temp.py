import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # 导入模块
def filter(step, reward, p=0.99):
    filtered = [reward[0]]
    for r in reward:
        filtered.append(p*filtered[-1]+(1-p)*r)
    filtered = np.array(filtered[1:])
    return step, filtered
def smooth(step, reward, step_size=50):
    max_step = max(step)
    sl = [i for i in range(0, 1000, step_size)]
    smoothed_reward = []
    smmoothed_step = []
    for i in range(len(sl)-1):
        # print(sl[i], sl[i+1])
        idx = (sl[i] < step) & (step < sl[i+1])
        sum_reward = sum(reward[idx])
        num = sum(idx)
        smoothed_reward.append(sum_reward/num)
        smmoothed_step.append(sl[i+1])
    return np.array(smmoothed_step), np.array(smoothed_reward)

def concate_data(csv_list):
    rewards = []
    steps = []
    # csv_list = ["/home/llx/Downloads/run-ppo_cbf-tag-reward.csv", "/home/llx/Downloads/run-ppo_cbf2-tag-reward.csv"]
    for csv in csv_list:
        cbf_reward_pd = pd.read_csv(csv)
        step = cbf_reward_pd['Step'].to_numpy()
        reward = cbf_reward_pd['Value'].to_numpy()

        step, reward = smooth(step, reward, step_size=20)
        step, reward = filter(step, reward, p=0.9)

        rewards.append(reward)
        steps.append(step)
    reward = rewards = np.concatenate(rewards)
    episode = np.concatenate(steps)
    return episode, reward

sns.set() # 设置美化参数，一般默认就好


cbf1 = "/home/llx/Downloads/run-ppo_cbf-tag-reward.csv"
cbf2 = "/home/llx/Downloads/run-ppo_cbf2-tag-reward.csv"
ppo1 = "/home/llx/Downloads/run-ppo-tag-reward.csv"
ppo2 = "/home/llx/Downloads/run-ppo2-tag-reward.csv"
ppo3 = "/home/llx/Downloads/run-ppo3-tag-reward.csv"
ppo4 = "/home/llx/Downloads/run-ppo4-tag-reward.csv"

# cbf_reward_pd = pd.read_csv("/home/llx/Downloads/run-ppo_cbf-tag-reward.csv")
# cbf_reward_pd2 = pd.read_csv("/home/llx/Downloads/run-ppo_cbf2-tag-reward.csv")

# ppo_reward_pd = pd.read_csv("/home/llx/Downloads/run-ppo2-tag-reward.csv")

cbf_episode, cbf_reward = concate_data([cbf1, cbf2])
ppo_episode, ppo_reward = concate_data([ppo1, ppo2, ppo3, ppo4])

# cbf_step = cbf_reward_pd['Step'].to_numpy()
# cbf_reward = cbf_reward_pd['Value'].to_numpy()
# cbf_step2 = cbf_reward_pd2['Step'].to_numpy()
# cbf_reward2 = cbf_reward_pd2['Value'].to_numpy()
#
# ppo_step = ppo_reward_pd['Step'].to_numpy()
# ppo_reward = ppo_reward_pd['Value'].to_numpy()
#
#
# cbf_step, cbf_reward = smooth(cbf_step, cbf_reward, step_size=20)
# cbf_step, cbf_reward = filter(cbf_step, cbf_reward, p=0.9)
# cbf_step2, cbf_reward2 = smooth(cbf_step2, cbf_reward2, step_size=20)
# cbf_step2, cbf_reward2 = filter(cbf_step2, cbf_reward2, p=0.9)
# cbf_reward=rewards=np.concatenate((cbf_reward,cbf_reward2))
# episode=np.concatenate((cbf_step,cbf_step2))
#
#
#
# ppo_step, ppo_reward = smooth(ppo_step, ppo_reward, step_size=20)
# ppo_step, ppo_reward = filter(ppo_step, ppo_reward, p=0.9)

# print(cbf_reward.shape, ppo_reward.shape)
sns.lineplot(x=cbf_episode,y=cbf_reward)
sns.lineplot(x=ppo_episode,y=ppo_reward)
plt.legend(['PID-CBF-PPO', 'PPO'])
plt.xlabel("episode")
plt.ylabel("reward")
plt.show()