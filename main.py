import logging
import gym

import sys
sys.path.append('/home/llx/code/SMARTS')

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from process import *
from CBF.solvecbf2 import *
from map.map import GridMap
AGENT_ID = "Agent-007"
num_episodes = 10
linemap = GridMap.load(dir='map', name='mid')
sidemap1 = GridMap.load(dir='map', name='sidemap1')
sidemap2 = GridMap.load(dir='map', name='sidemap2')

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



class CBFAgent(Agent):
    def __init__(self):
        self.speed_control = PIDController(P=2, I=0.1, D=0)
        self.angle_control = PIDController(P=0.2, I=0, D=0)

    def act(self, observations: Observation):
        # print(observations.keys())
        obs = observations[AGENT_ID]
        cbf_obs = obs_Frenet(obs)

        position = cbf_obs.ego.p
        cxe, cye, re = linemap.curvature(position[0], position[1], size=[10, 5])
        cbf_obs.r = re
        cbf_obs.cxe = cxe
        cbf_obs.cye = cye



        # GridMap.show3(linemap.grid, sidemap1.grid, sidemap2.grid)
        linemap.show3circle(linemap.grid, sidemap1.grid, sidemap2.grid, [cxe, cye], re, cbf_obs)

        speed = cbf_obs.ego.speed
        target = 10
        speed_control = target - speed

        angle_control = 0
        waypoint = cbf_obs.road.waypoints[1, 5]
        position = cbf_obs.ego.p
        vec = waypoint - position
        vec_n = np.sqrt(np.sum(vec ** 2))
        vec = vec / vec_n
        direction = cbf_obs.ego.direction
        vec_o = np.zeros_like(vec)
        vec_o[0] = -vec[1]
        vec_o[1] = vec[0]
        err = np.sum(direction * vec_o)
        angle_control += err * 1

        u_a, u_w = solvecbf(speed_control, angle_control, cbf_obs)
        u_a = speed_control if u_a is None else u_a
        u_w = angle_control if u_w is None else u_w

        # print('u_cbf:', u_a, u_w)
        # print('now:', cbf_obs.ego.acc, cbf_obs.ego.w)

        speed_control = self.speed_control(u_a, cbf_obs.ego.acc)
        angle_control = self.angle_control(u_w, cbf_obs.ego.w)



        # print('after :', angle_control)
        return speed_control, angle_control
        # return self.nonlinear_map(speed_control, angle_control, speed)

    def nonlinear_map(self, speed_control, angle_control, speed):
        angle_control = np.arctan2(3.5 * angle_control, speed)
        return (speed_control, angle_control)

import time

def action_adapter(action):
    if action[0] >=0:
        return (action[0], 0, action[1])
    else:
        return (0, -action[0], action[1])


def main(scenarios, sim_name, headless, num_episodes, seed):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Full),
        agent_builder=CBFAgent,
        action_adapter=action_adapter,
    )
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
    )
    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)
        observations, rewards, dones, infos = env.step({AGENT_ID: (0, 0, 0)})

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_action = agent.act(observations)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    scenarios = ['./../SMARTS/scenarios/myself']

    main(
        scenarios=scenarios,
        sim_name=None,
        headless=False,
        num_episodes=10,
        seed=42,
    )