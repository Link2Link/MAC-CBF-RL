import logging
import gym

import sys

import numpy as np

sys.path.append("/home/llx/code/SMARTS")

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from process import *
from CBF.solvecbf2 import *
from map.map import GridMap
from collections import deque
import matplotlib.pyplot as plt
from algos.utils import ssid

AGENT_ID = "Agent-007"
num_episodes = 10
linemap = GridMap.load(dir="map", name="mid")
sidemap1 = GridMap.load(dir="map", name="sidemap1")
sidemap2 = GridMap.load(dir="map", name="sidemap2")


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

L = 50
cbf_aout = deque(maxlen=L)
cbf_wout = deque(maxlen=L)
real_a = deque(maxlen=L)
real_w = deque(maxlen=L)
a_control = deque(maxlen=L)
w_control = deque(maxlen=L)

v_expected = deque(maxlen=L)
v_real = deque(maxlen=L)

expected_angle = deque(maxlen=L)
real_angle = deque(maxlen=L)

a_pred = deque(maxlen=L)

class DisSpace:
    def __init__(self, A, B, C, D, s0):
        self.s = s0
        self.A = A
        self.B = B
        self.C = C
        self.D = D
    def __call__(self, u):
        self.s = self.A * self.s + self.B * u
        return self.C*self.s + self.D

class CBFAgent(Agent):
    def __init__(self):
        self.speed_control_ref = PIDController(P=2, I=0.1, D=1)
        self.speed_control = PIDController(P=2, I=1, D=0)
        self.angle_control = PIDController(P=1, I=0, D=0)
        self.a_pred = DisSpace(0.67, -1.89, -4.07, 1.21, 0)
        self.init_flag = 1
        self.u_v = 0
        self.angle = 0



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
        linemap.show3circle(
            linemap.grid, sidemap1.grid, sidemap2.grid, [cxe, cye], re, cbf_obs
        )

        speed_control = 0.3
        angle_control = 0
        u_a, u_w = solvecbf(speed_control, angle_control, cbf_obs)
        u_a = speed_control if u_a is None else u_a
        u_w = angle_control if u_w is None else u_w
        self.u_v += u_a * 0.1
        self.u_v = np.clip(self.u_v, 0, 15)
        angle_control = np.arctan2(u_w * 10 + self.angle_control(u_w, cbf_obs.ego.w), cbf_obs.ego.speed)/np.pi*2
        speed_control = self.speed_control(self.u_v, cbf_obs.ego.speed)

        speed_control = np.clip(speed_control, -1, 1)


        cbf_aout.append(u_a)
        real_a.append(cbf_obs.ego.acc)
        cbf_wout.append(u_w)
        real_w.append(cbf_obs.ego.w)
        a_control.append(speed_control)
        w_control.append(angle_control)
        v_expected.append(self.u_v)
        v_real.append(cbf_obs.ego.speed)
        expected_angle.append(self.angle)
        real_angle.append(cbf_obs.ego.angle)


        if self.init_flag:
            self.a_pred.s = (cbf_obs.ego.acc - self.a_pred.D*speed_control)/self.a_pred.C
            self.init_flag = 0

        a_pred.append(self.a_pred(speed_control))

        plt.clf()
        # plt.subplot(121)
        plt.plot(cbf_wout, label=u"cbf_wout")
        plt.plot(real_w, label=u"real_w")
        plt.plot(w_control, label=u"w_control")
        plt.legend(bbox_to_anchor=(1, 1))
        # plt.ylim([-1.5, 1.5])

        # plt.subplot(122)
        # plt.plot(v_expected, label=u"v_expected")
        # plt.plot(v_real, label=u"v_real")
        # plt.plot(a_control, label=u"a_control")
        # plt.legend(bbox_to_anchor=(1, 1))


        plt.pause(0.01)
        plt.ioff()

        # print('after :', angle_control)
        return speed_control, -angle_control
        # return self.nonlinear_map(speed_control, angle_control, speed)

    def nonlinear_map(self, speed_control, angle_control, speed):
        angle_control = np.arctan2(3.5 * angle_control, speed)
        return (speed_control, angle_control)


import time


def action_adapter(action):
    if action[0] >= 0:
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

        global real_a
        global a_control

        real_a = deque(maxlen=L)
        a_control = deque(maxlen=L)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_action = agent.act(observations)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)
            u_in = np.array(a_control)[np.newaxis, :]
            a_out = np.array(real_a)[np.newaxis, :]

            if u_in.shape[1] > 10:
                A, B, C, D, Cov, Sigma = ssid.N4SID(u_in, a_out, 2, 2, 1)
                print('length {num}, A {A}, B {B}, C {C}, D {D}'.format(num=u_in.shape[1],A=A,B=B,C=C,D=D))


    env.close()


if __name__ == "__main__":
    scenarios = ["./../SMARTS/scenarios/myself"]

    main(
        scenarios=scenarios,
        sim_name=None,
        headless=False,
        num_episodes=10,
        seed=42,
    )
