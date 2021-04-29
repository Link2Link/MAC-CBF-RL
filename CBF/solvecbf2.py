import numpy as np
from process import *
from cvxpy import *

alpha_0 = 3.0
alpha_1 = 4.0


def solvecbf(speed_control, angle_control, cbf_obs:CBFObservation):

    x = cbf_obs.ego.p[0]
    y = cbf_obs.ego.p[1]

    x_cur = cbf_obs.cxe
    y_cur = cbf_obs.cye
    diff = 4

    r1 = np.abs(cbf_obs.r)
    r2 = np.abs(cbf_obs.r) + diff

    v = cbf_obs.ego.speed
    phi = cbf_obs.ego.angle

    cars_p = cbf_obs.obstocal.p
    num = len(cars_p)
    cars_v = cbf_obs.obstocal.v

    # define variable
    u_a = Variable()
    u_w = Variable()
    slack1 = Variable()
    slack2 = Variable()

    sin = np.sin
    cos = np.cos

    # calc h
    h_r = 0.5*(x - x_cur)**2 +0.5*(y-y_cur)**2
    dh_r = (x - x_cur)*v*cos(phi) + (y - y_cur)*v*sin(phi)
    ddh_r = v**2 + (x-x_cur)*(u_a * cos(phi)-v*sin(phi)*u_w) + (y-y_cur)*(u_a * sin(phi)+v*cos(phi)*u_w)



    constraints = [ddh_r + alpha_1*dh_r + alpha_0*(h_r - 0.5*(r1**2)) >=-slack1,
                   -ddh_r - alpha_1*dh_r - alpha_0*(h_r - 0.5*(r2**2)) >=-slack1,
                   slack1 >=0,
                   slack2 >=0,
                   ]
    cars_constraints = []
    hc = []
    dhc = []
    ddhc = []
    for i in range(num):
        hc.append(0.5*(x - cars_p[i, 0])**2 +0.5*(y - cars_p[i, 1])**2)
        dhc.append((x - cars_p[i, 0])*(v*cos(phi)-cars_v[i, 0]) + (y - cars_p[i, 1])*(v*sin(phi)-cars_v[i, 1]))
        ddhc.append(
            (v * cos(phi) - cars_v[i, 0]) ** 2 + (x - cars_p[i, 0]) * (u_a * cos(phi) - v * sin(phi) * u_w) +
            (v * sin(phi) - cars_v[i, 1]) ** 2 + (y - cars_p[i, 1]) * (u_a * sin(phi) + v * cos(phi) * u_w)
        )
    for i in range(num):
        constraints.append(ddhc[i] + alpha_1*dhc[i] + alpha_0*hc[i] >= -slack2)

    obj = Minimize(square(u_a - speed_control) + 10 * square(u_w - angle_control) + 1000*(slack1 + slack2))
    prob = Problem(obj, constraints)
    prob.solve(solver='CVXOPT')  # Returns the optimal value.
    status = prob.status

    # print('h', h_r - 0.5*(r1**2), 0.5*(r2**2)-h_r)
    # print('cbf1', ddh_r.value + alpha_1*dh_r + alpha_0*(h_r - 0.5*(r1**2)))
    # print('cbf2', -ddh_r.value - alpha_1*dh_r - alpha_0*(h_r - 0.5*(r2**2)))


    # print(angle_control, u_w.value)
    # print(prob.status, slack1.value, slack2.value)
    if prob.status != 'optimal':
        return None, None,
    return u_a.value, -u_w.value
