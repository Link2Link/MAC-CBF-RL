import numpy as np
from process import *
from cvxpy import *
import control



def calc_alpha(h, dh, ddh):
    y0 = h
    dy0 = dh
    a = np.max([-dy0/y0, 0]) + 1

    y1 = dh + a*y0
    dy1 = ddh + a*dy0
    b = np.max([-dy1/y1, 0]) + 3

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    p = -np.array([a, b])
    K = control.place(A, B, p)
    # print('a,b', a,b, 'K', K)
    return K[0,0], K[0,1]

def solvecbf(speed_control, angle_control, cbf_obs: CBFObservation):

    x = cbf_obs.ego.p[0]
    y = cbf_obs.ego.p[1]

    x_cur = cbf_obs.cxe
    y_cur = cbf_obs.cye
    diff = 3

    r1 = np.abs(cbf_obs.r) - diff
    r2 = np.abs(cbf_obs.r) + diff

    v = cbf_obs.ego.speed
    phi = cbf_obs.ego.angle

    acc = cbf_obs.ego.acc
    w = cbf_obs.ego.w

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
    h_r = 0.5 * (x - x_cur) ** 2 + 0.5 * (y - y_cur) ** 2
    dh_r = (x - x_cur) * v * cos(phi) + (y - y_cur) * v * sin(phi)
    ddh_r = (
        v ** 2
        + (x - x_cur) * (u_a * cos(phi) - v * sin(phi) * u_w)
        + (y - y_cur) * (u_a * sin(phi) + v * cos(phi) * u_w)
    )

    # calc alpha parameter
    h = h_r
    dh = dh_r
    ddh = (
        v ** 2
        + (x - x_cur) * (acc * cos(phi) - v * sin(phi) * w)
        + (y - y_cur) * (acc * sin(phi) + v * cos(phi) * w)
    )

    alpha1_1, alpha1_0 = calc_alpha(h[0] - 0.5 * (r1[0] ** 2), dh[0], ddh[0])
    alpha2_1, alpha2_0 = calc_alpha(h[0] - 0.5 * (r2[0] ** 2), dh[0], ddh[0])


    constraints = [
        ddh_r + alpha1_1 * dh_r + alpha1_0 * (h_r - 0.5 * (r1 ** 2)) >= -slack1,
        -ddh_r - alpha2_1 * dh_r - alpha2_0 * (h_r - 0.5 * (r2 ** 2)) >= -slack1,
        slack1 >= 0,
        slack2 >= 0,
    ]
    cars_constraints = []
    hc = []
    dhc = []
    ddhc = []
    ddhc_estimated = []
    for i in range(num):
        hc.append(0.5 * (x - cars_p[i, 0]) ** 2 + 0.5 * (y - cars_p[i, 1]) ** 2)
        dhc.append(
            (x - cars_p[i, 0]) * (v * cos(phi) - cars_v[i, 0])
            + (y - cars_p[i, 1]) * (v * sin(phi) - cars_v[i, 1])
        )
        ddhc.append(
            (v * cos(phi) - cars_v[i, 0]) ** 2
            + (x - cars_p[i, 0]) * (u_a * cos(phi) - v * sin(phi) * u_w)
            + (v * sin(phi) - cars_v[i, 1]) ** 2
            + (y - cars_p[i, 1]) * (u_a * sin(phi) + v * cos(phi) * u_w)
        )
        ddhc_estimated.append(
            (v * cos(phi) - cars_v[i, 0]) ** 2
            + (x - cars_p[i, 0]) * (acc * cos(phi) - v * sin(phi) * w)
            + (v * sin(phi) - cars_v[i, 1]) ** 2
            + (y - cars_p[i, 1]) * (acc * sin(phi) + v * cos(phi) * w)
        )
    for i in range(num):
        alpha_c1, alpha_c0 = calc_alpha(hc[i], dhc[i], ddhc_estimated[i])

        constraints.append(ddhc[i] + alpha_c1 * dhc[i] + alpha_c0 * hc[i] >= -slack2)

    obj = Minimize(
        square(u_a - speed_control)
        + 10 * square(u_w - angle_control)
        + 1000 * (slack1 + slack2)
    )
    prob = Problem(obj, constraints)
    prob.solve(solver="SCS")  # Returns the optimal value.
    status = prob.status

    # print('h', h_r - 0.5*(r1**2), 0.5*(r2**2)-h_r)
    # print('cbf1', ddh_r.value + alpha_1*dh_r + alpha_0*(h_r - 0.5*(r1**2)))
    # print('cbf2', -ddh_r.value - alpha_1*dh_r - alpha_0*(h_r - 0.5*(r2**2)))

    # print(angle_control, u_w.value)
    # print(prob.status, slack1.value, slack2.value)

    r = np.sqrt(2 * h_r)
    # print(r, r1, r2, (r-r1)/(r2-r1))
    if prob.status != "optimal":
        return (
            None,
            None,
        )
    return u_a.value, u_w.value


if __name__ == "__main__":
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    p = -np.array([1, 3])
    K = control.place(A, B, p)
    print(K)
    print(K[0, 0])
