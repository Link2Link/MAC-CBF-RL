import numpy as np

class model:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        # self.force_mag = 10.0
        self.force_max = 20.0
        self.tau = 0.02  # seconds between state updates

    def calc_acc(self, state, F):
        x, x_dot, theta, theta_dot = state
        force = F
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))

        return thetaacc

    def Lfh(self, state):
        return self.calc_acc(state, 0)

    def Lgh(self, state):
        return self.calc_acc(state, 1) - self.calc_acc(state, 0)

m = model()
pole = np.array([-1, -3])
k1 = pole[0] * pole[1]
k2 = -(pole[0] + pole[1])
a0 = k1
a1 = k2

def findrange(A, B):
    r = np.array([-np.inf, np.inf])
    if B > 0:
        r[0] = -A/B
    else:
        r[1] = -A/B
    return r

def applyrange(input, r):
    if input > r[1]:
        output = np.array([r[1]])
    elif input < r[0]:
        output = np.array([r[0]])
    else:
        output = input
    return output

def solvecbf(state, u_ref, threshold=[-0.2617, 0.2617]):
    x, x_dot, theta, theta_dot = state
    threshold = np.array(threshold, dtype=np.float32)
    A1 = m.Lfh(state) + a0*(theta - threshold[0]) + a1*theta_dot
    B1 = m.Lgh(state)
    r1 = findrange(A1, B1)

    A2 = -m.Lfh(state) + a0 * (threshold[1] - theta) - a1 * theta_dot
    B2 = -m.Lgh(state)
    r2 = findrange(A2, B2)

    if r1[1] > r2[0] and r1[0] < r2[1]:     # if feasible
        u = applyrange(u_ref, r1)
        u = applyrange(u, r2)
    else:
        u = applyrange(u_ref, r1)           # if not feaisble, use r1

    return u

