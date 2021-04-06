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

def solvecbf(state, u_ref, threshold= -0.2617):
    x, x_dot, theta, theta_dot = state
    A = m.Lfh(state) + a0*(theta - threshold) + a1*theta_dot
    B = m.Lgh(state)
    u_threshold = -A/B
    u = u_ref if u_ref <= u_threshold else u_threshold
    return u

