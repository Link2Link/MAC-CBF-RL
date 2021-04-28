import numpy as np
from smarts.core.sensors import Observation
import cv2
def norm(vec):
    value = np.sqrt(np.sum(np.square(vec), axis=-1, keepdims=True))
    return value

def circle_fitting(x, y):
    """
    Circle Fitting with least squared
        input: point x-y positions
        output  cxe x center position
                cye y center position
                re  radius of circle
                error: prediction error
    """

    sumx = np.sum(x)
    sumy = np.sum(y)
    sumx2 = np.sum(np.square(x))
    sumy2 = np.sum(np.square(y))
    sumxy = np.sum(x*y)

    F = np.array([[sumx2, sumxy, sumx],
                  [sumxy, sumy2, sumy],
                  [sumx, sumy, len(x)]])

    G = np.array([[-np.sum(x ** 3 + x * (y ** 2))],
                  [-np.sum((x ** 2)*y + y ** 3)],
                  [-np.sum(x ** 2 + y ** 2)]])

    D = np.linalg.det(F)
    if D < 0.01:
        X = np.array([x, y]).transpose()
        I = np.ones([len(x), 1])
        A = -np.linalg.inv(X.transpose()@X)@X.transpose()@I
        A_num = np.sqrt(np.sum(A**2))
        A /= A_num
        re = np.array([2000.0], dtype=np.float64)
        cxe = np.mean(x) + re*A[0]
        cye = np.mean(y) + re*A[1]
        error = 0

        return (cxe, cye, re, error)
    T = np.linalg.inv(F).dot(G)

    cxe = T[0] / -2
    cye = T[1] / -2

    re = np.sqrt(cxe**2 + cye**2 - T[2])
    error = sum([np.hypot(cxe - ix, cye - iy) - re for (ix, iy) in zip(x, y)])

    return (cxe, cye, re, error)

def circle_fitting2(x, y):
    sumx3 = np.sum(x**3)
    sumx1y2 = np.sum(x*(y**2))
    sumx2 = np.sum(x**2)
    r = -sumx3 - sumx1y2
    r = r / (2*sumx2)
    return r

class vehicle:
    def __init__(self):
        self.p = None
        self.v = None
        self.speed = 0
        self.speed_last = 0
        self.acc = 0
        self.angle = 0
        self.angle_last = 0
        self.w = 0
        self.p_last = None
        self.direction = 0
        self.dt = 0.1

    def __call__(self, obs:Observation):
        state = obs.ego_vehicle_state
        position = state.position[:2]

        self.p_last = self.p
        self.angle_last = self.angle

        self.p = position
        self.speed_last = self.speed
        self.speed = state.speed
        self.angle = state.heading + np.pi/2

        self.direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.v = self.speed*self.direction
        # self.a = state.linear_acceleration[:2]
        self.acc = (self.speed - self.speed_last)/self.dt

        if np.abs(self.angle - self.angle_last) > np.pi*1.9:
            if self.angle >= 0:
                w = self.angle - self.angle_last - 2*np.pi
            else:
                w = self.angle - self.angle_last + 2*np.pi
        else:
            w = self.angle - self.angle_last

        self.w = w / self.dt


class obstocal:
    def __init__(self):
        self.p = None
        self.v = None
        self.angle = None

    def __call__(self, obs:Observation):
        neighbor = obs.neighborhood_vehicle_states
        num = len(neighbor)
        pos = np.array([neighbor[i].position[:2] for i in range(num)])
        heading = np.array([neighbor[i].heading + np.pi/2 for i in range(num)])
        speed = np.array([neighbor[i].speed for i in range(num)])
        v = np.zeros_like(pos)
        v[:, 0] = speed * np.cos(heading)
        v[:, 1] = speed * np.sin(heading)

        self.p = pos
        self.angle = heading
        self.v = v

class road:
    def __init__(self):
        self.pos = None
        self.heading = None
        self.curvature = None
        self.direction = None
        self.ds = None
        self.dpos = None

    def __call__(self, obs:Observation):
        paths = obs.waypoint_paths
        pos = np.array([[paths[i][j].pos for j in range(33)] for i in range(3)])
        # pos2 = np.array([[paths[i][j].pos for j in range(31)] for i in range(3)])
        self.waypoints = pos
        self.pos = pos[:, :30, :]
        self.dpos = pos[:, 1:31, :] - self.pos
        self.ds = np.sqrt(np.sum(np.square(self.dpos), axis=2))
        self.heading = np.array([[paths[i][j].heading + np.pi/2 for j in range(30)] for i in range(3)])
        self.direction = np.zeros_like(self.dpos)
        self.direction[:, :, 0] = np.cos(self.heading)
        self.direction[:, :, 1] = np.sin(self.heading)
        # self.curvature = self.calc_curvature(0, 15)


    def calc_curvature(self, num, length):
        ds = self.ds[:, num:num+length]
        vec = self.dpos[:, num:num+length]


        s = np.sum(ds, axis=1)
        vec1 = vec[:, 0, :] /norm(vec[:, 0, :])
        vec2 = vec[:, -1, :] /norm(vec[:, -1, :])

        vec1_ = np.zeros_like(vec2)
        vec1_[:, 0] = vec1[:, 1]
        vec1_[:, 1] = -vec1[:, 0]


        # theta1 = np.arctan2(vec1[:, 1], vec1[:, 0])
        inner = np.sum(vec1 * vec2, axis=-1) - 1e-7
        inner_ = np.sum(vec1_ * vec2, axis=-1)
        theta = np.arccos(inner)
        theta = theta * np.sign(inner_)

        curvature = theta / s
        return curvature

class CBFObservation:
    def __init__(self):
        self.ego = vehicle()
        self.road = road()
        self.obstocal = obstocal()

    def __call__(self, obs:Observation):
        self.obs = obs
        self.road(obs)
        self.ego(obs)
        self.obstocal(obs)

        # self.set_Frenet()
        # self.transfer2Frenet()

        # self.draw()
    def transfer2Frenet(self):
        cos = np.cos
        sin = np.sin
        pi = np.pi

        origin = self.Frenet_origin
        angle = self.Frenet_heading - pi/2
        R = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

        #  transfer the road information

        waypoints = self.road.waypoints
        pos = waypoints - origin[np.newaxis, np.newaxis, :]
        pos = pos @ R

        self.road.pos = pos[:, :30, :]
        self.road.dpos = pos[:, 1:31, :] - pos[:, :30, :]
        self.road.ds = np.sqrt(np.sum(np.square(self.road.dpos), axis=2))
        self.road.heading = self.road.heading - angle
        self.road.direction = np.zeros_like(self.road.dpos)
        self.road.direction[:, :, 0] = cos(self.road.heading)
        self.road.direction[:, :, 1] = sin(self.road.heading)
        self.road.curvature = self.road.calc_curvature(0, 15)


        # transfer the ego information
        self.ego.p = self.ego.p - origin
        self.ego.p_last = self.ego.p_last - origin if self.ego.p_last is not None else None

        v = self.ego.v[np.newaxis, :] @ R
        self.ego.v = np.squeeze(v)
        self.ego.angle = self.ego.angle - angle
        self.ego.direction = np.array([cos(self.ego.angle), sin(self.ego.angle)])

        self.r = -circle_fitting2(self.road.pos[1, :5, 0], self.road.pos[1, :5, 1])
        self.r_diff = 5



        # transfer obstacle
        self.obstocal.p = self.obstocal.p - origin
        self.obstocal.angle = self.obstocal.angle - angle
        self.obstocal.v = self.obstocal.v @ R

        # scale = 5
        # coordinate = pos
        # img = np.zeros([125 * scale, 250 * scale])
        # for i in range(3):
        #     for j in range(30):
        #         cv2.circle(img, (self.toint(coordinate[i, j, 0] + 125), self.toint(100 - coordinate[i, j, 1])), radius=1,
        #                    color=(1, 1, 1))
        # cv2.circle(img, (self.toint(self.ego.p[0] + 125), self.toint(100 - self.ego.p[1])), radius=10, color=(1, 1, 1))
        # arrow1 = self.ego.p
        # arrow2 = self.ego.p + self.ego.v * self.ego.speed/4
        # # arrow3 = self.ego.p + self.ego.direction * 5
        # cv2.arrowedLine(img, (self.toint(arrow1[0] + 125), self.toint(100 - arrow1[1])),
        #                 (self.toint(arrow2[0] + 125), self.toint(100 - arrow2[1])), color=1, thickness=2)
        # # cv2.arrowedLine(img, (self.toint(arrow1[0] + 125), self.toint(100 - arrow1[1])),
        # #                 (self.toint(arrow3[0] + 125), self.toint(100 - arrow3[1])), color=(255,255,255), thickness=2)
        #
        # if np.abs(self.r) < 125:
        #     cv2.circle(img, (self.toint(self.r + 125), self.toint(np.array(100))), radius=self.toint(np.abs(self.r)),
        #                color=(1, 1, 1))
        #     cv2.circle(img, (self.toint(self.r + 125), self.toint(np.array(100))), radius=self.toint(np.abs(self.r)-5),
        #                color=(1, 1, 1))
        #     cv2.circle(img, (self.toint(self.r + 125), self.toint(np.array(100))), radius=self.toint(np.abs(self.r)+5),
        #                color=(1, 1, 1))
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)



    def set_Frenet(self):
        mid_line = self.obs.waypoint_paths[1]
        Frenet_origin = mid_line[0].pos
        Frenet_heading = mid_line[0].heading + np.pi/2
        self.Frenet_origin = Frenet_origin
        self.Frenet_heading = Frenet_heading
        # print(self.Frenet_origin, self.Frenet_heading)


    def toint(self, input, scale=5):
        temp = input*scale
        temp = temp.astype(int)
        return temp

    def draw(self):
        scale = 5
        coordinate = self.road.pos
        coordinate = coordinate.astype(int)
        img = np.zeros([250* scale,250* scale])
        for i in range(3):
            for j in range(30):
                cv2.circle(img, (self.toint(coordinate[i, j, 0]), self.toint(250-coordinate[i, j, 1])), radius=1, color=(1, 1, 1))
        cv2.circle(img, (self.toint(self.ego.p[0]), self.toint(250-self.ego.p[1])), radius=4, color=(1, 1, 1))

        # num = 15
        # curventure = self.road.curvature
        # r = 1/curventure
        # r = r[:, np.newaxis]
        # direction = self.road.direction[:, 0, :]
        # L = np.zeros_like(direction)
        # L[:, 0] = direction[:, 1]
        # L[:, 1] = -direction[:, 1]
        # base = coordinate[:, 0, :]
        # center = base + L * r


        # r = np.abs(r)
        # (cxe0, cye0, re0, error) = circle_fitting(coordinate[0, :num, 0], coordinate[0, :num, 1])
        # (cxe1, cye1, re1, error) = circle_fitting(coordinate[1, :num, 0], coordinate[1, :num, 1])

        # (cxe2, cye2, re2, error) = circle_fitting(coordinate[2, :num, 0], coordinate[2, :num, 1])

        # if re0 > 0:
        #     cv2.circle(img, (self.toint(cxe0[0]), self.toint(250-cye0[0])), radius=self.toint(re0[0]), color=(1, 0, 0))
        # if re1 > 0:
        #     cv2.circle(img, (self.toint(cxe1[0]), self.toint(250 - cye1[0])), radius=self.toint(re1[0]),
        #                color=(1, 1, 1))
        # if np.abs(r[1, 0]) < 100:
        #     cv2.circle(img, (self.toint(center[1,0]), self.toint(250 - center[1,1])), radius=self.toint(np.abs(r[1, 0])), color=(1, 1, 1))

        # if re2 > 0:
        #     cv2.circle(img, (self.toint(cxe1[0]), self.toint(250-cye2[0])), radius=self.toint(re2[0]), color=(0, 1, 0))


        cv2.imshow("Image", img)
        cv2.waitKey(1)



def Change2Frenet(x, origin, heading):
    print(x.shape)

cbf_obs = CBFObservation()

def obs_Frenet(obs:Observation):
    cbf_obs(obs)
    return cbf_obs