# _*_ coding: utf-8 _*_
# @Time : 2021/4/26
# @Author : Chenfei Wang
# @File : map.py
# @desc :
# @note :

import numpy as np
from enum import Enum
from pathlib import Path
import cv2
from process import circle_fitting
from process import CBFObservation
class DataType(Enum):
    img = 0
    points = 1


class GridMap:
    def __init__(self, scale=10, x_range=[-20.0, 230.0], y_range=[-20.0, 120.0], name='mid'):
        size_x = int((x_range[1] - x_range[0]) * scale)
        size_y = int((y_range[1] - y_range[0]) * scale)
        self.grid = np.zeros([size_x, size_y], dtype=np.bool)
        self.scale = scale
        self.x_range = x_range
        self.y_range = y_range
        self.name = name

        self.grid_data = np.zeros([size_x, size_y, 2], dtype=np.float)

    def input_trans(self, x, y):
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        x_trans =(x - x_min) * self.scale
        y_trans =(y - y_min) * self.scale
        return np.floor(x_trans).astype(int), np.floor(y_trans).astype(int)

    def add_to_map(self, points):
        x = points[:, 0]
        y = points[:, 1]
        idx = self.input_trans(x, y)
        self.grid[idx] = True
        self.grid_data[idx] = points



    def __len__(self):
        return np.sum(self.grid == 1)

    def neighbor(self, x, y, size=[10, 10], show=False):
        x, y = self.input_trans(x, y)
        x_min = x - size[0] * self.scale
        x_max = x + size[0] * self.scale
        y_min = y - size[1] * self.scale
        y_max = y + size[1] * self.scale
        img = self.grid[x_min:x_max, y_min:y_max]
        points = self.grid_data[x_min:x_max, y_min:y_max]
        points = points[img==True]
        if show:
            show_img = img.astype(np.float)
            show_img = show_img.transpose()
            show_img = cv2.flip(show_img, 0)
            cv2.imshow(self.name, show_img)
            cv2.waitKey(1)
        return img, points


    def curvature(self, x, y, size=[15, 15]):

        img, points = self.neighbor(x, y, size, show=False)
        cxe, cye, re, error = circle_fitting(points[:, 0], points[:, 1])

        return cxe, cye, re


    def save(self, dir='.'):
        path = Path(dir)
        config_dict = {'scale':self.scale, 'x_range':self.x_range, 'y_range':self.y_range}
        np.save(path.joinpath(self.name + 'grid.npy'), self.grid, allow_pickle=True)
        np.save(path.joinpath(self.name + 'grid_data.npy'), self.grid_data, allow_pickle=True)
        np.save(path.joinpath(self.name + 'grid_config.npy', ), config_dict, allow_pickle=True)
        # print('save to ', path.joinpath(self.name))


    @staticmethod
    def load(dir='.', name='mid'):
        path = Path(dir)
        config_dict = np.load(path.joinpath(name + 'grid_config.npy'), allow_pickle=True)
        grid = np.load(path.joinpath(name + 'grid.npy'), allow_pickle=True)
        grid_data = np.load(path.joinpath(name + 'grid_data.npy'), allow_pickle=True)

        map = GridMap(scale=config_dict.item()['scale'], x_range=config_dict.item()['x_range'], y_range=config_dict.item()['y_range'])
        map.grid = grid
        map.grid_data = grid_data
        return map



    def show(self):
        img = self.grid.astype(float).transpose()
        img = cv2.flip(img, 0)
        cv2.imshow(self.name, img)
        cv2.waitKey(1)

    @staticmethod
    def show3(mid, side1, side2):
        [W, H] = mid.shape
        r = np.zeros([W,H], dtype=float)
        g = np.zeros([W,H], dtype=float)
        b = np.zeros([W,H], dtype=float)
        r[mid == True] = 1
        g[mid == True] = 1
        b[mid == True] = 1

        g[side1 == True] = 1
        r[side2 == True] = 1
        r = r.transpose()
        g = g.transpose()
        b = b.transpose()

        img = cv2.merge([b, g, r])
        img = cv2.flip(img, 0)
        cv2.imshow('3 lines', img)
        cv2.waitKey(1)


    def show3circle(self, mid, side1, side2, center, radius, cbf_obs:CBFObservation):


        [W, H] = mid.shape
        r = np.zeros([W, H], dtype=float)
        g = np.zeros([W, H], dtype=float)
        b = np.zeros([W, H], dtype=float)
        r[mid == True] = 1
        g[mid == True] = 1
        b[mid == True] = 1

        g[side1 == True] = 1
        r[side2 == True] = 1
        r = r.transpose()
        g = g.transpose()
        b = b.transpose()
        img = cv2.merge([b, g, r])
        img = cv2.flip(img, 0)

        size_y = int((self.y_range[1] - self.y_range[0]) * self.scale)
        center = self.input_trans(center[0], center[1])

        img = cv2.circle(img, (center[0], size_y - center[1]), radius=int(radius * self.scale), color=1)

        # img = cv2.circle(img, (center[0], size_y - center[1]), radius=int((radius-4) * self.scale), color=(0,255,0))
        # img = cv2.circle(img, (center[0], size_y - center[1]), radius=int((radius+4) * self.scale), color=(0,255,0))



        # drawcar

        position = cbf_obs.ego.p
        angle = cbf_obs.ego.angle
        L = 3.68/2
        Lcos = L*np.cos(angle)
        Lsin = L*np.sin(angle)
        W = 1.47/2
        Wcos = W*np.cos(angle)
        Wsin = W*np.sin(angle)



        box = np.array([[L, W], [-L, W], [-L, -W], [L, -W]])
        R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        box = box@R.transpose()
        box += position


        car_x, car_y = self.input_trans(box[:, 0], box[:, 1])
        car_y = size_y - car_y

        cv2.line(img, (car_x[0], car_y[0]), (car_x[1], car_y[1]), color=(255,255,255), thickness=None)
        cv2.line(img, (car_x[1], car_y[1]), (car_x[2], car_y[2]), color=(255, 255, 255), thickness=None)
        cv2.line(img, (car_x[2], car_y[2]), (car_x[3], car_y[3]), color=(255, 255, 255), thickness=None)
        cv2.line(img, (car_x[3], car_y[3]), (car_x[0], car_y[0]), color=(255, 255, 255), thickness=None)

        # print(car_x, car_y)
        #
        # img = cv2.boxPoints(img, )
        # # print(position)
        #
        cv2.imshow('3 lines', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    map = GridMap.load('.')
    map.show()
    cv2.waitKey(0)
    print(len(map))
    print(map.grid_data)