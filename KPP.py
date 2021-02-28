"""
RRT_CONNECT_2D
@author: shujie guo and huiming zhou
"""

import os
import sys
import math
import copy
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

import env, plotting, utils


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
    def __eq__(self, other):

        if isinstance(other, self.__class__):
            return (abs(self.x - other.x) <0.000000001) and  (abs(self.x - other.x) <0.000000001)
        else:
            return False

class RrtConnect:
    def __init__(self, step_len, goal_sample_rate, iter_max):

        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max

        self.env = env.Env()
        self.plotting = plotting.Plotting( "RRT_CONNECT")
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.s_start = Node((0,0))
        self.s_goal = Node((0,0))
        self.path=[]
        self.key_points = []
    def ini_start_end(self,s_start, s_goal):

        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]
    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.s_goal, self.goal_sample_rate)#产生随机节点
            node_near = self.nearest_neighbor(self.V1, node_rand)#找到V1中距离随机节点最近的点
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.V1.append(node_new)
                node_near_prim = self.nearest_neighbor(self.V2, node_new)
                node_new_prim = self.new_state(node_near_prim, node_new)

                if node_new_prim and not self.utils.is_collision(node_new_prim, node_near_prim):
                    self.V2.append(node_new_prim)

                    while True:
                        node_new_prim2 = self.new_state(node_new_prim, node_new)
                        if node_new_prim2 and not self.utils.is_collision(node_new_prim2, node_new_prim):
                            self.V2.append(node_new_prim2)
                            node_new_prim = self.change_node(node_new_prim, node_new_prim2)
                        else:
                            break

                        if self.is_node_same(node_new_prim, node_new):
                            break

                if self.is_node_same(node_new_prim, node_new):
                    return self.extract_path(node_new, node_new_prim)

            if len(self.V2) < len(self.V1):
                list_mid = self.V2
                self.V2 = self.V1
                self.V1 = list_mid
        return None

    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        node_new = Node((node_new_prim2.x, node_new_prim2.y))
        node_new.parent = node_new_prim

        return node_new

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        if node_new_prim.x == node_new.x and \
                node_new_prim.y == node_new.y:
            return True

        return False

    def generate_random_node(self, sample_goal, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return sample_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    @staticmethod
    def extract_path(node_new, node_new_prim):
        path1 = [(node_new.x, node_new.y)]
        node_now = node_new

        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))

        path2 = [(node_new_prim.x, node_new_prim.y)]
        node_now = node_new_prim

        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))

        return list(list(reversed(path1)) + path2)

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    '''
    #完成两个节点间的路径查找
    '''
    def one_step(self,x_start,x_goal):
        self.ini_start_end(x_start, x_goal)
        self.plotting.ini(x_start, x_goal)
        self.plotting.draw_start_end()
        temp_path = self.planning()
        if temp_path == None:
            print("没找到路径")
            return
        if self.utils.get_dist(Node(temp_path[0]), Node(x_start)) \
                > self.utils.get_dist(Node(temp_path[len(temp_path) - 1]), Node(x_start)):
            self.path.extend(reversed(temp_path))###防止出现因将V1、V2互换导致的交叉连接
        else:
            self.path.extend(temp_path)
    '''
    #完成总路径查找
    '''
    def run(self):
        len_gp = len(self.key_points)
        for i in range(len_gp-1):
            self.one_step(self.key_points[i],self.key_points[i+1])

    def forward(self,step_len, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)
        dist = min(step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        return node_new

    def get_nearest_inter_point(self,point):
        for i in range(len(self.path)):
            if not self.utils.is_collision(Node(point),Node(self.path[i])):
                return i
        return -1
    def compress(self):
        temp_path = []
        curr_point = self.path[-1]
        temp_path.append(copy.deepcopy(curr_point))
        while not self.is_node_same(Node(curr_point),Node(self.path[0])):
            index = self.get_nearest_inter_point(curr_point)
            temp_path.append(copy.deepcopy(self.path[index]))
            curr_point = self.path[index]

        self.path.clear()
        self.path = temp_path

def main():

    rrt_conn = RrtConnect( 0.8, 0.05, 50000)
    x_start, x_goal = rrt_conn.plotting.set_start_end()  # 设置起始点

    start = time.process_time()
    rrt_conn.utils.get_keypoints(x_start, x_goal,0.3)
    rrt_conn.key_points = rrt_conn.utils.select_keypoints()
    rrt_conn.run()
    rrt_conn.compress()

    end = time.process_time()
    info = str(((end - start) * 1000))
    print("耗时：" + info)
    total_len = 0.0
    if len(rrt_conn.path) > 0:
        for k in range(len(rrt_conn.path) - 1):
            total_len += rrt_conn.utils.get_dist(Node((rrt_conn.path[k][0], rrt_conn.path[k][1])),
                                                 Node((rrt_conn.path[k + 1][0], rrt_conn.path[k + 1][1])))
        info = ("路径长度： %f\n" % total_len)
    else:
        info = (" 未找到路径\n")
    print(info)
    rrt_conn.plotting.animation_connect(rrt_conn.V1, rrt_conn.V2, rrt_conn.path)


if __name__ == '__main__':
    main()
