"""
@author: shujie guo and huiming zhou
"""

import math
import numpy as np
import os
import sys
import copy

import env
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")
class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Utils:
    def __init__(self):
        self.env = env.Env()

        self.delta = 0.00#######要改的，原来是0.5 太大了
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_vertex=[]
        self.get_obs_vertex()
        self.step = 1
        self.keypoints=[]
    def update_obs(self, obs_cir, obs_bound, obs_rec):
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec

    #得到obs_rectangle的四个顶点坐标
    def get_obs_vertex(self):
        delta = self.delta
        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [ox - delta, oy - delta,ox + w + delta, oy + h + delta]
            self.obs_vertex.append(vertex_list)


    #两条线段是否相交
    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False

    def is_intersect_circle(self, o, d, a, r):
        d2 = np.dot(d, d)
        delta = self.delta

        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r + delta:
                return True

        return False

    def cross(self,p1, p2, p3):  # 叉积判定
        x1 = p2[0] - p1[0]
        y1 = p2[1] - p1[1]
        x2 = p3[0] - p1[0]
        y2 = p3[1] - p1[1]
        return x1 * y2 - x2 * y1

    def segment(self,p1, p2, p3, p4):  # 判断两线段是否相交
        # 矩形判定，以l1、l2为对角线的矩形必相交，否则两线段不相交
        if (max(p1[0], p2[0]) >= min(p3[0], p4[0])  # 矩形1最右端大于矩形2最左端
                and max(p3[0], p4[0]) >= min(p1[0], p2[0])  # 矩形2最右端大于矩形1最左端
                and max(p1[1], p2[1]) >= min(p3[1], p4[1])  # 矩形1最高端大于矩形2最低端
                and max(p3[1], p4[1]) >= min(p1[1], p2[1])):  # 矩形2最高端大于矩形1最低端
            if self.cross(p1, p2, p3) * self.cross(p1, p2, p4) <= 0 and self.cross(p3, p4, p1) * self.cross(p3, p4, p2) <= 0:
                D = True
            else:
                D = False
        else:
            D = False
        return D

    def is_collision(self,start, end):
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True
        for (v0, v1, v2, v3) in self.obs_vertex:
            # step 2 check if diagonal cross the segment
            p1 = [v0, v1]
            p2 = [v2, v3]
            p3 = [v2, v1]
            p4 = [v0, v3]
            if self.segment([start.x,start.y], [end.x,end.y], p1, p2) or self.segment([start.x,start.y], [end.x,end.y], p3, p4):
                return True
        return False

    '''
    输入：线段的起止点start、end    
    输出：障碍物序号列表
    '''
    def get_obses_between_SE(self,start,end):
        indexes = []
        for i in range(len(self.obs_vertex)):
            p1 = [self.obs_vertex[i][0],self.obs_vertex[i][1]]
            p2 = [self.obs_vertex[i][2], self.obs_vertex[i][3]]
            p3 = [self.obs_vertex[i][2], self.obs_vertex[i][1]]
            p4 = [self.obs_vertex[i][0], self.obs_vertex[i][3]]
            if self.segment([start[0], start[1]], [end[0], end[1]], p1, p2) or self.segment([start[0], start[1]], [end[0], end[1]], p3, p4):
                indexes.append(i)

        return indexes
    '''
    输入：点    
    输出：点所在的障碍物的下标
    '''
    def get_obsindex(self,point):
        for i in range(len(self.obs_vertex)):
            if point[0] > self.obs_vertex[i][0] - self.delta and point[0] < self.obs_vertex[i][2] + self.delta \
                    and point[1] > self.obs_vertex[i][1] - self.delta and point[1] < self.obs_vertex[i][3] + self.delta:
                return i

    def get_distance_and_angle(self,start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return math.hypot(dx, dy), math.atan2(dy, dx)
    def get_dist_point(self,start, end):
        return math.hypot(end[0] - start[0], end[1] - start[1])
    '''
    沿线段方向向前一步
    '''
    def forward(self,step, start, end):
        dist, theta = self.get_distance_and_angle(start, end)
        dist = min(step, dist)
        point = ((start[0] + dist * math.cos(theta),
                         start[1] + dist * math.sin(theta)))
        return point

    '''
    防止相近的keypoint被重复加入
    '''
    def append_keypoints(self,point):
        flag = True
        for p in self.keypoints:
            if self.get_dist_point(p,point) < self.step:
                flag = False
                break
        if flag:
            self.keypoints.append(point)

    def get_noncollision(self,point,kps):
        num = 0
        for i in range(1, len(kps) - 1):
            if not self.is_collision(Node(point), Node(kps[i])):
                num += 1
        return num
    '''
    获取与起止点通视的点的数量
    '''
    def get_nonecollision_num(self):
        start = self.keypoints[0]
        end = self.keypoints[-1]
        num_start = self.get_noncollision(start,self.keypoints)
        num_end =  self.get_noncollision(end,self.keypoints)
        return num_start, num_end


    def find_kp(self, point, kps, order):

        distance = 1000
        temp_point = None
        point_to_del=[]

        if order:
            for kp in kps:
                if not self.is_collision(Node(point), Node(kp)):
                    point_to_del.append(kp)
                    curr_dis = self.get_dist_point(point, kp)
                    if  curr_dis < distance:
                        temp_point = kp
                        distance = curr_dis

        else:
            for kp in kps:
                if not self.is_collision(Node(point), Node(kp)):
                    point_to_del.append(kp)
                    curr_dis = self.get_dist_point(point, kp)
                    if curr_dis  <  distance:
                        temp_point = kp
                        distance = curr_dis
        return temp_point, point_to_del # temp_point== None表示没找到

    def get_intervisibility_point(self,point,kps):#order：顺序还是逆序

        indexs =[]
        for i in range(1,len(kps)-1):
            if  ((point[0]!=kps[i][0]) or (point[1]!=kps[i][1])) and not self.is_collision(Node(point), Node(kps[i])):#找到所有通视点
                indexs.append(i)
        return indexs
    '''
    对keypoints进行删减
    '''
    def select_keypoints(self):
        selected_kps = []
        temp_kps = copy.deepcopy(self.keypoints)
        find_all = False
        current_point = self.keypoints[0]# 把start选入
        selected_kps.append(current_point)
        temp_kps.remove(current_point)
        current_point,del_points = self.find_kp(current_point,temp_kps,True)
        while current_point != None:
            selected_kps.append(copy.deepcopy(current_point))
            temp_kps.remove(current_point)
            # for p in del_points:
            #     temp_kps.remove(p)
            if not self.is_collision(Node(current_point), Node(self.keypoints[-1])):  # 连接上了
                find_all = True
                selected_kps.append(self.keypoints[-1])
                break
            current_point ,del_points= self.find_kp(current_point, temp_kps,True)
        if not find_all:
            temp_rev_kps = []
            current_point = self.keypoints[-1]  # 把end选入
            temp_rev_kps.append(current_point)
            temp_kps.remove(current_point)
            current_point, del_points = self.find_kp(current_point, temp_kps, False)
            while current_point != None:
                temp_rev_kps.append(copy.deepcopy(current_point))
                temp_kps.remove(current_point)
                # for p in del_points:
                #     temp_kps.remove(p)
                if current_point in selected_kps:  # 连接上了###错啦
                    break
                current_point, del_points = self.find_kp(current_point, temp_kps, False)
            selected_kps.extend(reversed(temp_rev_kps))
        return selected_kps



    def get_keypoints(self,start,end,step):
        self.keypoints.append((start[0], start[1]))
        obs_index = self.get_obses_between_SE(start,end)
        for i in range(len(obs_index)):
            temp_point_lu = [self.obs_vertex[obs_index[i]][0], self.obs_vertex[obs_index[i]][3]]
            temp_point_ll = [self.obs_vertex[obs_index[i]][0], self.obs_vertex[obs_index[i]][1]]
            temp_point_ru = [self.obs_vertex[obs_index[i]][2], self.obs_vertex[obs_index[i]][3]]
            temp_point_rl= [self.obs_vertex[obs_index[i]][2], self.obs_vertex[obs_index[i]][1]]
            find_lu = False
            find_ll = False
            find_ru = False
            find_rl = False


            while self.is_inside_obs(Node(temp_point_lu)) and temp_point_lu[0] > self.env.x_range[0]:
                temp_point_lu[0] -= step
            if temp_point_lu[0] > self.env.x_range[0]:
                temp_point_lu[0] += step
                while self.is_inside_obs(Node(temp_point_lu)) and temp_point_lu[1] < self.env.y_range[1]:
                    temp_point_lu[1] += step
                if temp_point_lu[1] < self.env.y_range[1]:
                    temp_point_lu[0] -= step
                    find_lu = True

            while self.is_inside_obs(Node(temp_point_ll)) and temp_point_ll[0] > self.env.x_range[0]:
                temp_point_ll[0] -= step
            if temp_point_ll[0] > self.env.x_range[0]:
                temp_point_ll[0] += step
                while self.is_inside_obs(Node(temp_point_ll)) and temp_point_ll[1] > self.env.y_range[0]:
                    temp_point_ll[1] -= step
                if temp_point_ll[1] > self.env.y_range[0]:
                    temp_point_ll[0] -= step
                    find_ll = True


            while self.is_inside_obs(Node(temp_point_ru)) and temp_point_ru[0] < self.env.x_range[1]:
                temp_point_ru[0] += step
            if temp_point_ru[0] < self.env.x_range[1]:
                temp_point_ru[0] -= step
                while self.is_inside_obs(Node(temp_point_ru)) and temp_point_ru[1] < self.env.y_range[1]:
                    temp_point_ru[1] += step
                if temp_point_ru[1] < self.env.y_range[1]:
                    temp_point_ru[0] += step
                    find_ru = True


            while self.is_inside_obs(Node(temp_point_rl)) and temp_point_rl[0] < self.env.x_range[1]:
                temp_point_rl[0] += step
            if temp_point_rl[0] > self.env.x_range[0]:
                temp_point_rl[0] -= step
                while self.is_inside_obs(Node(temp_point_rl)) and temp_point_rl[1] > self.env.y_range[0]:
                    temp_point_rl[1] -= step
                if temp_point_rl[1] > self.env.y_range[0]:
                    temp_point_rl[0] += step
                    find_rl = True

            if find_lu:
                self.append_keypoints((temp_point_lu[0], temp_point_lu[1]))
            if find_ll:
                self.append_keypoints((temp_point_ll[0], temp_point_ll[1]))
            if find_ru:
                self.append_keypoints((temp_point_ru[0], temp_point_ru[1]))
            if find_rl:
                self.append_keypoints((temp_point_rl[0], temp_point_rl[1]))
        self.keypoints.append((end[0], end[1]))

    def is_inside_obs(self, node):
        delta = self.delta #误差

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        return False

    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)
