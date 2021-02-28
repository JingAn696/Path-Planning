"""
Plotting tools for Sampling-based algorithms
@author: huiming zhou
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

import env


class Plotting:
    points = []
    obsPoint=[]
    keypoints=[]
    startPoint=None
    endPoint = None
    def __init__(self,name):
        self.env = env.Env()
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.plot_grid(name)

    def ini(self, x_start, x_goal):
        self.xI, self.xG = x_start, x_goal

    def draw_start_end(self):
        plt.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        plt.plot(self.xG[0], self.xG[1], "ys", linewidth=3)

    def animation(self, nodelist, path, name, animation=False):
        self.plot_visited(nodelist, animation)
        self.plot_path(path)

    def animation_connect(self, V1, V2, path):

        self.plot_visited_connect(V1, V2)
        self.plot_path(path)

    def plot_grid(self, name):
        fig, ax = plt.subplots()

        for (ox, oy, w, h) in self.obs_bound:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )


        plt.title(name)
        plt.axis("equal")

    def draw_debug(self):
        for point in self.points:
            plt.plot(point.x, point.y, "rs", linewidth=3)
        # for point in self.obsPoint:
        #     plt.plot(point.x, point.y, "ys", linewidth=3)

        # for i in range (len(self.keypoints)-1):
        #     plt.arrow(self.keypoints[i][0], self.keypoints[i][1], self.keypoints[i+1][0]-self.keypoints[i][0], self.keypoints[i+1][1]-self.keypoints[i][1],
        #               length_includes_head=True, head_width=1, lw=0.4,
        #               color="C2")
        if self.startPoint != None:
            plt.plot(self.startPoint[0], self.startPoint[1], "bs", marker='*', linewidth=3)

        if self.endPoint != None:
           plt.plot(self.endPoint[0], self.endPoint[1], "ys", marker='>', linewidth=3)

        plt.show()
    @staticmethod
    def plot_visited(nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    @staticmethod
    def plot_visited_connect(V1, V2):
        len1, len2 = len(V1), len(V2)

        for k in range(max(len1, len2)):
            if k < len1:
                if V1[k].parent:
                    plt.plot([V1[k].x, V1[k].parent.x], [V1[k].y, V1[k].parent.y], "-g")
            if k < len2:
                if V2[k].parent:
                    plt.plot([V2[k].x, V2[k].parent.x], [V2[k].y, V2[k].parent.y], "-g")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 2 == 0:
                plt.pause(0.001)

        plt.pause(0.01)

    @staticmethod
    def plot_path(path):
        if path != None and  len(path) != 0:# 郭书杰增加了path != None and，以处理没找到路径的情况
            plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
            plt.pause(0.01)
        plt.show()

    @staticmethod
    def set_start_end():
        pos = plt.ginput(2)
        return pos[0],pos[1]
