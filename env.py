"""
Environment for rrt_2D
@author: huiming zhou and guoshu jie
"""
'''
生成地图环境
包括地图边界、圆形障碍物和长方形障碍物
'''

class Env:
    def __init__(self):
        self.x_range = (0, 60)
        self.y_range = (0, 60)#

        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():#地图边界
        obs_boundary = [
            [0, 0, 1, 60],#左  (x, y, w, h)
            [0, 60, 60, 1],#上
            [1, 0, 60, 1],#右
            [60, 1, 1, 60]#下
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():#格式如下#[x,y,width,height]
        obs_rectangle = [
            #离散障碍物 Environment：
            # #------begin
            # [14, 12, 10, 3],
            # [18, 22, 8,10],
            # [26, 7, 2, 12],
            # [32, 14, 10, 2],
            # [40, 21, 8, 6],
            # [9, 30, 2, 12],
            # [20, 20, 10, 2],
            #
            # [23, 50, 5, 6],
            # [4, 3, 5, 5],
            # [23, 40, 7, 6],
            # [31, 37, 4, 6],
            # [45, 14, 10, 2],
            # [47, 40, 8, 6],
            # [40, 50, 2, 10],
            # [28, 30, 10, 2],
            #
            # [29, 50, 10, 2],
            # [47, 40, 8, 6],
            # [52, 50, 2, 10],
            # [38, 40, 10, 2]
            # #----end

            # #Narrow Passage 1 Environment：
            # #----begin
            # [20, 1, 20, 29],
            # [20, 31, 20, 29]
            # #----end
            # #Narrow Passage 2 Environment：
            # #----begin
            # [20, 1, 10,44],
            # [30, 1, 10, 30],
            # [20, 48, 20, 12],
            # [32,34, 8, 14]
            # #----end

            # # # 迷宫1 Environment：
            # # # ----begin
            # [15, 10, 20, 5],
            # [15, 15, 5, 5],
            #
            # [15, 20, 20, 5],
            # [30, 25, 5, 5],
            #
            # [15, 30, 20, 5],
            # [15, 35, 5, 5],
            #
            # [15, 40, 20, 5],
            # [30, 45, 5, 5],
            # [15, 50, 20, 5]
            # # # ----end
            # #
            # # Z字形 Environment：
            # # ----begin
            # [1,1, 59, 5],
            # [10, 10, 50, 5],
            # [1, 20, 50, 5],
            # [10, 30, 50, 5],
            # [1, 40, 50, 5],
            # [10, 50, 50, 5],
            # [54,40,5,5],
            # [26,55,6,3]
            # # ----end
            #
            #房间1 Environment：
            # # # ----begin
            # [5, 10, 20, 5],
            # # [20, 15, 5, 5],
            #
            # [5, 15, 5, 15],
            # [20, 20, 5, 10],
            #
            # [5, 30, 20, 5],
            #
            # [35, 20, 20, 5],
            # [35, 25, 5, 10],
            #
            # [50, 25, 5, 15],
            # # [35, 25, 5, 5],
            #
            # [35, 40, 20, 5]
            # # ----end

            # 房间2 Environment：
            # # # ----begin
            [10, 50, 40, 2],
            [20, 40, 2, 10],
            [10, 38, 12, 2],
            [10, 18, 2, 20],
            [12, 26, 20, 2],
            [37, 10, 2, 25],
            [25, 10, 12, 2],
            [25, 1, 2, 9],
            [50, 30, 2, 22],
            [45, 15, 15, 2],
            # # ----end
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():#格式如下#[x,y,raidus]
        obs_cir = [
            # [7, 12, 3],
            # [46, 20, 4],
            # [15, 5, 4],
            # [37, 7, 3],
            # [37, 23, 3],
            # [45, 32, 3],
            # [5,20, 3],
            # [24, 38, 3]
        ]

        return obs_cir
