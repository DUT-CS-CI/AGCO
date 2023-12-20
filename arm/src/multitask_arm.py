  #! /usr/bin/env python3
# Kinematic arm experiment from:
# Mouret JB and Maguire G. (2020) Quality Diversity for Multitask Optimization. Proc of ACM GECCO/

# (so that we do not need to install the module properly)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'pymap_elites')))

import math
import numpy as np
import map_elites.multitask as mt_map_elites
import map_elites.common as cm_map_elites


class Arm:
    def __init__(self, lengths): #lengths 是数组，代表各节的长度
        self.n_dofs = len(lengths) #n_dofs 是有几节
        self.lengths = np.concatenate(([0], lengths)) #lengths 前加了一个0
        self.joint_xy = []

    def fw_kinematics(self, p):
        from math import cos, sin, pi, sqrt
        assert(len(p) == self.n_dofs) #assert 如果它的条件返回错误，则终止程序执行。即确保 角的个数 == 节数。p[] 代表各个α
        p = np.append(p, 0)
        self.joint_xy = []
        mat = np.matrix(np.identity(4)) # identity 创建4*4 方阵
        for i in range(0, self.n_dofs + 1):
            m = [[cos(p[i]), -sin(p[i]), 0, self.lengths[i]],
                 [sin(p[i]),  cos(p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            self.joint_xy += [v[0:2].A.flatten()] #变平
        return self.joint_xy[self.n_dofs], self.joint_xy



def fitness_arm(angles, task):
    angular_range = task[0] / len(angles)
    # angles表示传入的基因型，task控制任务，第一维用以规定角度范围，第二维用以规定杆长
    lengths = np.ones(len(angles)) * task[1] / len(angles) 
    target = 0.5 * np.ones(2) #目标点
    a = Arm(lengths)
    command = (angles - 0.5) * angular_range * math.pi * 2 # 基因分布为0-1，减去0.5是为了角度存在正负
    ef, _ = a.fw_kinematics(command) # ef 为 pd
    f = -np.linalg.norm(ef - target)
    return f # 得fitness

if len(sys.argv) == 1 or ('help' in sys.argv):
    print("Usage: \"python3 ./examples/multitask_arm.py dimension [no_distance]\"")
    exit(0)


dim_x = int(sys.argv[1])  #python 后的第一个参数

for i in range(25):
        # dim_map, dim_x, function
        px = cm_map_elites.default_params.copy()
        px["dump_period"] = 2000 #2000 代记录一次
        px["min"] = np.zeros(dim_x) # 规约在x的dim中，
        px["max"] = np.ones(dim_x)
        px["parallel"] = False
        
        # number of tasks
        n_tasks = 2000
        loc='cover_max_mean'+str(i)+".dat"
        # example : create centroids using a CVT (you can also create them randomly)
        # -> this is a numpy array with rows=number of tasks, and cols=dimension of the task
        c,data,cluster_number = cm_map_elites.cvt(n_tasks, 2, 12000, True,dim_x) 
        # CVT-based version
        #if len(sys.argv) == 2 or sys.argv[2] == 'distance':
            # dim_x = 2,f 是评估函数，c 是计算的质心位置，centroids是 5000 个质心
        archive = mt_map_elites.compute(dim_x=dim_x, f=fitness_arm, centroids=c, num_evals=1e6, params=px, log_file=open(loc, 'w+'), datas = data,cluster_number=cluster_number, log_file1=open('percent.dat', 'w'))
#else:
    # no distance:
 #   archive = mt_map_elites.compute(dim_x=dim_x, f=fitness_arm, tasks=c, num_evals=1e5, params=px, log_file=open('cover_max_mean.dat', 'w'), datas= data,cluster_number=cluster_number)
