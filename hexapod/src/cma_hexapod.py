import sys
import numpy as np
import cma
import math
import pyhexapod.simulator as simulator
import pycontrollers.hexapod_controller as ctrl
import pybullet
import time
import multiprocessing

'''
这段代码是用于评估给定六足机器人形态（由URDF文件描述）在特定条件下（如移动、角度等）的适应性。
它尝试模拟机器人行走，并在满足某些条件时标记机器人为 "死亡"，然后返回机器人的水平位置作为适应性分数。这个适应性分数可能会用于后续的实验或优化过程中。
需要注意的是，这段代码的执行可能会很耗时，因为它需要模拟机器人的行走，并且在满足特定条件时才会终止模拟。
'''
def hexapod(x, features):
    t0 = time.perf_counter()
    urdf_file = features[1] # 从参数 features 中获取六足机器人的URDF文件的路径。urdf_file 变量用于存储这个文件路径。
    simu = simulator.HexapodSimulator(gui=False, urdf=urdf_file) # 创建了一个名为 simu 的六足机器人模拟器实例
    controller = ctrl.HexapodController(x) # 创建了一个名为 controller 的六足机器人控制器实例
    dead = False # 创建一个名为 dead 的布尔变量，用于跟踪六足机器人是否处于 "死亡" 状态
    fit = -1e10 # 存储六足机器人的适应性分数
    steps = 3. / simu.dt # 计算一个变量 steps，它代表了六足机器人模拟的步数。这个值基于模拟器的时间步长 simu.dt
    i = 0
    while i < steps and not dead: # 开始一个循环，条件是 i 小于 steps 并且 dead 为 False
        simu.step(controller) # 模拟器执行一次步进，通过控制器 controller 来控制六足机器人的行为
        p = simu.get_pos()[0] # 获取六足机器人的位置信息
        a = pybullet.getEulerFromQuaternion(simu.get_pos()[1]) # 获取六足机器人的方向信息，并从中提取出欧拉角 a，用于描述机器人的方向
        out_of_corridor = abs(p[1]) > 0.5 # 用于检测机器人是否走出了预定的走廊区域
        out_of_angles = abs(a[0]) > math.pi/8 or abs(a[1]) > math.pi/8 or abs(a[2]) > math.pi/8 # 用于检测机器人是否旋转角度过大
        if out_of_angles or out_of_corridor: # 查上述条件，如果任何一个满足，将 dead 设置为 True，表示机器人已经死亡
            dead = True
        i += 1
    fit = p[0] # 更新适应性分数 fit 为机器人的水平位置 p[0]
    #print(time.perf_counter() - t0, " ms", '=>', fit)
    return fit, features    


def write_array(a, f):
    for i in a:
        f.write(str(i) + ' ')

def load(directory, k):
    tasks = []
    centroids = [] # 存储任务描述符
    for i in range(0, k):
        centroid = np.loadtxt(directory + '/lengthes_' + str(i) + '.txt') # 加载当前任务的任务描述符
        urdf_file = directory + '/pexod_' + str(i) + '.urdf'
        centroids += [centroid]
        tasks += [(centroid, urdf_file)]
    return np.array(centroids), tasks

def evaluate(y):
    x, task, func = y # 此处 func = hexpod
    return func(x, task)[0] # 返回fit

# cma
def test_cma(urdf_directory, dim):

    print('loading files...', end='')
    centroids, tasks = load(urdf_directory, 2) # urdf_directory = urdf
    print('data loaded')
    
    opts = cma.CMAOptions() # 创建一个配置对象，用于设置CMA-ES算法的参数
    #for i in opts:
    #    print(i, ' => ', opts[i])
    max_evals = 1e6 # 设置总评估次数
    opts.set('tolfun', 1e-20) # 设置适应度函数值的停止条件，当适应度函数值的变化小于1e-20时，算法将停止
    opts['tolx'] = 1e-20 # 设置控制参数的停止条件，当控制参数的变化小于1e-20时，算法将停止
    opts['verb_disp'] = 1e10 # 设置显示详细信息的间隔，每迭代1e10次后显示一些信息
    opts['maxfevals'] = max_evals / centroids.shape[0] # 设置最大评估次数，以确保算法在达到一定数量的评估次数后停止
    opts['BoundaryHandler'] = cma.BoundPenalty # 边界处理器
    opts['bounds'] = [0, 1] # 设置了CMA-ES算法中的搜索空间的边界
    
    es_vector = []
    # 用于存储多个CMAEvolutionStrategy的实例。在这段代码中，您正在为每个任务（每个任务对应一个机器人控制参数）创建一个CMAEvolutionStrategy的实例。
    # centroids.shape[0] 表示任务的数量，即您要优化的机器人的数量。
    # 每个实例都使用相同的初始解 dim * [0.5]（在搜索空间中所有维度的初始值都设置为0.5），初始的步长（0.5），以及之前设置的优化选项 opts
    for c in range(0, centroids.shape[0]):
        es_vector += [cma.CMAEvolutionStrategy(dim * [0.5], 0.5, opts)]

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores) # 并行地在多个核心上评估候选解

    total_evals = 0
    log = open('cover_max_mean.dat', 'w')
    while total_evals < max_evals:
        #result_file = open('archive_'+ str(total_evals) + '.dat', 'w')
        archive = [] # 存储每个任务的最优适应性值
        for c in range(0, centroids.shape[0]): # 循环迭代每个任务
            centroid = centroids[c, :] # 获取当前任务的 "centroid" 数据，并将其存储在 centroid 变量中
            task = tasks[c] # 获取当前任务的 URDF 文件路径
            def func(angles):
                return hexapod(angles, task)[0]
            solutions = es_vector[c].ask() # 使用 CMA-ES 的 ask 方法生成一批新的解决方案 solutions
#            print(len(solutions))# pop =14
            # 使用了 evaluate 函数，该函数接受 x（控制参数）、task（任务数据）和 hexapod 函数作为输入，并返回适应性值
            s_list = pool.map(evaluate, [(x, task, hexapod) for x in solutions]) # 使用 pool.map 并行地计算每个解决方案的适应性值
            es_vector[c].tell(solutions, s_list) # 使用 CMA-ES 的 tell 方法告知算法有关每个解决方案的适应性值
            total_evals += len(solutions)
            # save to file
            xopt = es_vector[c].result[0] # 从 CMA-ES 算法的结果中获取最优解 xopt 和相应的适应性值 xval
            xval = es_vector[c].result[1]
            # save
            archive += [-xval]
            # write
            #result_file.write(str(-xval) + ' ')
            #write_array(centroid, result_file)
            #write_array(xopt, result_file)
            #result_file.write('\n')
        mean = np.mean(archive) # 计算 archive 中适应性值的均值 (mean)
        max_v = max(archive) # 计算 archive 中适应性值的最大值 (max_v)
        coverage = len(archive) # 计算 archive 中覆盖的任务数量 (coverage)
        log.write(str(total_evals) + ' ' + str(coverage) + ' ' + str(max_v) + ' ' + str(mean) + '\n')
        log.flush()
        print(total_evals)

test_cma(sys.argv[1], 36) #本实验中argv[1]输入urdf
