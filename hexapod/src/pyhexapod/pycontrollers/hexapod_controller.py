import numpy as np
import math
from .open_loop_controller import OpenLoopController

'''
这段代码定义了一个六足机器人的控制器，它根据给定的参数生成六足机器人的运动轨迹。这些轨迹将用于控制机器人的关节运动，使其执行特定的步态。
控制器的参数 params 包含了36个元素，每个元素控制一个关节的运动。控制器的目标是生成一组运动轨迹，以使机器人能够按照预定的步态前进。
'''

class HexapodController(OpenLoopController):
    ''' 
        This should be the same controller as Cully et al., Nature, 2015
        example values: ctrl = [1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0.5, 0.5, 0.25, 0.75, 0.5, 1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5, 1, 0.5, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5]
    '''
    def __init__(self, params, array_dim=100):
        super(HexapodController, self).__init__(params, array_dim)
        print('param:') # params 是控制器的参数，是一个包含36个元素的数组，每个元素控制一个关节的运动
        print(params)
        self.trajs = self._compute_trajs(params, array_dim) # 计算六足机器人的运动轨迹。它接受控制器参数 params 和轨迹数组的维度 array_dim 作为输入


    def _compute_trajs(self, params, array_dim):
        trajs = np.zeros((6 * 3, array_dim)) # 控制器首先创建一个全零的轨迹数组 trajs
        k = 0
        print('params:')
        print(params)
        
        for i in range(0, 36, 6):
            for j in range(0, 6): # 裁剪超范围的数据
                if params[i + j] < 0:
                    params[i + j] = 0
                elif params[i + j] > 1:
                    params[i + j] = 1
            # 每条机械腿有 3 个关节
            # 第1个关节调用 _control_signal 方法生成一个控制信号，然后将其乘以0.5并存储在 trajs 数组的相应位置，三个param参数代表相移、振幅和偏移  
            trajs[k, :] =  0.5 * self._control_signal(params[i], params[i + 1], params[i + 2], array_dim)
            # 第2个关节调用 _control_signal 方法生成一个控制信号，三个param参数代表相移、振幅和偏移   
            trajs[k + 1, :] = self._control_signal(params[i + 3], params[i + 4], params[i + 5], array_dim)
            # 第3个关节和第2个关节运动规律相同
            trajs[k + 2, :] = trajs[k + 1, :]
            k += 3
        return trajs * math.pi / 4.0 # 轨迹数组中的值被乘以 π/4.0，以将控制信号转换为弧度