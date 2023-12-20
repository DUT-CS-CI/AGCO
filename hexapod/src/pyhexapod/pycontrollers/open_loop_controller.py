import numpy as np
import math

'''
_control_signal 方法用于生成一个周期性的、平滑的控制信号，该信号可以用于控制六足机器人的关节运动，以执行特定的步态。
信号的振幅、相位和占空比由输入参数确定，而生成的信号具有更平滑的过渡，以确保机器人的运动更加流畅。
'''
class OpenLoopController:
    ''' 
        Implement an open-loop controller based on periodic signals
        Please see the supplementary information of Cully et al., Nature, 2015
    '''
    def __init__(self, params, array_dim=100):
        self.array_dim = array_dim
        self.trajs = np.zeros(1)

    def step(self, simu): # 根据当前模拟时间 simu.t 来获取控制信号并执行相应的步骤
        assert(self.trajs.shape[0] != 1) # 确保控制信号已正确初始化
        k = int(math.floor(simu.t * self.array_dim)) % self.array_dim # 从控制信号数组中选择的索引
        return self.trajs[:, k] # 从 self.trajs 数组中选择与 k 索引对应的控制信号，并返回这些信号

    def _control_signal(self, amplitude, phase, duty_cycle, array_dim=100):
        '''
        create a smooth periodic function with amplitude, phase, and duty cycle,
        amplitude, phase and duty cycle are in [0, 1]
        '''
        assert(amplitude >= 0 and amplitude <= 1)
        assert(phase >= 0 and phase <= 1)
        assert(duty_cycle >= 0 and duty_cycle <= 1)
        command = np.zeros(array_dim)


        # 创建一个带有平滑过渡的信号，该信号在一段时间内从正值切换到负值，并且在切换点附近有一个平滑的过渡。
        # 将一个信号进行平滑处理，并根据给定的相位偏移进行调整，以获得最终的平滑信号
        # 这种平滑信号通常用于控制系统中，以避免突然的变化和振荡。
        # create a 'top-hat function'
        up_time = array_dim * duty_cycle
        temp = [amplitude if i < up_time else -amplitude for i in range(0, array_dim)] # 在一段时间内保持信号值为正，然后在另一段时间内将信号值变为负

        # smoothing kernel
        kernel_size = int(array_dim / 10)
        kernel = np.zeros(int(2 * kernel_size + 1))
        sigma = kernel_size / 3
        for i in range(0, len(kernel)):
            kernel[i] =  math.exp(-(i - kernel_size) * (i - kernel_size) / (2 * sigma**2)) / (sigma * math.sqrt(math.pi))
        sum = np.sum(kernel)

        # smooth the function
        for i in range(0, array_dim):
            command[i] = 0
            for d in range(1, kernel_size + 1):
                if i - d < 0:
                    command[i] += temp[array_dim + i - d] * kernel[kernel_size - d]
                else:
                    command[i] += temp[i - d] * kernel[kernel_size - d]
            command[i] += temp[i] * kernel[kernel_size]
            for d in range(1, kernel_size + 1):
                if i + d >= array_dim:
                    command[i] += temp[i + d - array_dim] * kernel[kernel_size + d]
                else:
                    command[i] += temp[i + d] * kernel[kernel_size + d]
            command[i] /= sum

        # shift according to the phase
        final_command = np.zeros(array_dim)
        start = int(math.floor(array_dim * phase))
        current = 0
        for i in range(start, array_dim):
            final_command[current] = command[i]
            current += 1
        for i in range(0, start):
            final_command[current] = command[i]
            current += 1

        assert(len(final_command) == array_dim)
        return final_command
