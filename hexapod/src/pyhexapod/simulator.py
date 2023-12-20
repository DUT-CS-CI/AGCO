import os
import time
import math
from timeit import default_timer as timer
import subprocess 
import time
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

from pycontrollers.hexapod_controller import HexapodController

class HexapodSimulator:
	def __init__(self, gui=False, # 指定是否在模拟中启用图形用户界面 
				urdf=(os.path.dirname(os.path.abspath(__file__))+'/urdf/pexod.urdf'), # 六足机器人的URDF文件的路径
				dt = 1./240.,  # the default for pybullet (see doc) 模拟时间步长，默认为1/240秒
				control_dt=0.05, # 控制器的时间步长，默认为0.05秒
				video=''): # 视频输出文件的路径
		self.GRAVITY = -9.81 # 设置了模拟环境中的重力
		self.dt = dt # 设置了模拟中的时间步长（时间间隔）
		self.control_dt = control_dt # 设置了控制器的时间步长（时间间隔）
		# we call the controller every control_period steps
		self.control_period = int(control_dt / dt) # 计算了控制器的调用周期，即多少个模拟步长（模拟周期）后调用一次控制器
		self.t = 0 # 初始化模拟的时间为0，用于跟踪模拟的运行时间
		self.i = 0 # 初始化计数器 i 为0，用于跟踪当前模拟步骤的数量
		self.safety_turnover = True # 表示是否启用了安全翻转
		self.video_output_file = video # 设置了模拟过程中保存视频的文件名

		# the final target velocity is computed using:
		# kp*(erp*(desiredPosition-currentPosition)/dt)+currentVelocity+kd*(m_desiredVelocity - currentVelocity).
		# here we set kp to be likely to reach the target position
		# in the time between two calls of the controller
		self.kp = 1./12.# * self.control_period 设置了控制器中的比例增益参数 kp，这是用于计算目标位置与当前位置之间的误差的增益
		self.kd = 0.4 # 设置了控制器中的微分增益参数 kd，这是用于控制目标速度与当前速度之间的差异的增益
		# the desired position for the joints
		self.angles = np.zeros(18) # 初始化一个长度为18的数组 angles，用于存储机器人关节的目标位置
		# setup the GUI (disable the useless windows)
		if gui:
			self.physics = bc.BulletClient(connection_mode=p.GUI)
			self.physics.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
			self.physics.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
			self.physics.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
			self.physics.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
			self.physics.resetDebugVisualizerCamera(cameraDistance=1,
                            						cameraYaw=20,
			                             			cameraPitch=-20,
            			                			cameraTargetPosition=[1, -0.5, 0.8])
		else:
			self.physics = bc.BulletClient(connection_mode=p.DIRECT)

		self.physics.setAdditionalSearchPath(pybullet_data.getDataPath()) # 设置PyBullet的数据路径，以便加载附加资源和URDF文件
		self.physics.resetSimulation() # 重置模拟环境，清除之前的仿真状态，以便开始一个新的模拟
		self.physics.setGravity(0,0,self.GRAVITY) # 设置模拟环境中的重力。这一行代码将重力设置为 self.GRAVITY
		self.physics.setTimeStep(self.dt) # 设置模拟的时间步长
		self.physics.setPhysicsEngineParameter(fixedTimeStep=self.dt) # 设置物理引擎参数，确保物理引擎的迭代与时间步长一致
		self.planeId = self.physics.loadURDF("plane.urdf") # 加载了一个平面的URDF模型，该模型用于模拟地面或地面上的平台

		start_pos = [0,0,0.15] # 机器人的初始位置的设置
		start_orientation = self.physics.getQuaternionFromEuler([0.,0,0]) # 机器人的初始方向的设置
		self.botId = self.physics.loadURDF(urdf, start_pos, start_orientation) # 加载了六足机器人的URDF模型
		self.joint_list = self._make_joint_list(self.botId) # 接受机器人的ID作为参数，查找并返回所有关节的ID

		# bullet links number corresponding to the legs
		self.leg_link_ids = [17, 14, 2, 5, 8, 11] # 列表，包含六足机器人身体上连接到腿部的链接的ID
		self.descriptor = {17 : [], 14 : [], 2 : [], 5 : [], 8 : [], 11 : []} # 字典，用于存储与六足机器人的不同链接（通过ID标识）相关的描述信息

		# video makes things much slower
		if (video != ''):
			self._stream_to_ffmpeg(self.video_output_file) # 输出视频

		# put the hexapod on the ground (gently)
		self.physics.setRealTimeSimulation(0) # 禁用了实时仿真模式，使仿真以更快的速度进行
		jointFrictionForce=1 # 定义了关节摩擦力的变量，设置为1
		for joint in range (self.physics.getNumJoints(self.botId)): # 遍历六足机器人的所有关节
			self.physics.setJointMotorControl2(self.botId, joint, 
				p.POSITION_CONTROL,
				force=jointFrictionForce) # 对每个关节应用了关节摩擦力
		for t in range(0, 100): # 计时器循环，它模拟了一段时间，包括100个时间步
			self.physics.stepSimulation() # 使仿真环境向前迈进一个时间步
			self.physics.setGravity(0,0, self.GRAVITY) # 设置了模拟环境中的重力，确保重力方向保持不变


	# 用于销毁模拟器对象
	def destroy(self):
		try:
			self.physics.disconnect()
			if self.video_output_file != '':
				self.ffmpeg_pipe.stdin.close()
				self.ffmpeg_pipe.stderr.close()
				self.ffmpeg_pipe.wait()
		except p.error as e:
			print("Warning (destructor of simulator):", e)


	def reset(self):
		assert(0), "not working for now"
		self.t = 0
		self.physics.resetSimulation()
#		self.physics.restoreState(self._init_state)
		

	# 获取六足机器人的位置和方向信息
	def get_pos(self):
		'''
		Returns the position list of 3 floats and orientation as list of 4 floats in [x,y,z,w] order.
		Use p.getEulerFromQuaternion to convert the quaternion to Euler if needed.
		'''
		return self.physics.getBasePositionAndOrientation(self.botId)


	# 六足机器人仿真的核心，它在每个时间步更新机器人的运动状态，检查安全性和视觉记录，以及与地面接触情况
	def step(self, controller):
		if self.i % self.control_period == 0:
			self.angles = controller.step(self)
		self.i += 1
		
		# 24 FPS dt =1/240 : every 10 frames
		if self.video_output_file != '' and self.i % (int(1. / (self.dt * 24))) == 0: 
			camera = self.physics.getDebugVisualizerCamera()
			img = p.getCameraImage(camera[0], camera[1], renderer=p.ER_BULLET_HARDWARE_OPENGL)
			self.ffmpeg_pipe.stdin.write(img[2].tobytes())

		#Check if roll pitch are not too high
		error = False
		self.euler = self.physics.getEulerFromQuaternion(self.get_pos()[1])
		if(self.safety_turnover):
			if((abs(self.euler[1]) >= math.pi/2) or (abs(self.euler[0]) >= math.pi/2)):
				error = True

		# move the joints
		missing_joint_count = 0
		j = 0
		for joint in self.joint_list:
			if(joint==1000):
				missing_joint_count += 1
			else:
				info = self.physics.getJointInfo(self.botId, joint)
				lower_limit = info[8]
				upper_limit = info[9]
				max_force = info[10]
				max_velocity = info[11]
				pos = min(max(lower_limit, self.angles[j]), upper_limit)
				self.physics.setJointMotorControl2(self.botId, joint,
					p.POSITION_CONTROL,
					positionGain=self.kp,
					velocityGain=self.kd,
					targetPosition=pos,
					force=max_force,
					maxVelocity=max_velocity)
			j += 1

		#Get contact points between robot and world plane
		contact_points = self.physics.getContactPoints(self.botId,self.planeId)
		link_ids = [] #list of links in contact with the ground plane
		if(len(contact_points) > 0):
			for cn in contact_points:
				linkid= cn[3] #robot link id in contact with world plane
				if linkid not in link_ids:
					link_ids.append(linkid)
		for l in self.leg_link_ids:
			cns = self.descriptor[l]
			if l in link_ids:
				cns.append(1)
			else:
				cns.append(0)
			self.descriptor[l] = cns

		# don't forget to add the gravity force!
		self.physics.setGravity(0, 0, self.GRAVITY)

		# finally, step the simulation
		self.physics.stepSimulation()
		self.t += self.dt
		return error

	def get_joints_positions(self):
		''' return the actual position in the physics engine'''
		p = np.zeros(len(self.joint_list))
		i = 0
		# be careful that the joint_list is not necessarily in the same order as 
		# in bullet (see make_joint_list)
		for joint in self.joint_list:
			p[i] = self.physics.getJointState(self.botId, joint)[0]
			i += 1
		return p


	# 将六足机器人仿真的图像数据流传输给FFmpeg
	def _stream_to_ffmpeg(self, fname): 
		camera = self.physics.getDebugVisualizerCamera()
		command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s',  '{}x{}'.format(camera[0], camera[1]),
                '-pix_fmt', 'rgba',
                '-r', str(24),
                '-i', '-',
                '-an',
                '-vcodec', 'mpeg4',
				'-vb', '20M',
                fname]
		print(command)
		self.ffmpeg_pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


	# 通过匹配关节名称和已知的名称列表，创建了一个包含六足机器人各个关节编号的列表
	def _make_joint_list(self, botId):
		joint_names = [b'body_leg_0', b'leg_0_1_2', b'leg_0_2_3',
		b'body_leg_1', b'leg_1_1_2', b'leg_1_2_3',
		b'body_leg_2', b'leg_2_1_2', b'leg_2_2_3',
		b'body_leg_3', b'leg_3_1_2', b'leg_3_2_3',
		b'body_leg_4', b'leg_4_1_2', b'leg_4_2_3',
		b'body_leg_5', b'leg_5_1_2', b'leg_5_2_3',
		]
		joint_list = []
		for n in joint_names:
			joint_found = False
			for joint in range (self.physics.getNumJoints(botId)):
				name = self.physics.getJointInfo(botId, joint)[1]
				if name == n:
					joint_list += [joint]
					joint_found = True
			if(joint_found==False):
				joint_list += [1000] #if the joint is not here (aka broken leg case) put 1000
		return joint_list


# 定义了一个名为 ctrl 的列表，其中包含了一系列控制参数，这些参数将被用于六足机器人的控制
# for an unkwnon reason, connect/disconnect works only if this is a function
def test_ref_controller():
	# this the reference controller from Cully et al., 2015 (Nature)
	# 定义了一个名为 ctrl 的列表，其中包含了一系列控制参数，这些参数将被用于六足机器人的控制
	ctrl = [1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0.5, 0.5, 0.25, 0.75, 0.5, 1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5, 1, 0.5, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5]
	simu = HexapodSimulator(gui=False) # 创建了一个名为 simu 的六足机器人模拟器对象，将 gui 参数设置为 False，以禁用可视化界面
	controller = HexapodController(ctrl) # 创建了一个名为 controller 的六足机器人控制器对象，使用之前定义的 ctrl 参数
	for i in range(0, int(3./simu.dt)): # seconds 模拟了3秒钟的时间，其中 simu.dt 表示模拟器的时间步长
		simu.step(controller) # 调用了模拟器的 step 方法，将 controller 作为参数传递，以执行控制器的一步操作
	print("=>", simu.get_pos()[0]) # 打印出当前六足机器人的位置信息（机器人的位置信息中的 X、Y 和 Z 坐标）
	simu.destroy() # 销毁了模拟器对象

if __name__ == "__main__":
	# we do 10 simulations to get some statistics (perfs, reproducibility)
	for k in range(0, 10): # 将模拟器测试运行10次以获取一些性能和可重复性统计信息
		t0 = time.perf_counter() # 记录了测试开始的时间
		test_ref_controller()# this needs to be in a sub-function... 调用了 test_ref_controller 函数，执行六足机器人的控制器测试
		print(time.perf_counter() - t0, " ms") # 计算并打印出测试所花费的时间
	