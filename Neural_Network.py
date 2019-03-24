
import math
import random
import numpy as np

random.seed(0)

def rand(a, b):
	# 生成区间[a, b)内的随机数
	return (b - a) * random.random() + a

def makeMatrix(I, J, fill=0.0):
	# 生成大小 I*J 的0矩阵
	m = np.zeros((I,J))
	# m = []
	# for i in range(I):
	# 	m.append([fill] * J)
	return m

def sigmoid(x):
	# 函数 sigmoid，这里采用 tanh，因为看起来要比标准的 1/(1+e^-x) 漂亮些
	return math.tanh(x)

def dsigmoid(y):
	# 函数 sigmoid 的派生函数, 为了得到输出 (即：y)
	return 1.0 - math.tanh(y) ** 2

class NN:
	''' 三层反向传播神经网络 '''

	def __init__(self, ni, nh, no):
		# 输入层、隐藏层、输出层的节点（数）
		self.ni = ni + 1  # 增加一个偏差节点 #3
		self.nh = nh  # 4
		self.no = no  # 1

		# 激活神经网络的所有节点（向量）
		self.ai = np.ones(self.ni)
		self.ah = np.ones(self.nh)
		self.ao = np.ones(self.no)

		# 建立权重（矩阵）
		self.wi = makeMatrix(self.ni, self.nh)
		self.wo = makeMatrix(self.nh, self.no)

		# 设为随机值
		for i in range(self.ni):
			for j in range(self.nh):
				self.wi[i][j] = rand(-0.2, 0.2)
		for j in range(self.nh):
			for k in range(self.no):
				self.wo[j][k] = rand(-2.0, 2.0)

		self.ci = makeMatrix(self.ni, self.nh)
		self.co = makeMatrix(self.nh, self.no)

	# 前向传递
	def update(self, inputs):
		if len(inputs) != self.ni - 1:
			raise ValueError('与输入层节点数不符！')

		# 输入层
		for i in range(self.ni - 1):
			self.ai[i] = inputs[i]
		# 隐藏层
		for j in range(self.nh):  # self.nh = 4
			sum = 0.0
			for i in range(self.ni):  # self.ni = 3
				sum = sum + self.ai[i] * self.wi[i][j]
			self.ah[j] = sigmoid(sum)

		# 输出层
		for k in range(self.no):
			sum = 0.0
			for j in range(self.nh):
				sum = sum + self.ah[j] * self.wo[j][k]
			self.ao[k] = sigmoid(sum)
		return self.ao[:]

	# 反向传播
	def backPropagate(self, targets, N, M):
		# 计算输出层的误差
		output_deltas = np.zeros(self.no)
		for k in range(self.no):
			output_deltas[k] = dsigmoid(self.ao[k]) * (targets[k] - self.ao[k])

		# 计算隐藏层的误差
		hidden_deltas = np.zeros(self.nh)
		for j in range(self.nh):
			error = 0.0
			for k in range(self.no):
				error = error + output_deltas[k] * self.wo[j][k]
			hidden_deltas[j] = dsigmoid(self.ah[j]) * error

		# 更新输出层权重
		for j in range(self.nh):
			for k in range(self.no):
				change = output_deltas[k] * self.ah[j]
				self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
				self.co[j][k] = change

		# 更新输入层权重
		for i in range(self.ni):
			for j in range(self.nh):
				change = hidden_deltas[j] * self.ai[i]
				self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
				self.ci[i][j] = change

		# 计算误差
		error = 0.0
		for k in range(len(targets)):
			error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
		return error

	def predict(self, patterns):
		for p in patterns:
			print(p[0], '->', self.update(p[0]))

	def weights(self):
		print('输入层权重:')
		for i in range(self.ni):
			print(self.wi[i])
		print('输出层权重:')
		for j in range(self.nh):
			print(self.wo[j])

	def train(self, patterns, iterations=3000, N=0.5, M=0.1):
		# N: learning rate
		# M: 动量因子(momentum factor)
		for i in range(iterations):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.update(inputs)  # 前向传递
				error = error + self.backPropagate(targets, N, M)  # 反向传播结果,把结果回带到
			if i % 100 == 0:
				print('误差 %-.5f' % error)


if __name__ == '__main__':
	# 一个演示：教神经网络学习逻辑异或（XOR）------------可以换成你自己的数据试试
	pat = [[[0, 0], [0]],
		[[0, 1], [1]],
		[[1, 0], [1]],
		[[1, 1], [0]]]

	n = NN(2, 2, 1)
	n.train(pat)
	n.weights()
	print('预测结果')
	n.predict(pat)