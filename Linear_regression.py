# -*- coding: utf-8 -*-
"""
Linear Regression with GD
Created on 2019.3.14
@author:Mx
"""

import numpy as np


def load_data(path):
    """
    读取数据文件
    :param path: 数据文件路径
    :return:
    """
    pass


def cal_loss(X, y, theta):
    """
    损失函数
    :param X: 变量矩阵
    :param y: 标签向量
    :param theta: 权重参数
    :return: 损失
    """
    delt = X * theta - y
    loss = 0.5 * delt.transpose() * delt
    return loss


def train(X, y, alpha, max_iter):
    """
    对数据进行训练，采用所有数据进行梯度下降
    :param X: 变量矩阵
    :param y: 标签向量
    :param alpha: 学习率
    :param max_iter: 最大迭代次数
    :return: 权重参数
    """
    theta = np.ones((X.shape[1], 1))
    for i in range(max_iter):
        delt = X.tranpose() * (X * theta - y)
        theta = theta - alpha * delt
        print(cal_loss(X, y, theta))
    return theta


if __name__ == '__main__':
    pass
