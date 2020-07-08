# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:45:18 2020

@author: Niezhijie
"""

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import Logistic_Regression as LR

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

#求导函数
def derivative(z, fn):
    if fn == 'sigmoid':
        return sigmoid(z)*(1-sigmoid(z))
    elif fn == 'tanh':
        return tanh(z)*(1-tanh(z))
    
#隐藏层，仅进行初始化
class hidden_layer(object):
    def __init__(self, n_in, n_out, rng, activation = 'sigmoid'):
        W_values = np.asarray(
            rng.uniform(
                low = -np.sqrt(6. / (n_in + n_out)),
                high = np.sqrt(6. / (n_in + n_out)),
                size = (n_in, n_out)
            )
        )
        if activation == 'sigmoid':
            W_values *= 4
        self.W = W_values
        self.b = np.zeros((n_out, ))
        self.activation = activation
    

class MLP(object):
    def __init__(self, n_in, n_hidden_1, n_hidden_2, n_hidden_3, n_out, rng):
        #三层隐藏层
        self.hidden_1 = hidden_layer(n_in, n_hidden_1, rng)
        self.hidden_2 = hidden_layer(n_hidden_1, n_hidden_2, rng)
        self.hidden_3 = hidden_layer(n_hidden_2, n_hidden_3, rng)
        #一层逻辑回归层
        self.logistic = LR.logistic_regression(n_hidden_3, n_out)
        #将连接系数、偏差和激活函数置于列表
        self.W = [self.hidden_1.W, self.hidden_2.W, self.hidden_3.W, self.logistic.W]
        self.b = [self.hidden_1.b, self.hidden_2.b, self.hidden_3.b, self.logistic.b]
        self.activation = [self.hidden_1.activation, self.hidden_2.activation, self.hidden_3.activation, 'sigmoid']
    
    def train(self, X, Y, L1_reg, L2_reg, num_iter, batch_size, lr, regularization = "L2", print_flag = False):
        self.l1_reg = L1_reg
        self.l2_reg = L2_reg
        train_data = list(zip(X, Y))
        for i in range(num_iter):
            #打乱数据并抽取batch
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+batch_size] for k in range(0, len(train_data), batch_size)]
            #对每个batch更新连接系数和偏差
            for mini_batch in mini_batches:
                self.batch_update(mini_batch, lr, X.shape[0], regularization)
            #每隔100个epoch测试正确率
            if print_flag and i % 100 == 0:
                y_pred_train = self.predict(X)
                print("{} 轮迭代训练集正确率为: {} %".format(
                        i, 100 - np.mean(np.abs(y_pred_train - Y)) * 100))
                
    def batch_update(self, mini_batch, lr, n, regularization):
        temp_w = [np.zeros(w.shape) for w in self.W]
        temp_b = [np.zeros(b.shape) for b in self.b]
        for x, y in mini_batch:
            delta_b, delta_w = self.back_propogation(x.reshape((1, x.shape[0])), y.reshape((1, 1)))
            temp_b = [nb + dnb for nb, dnb in zip(temp_b, delta_b)]
            temp_w = [nw + dnw for nw, dnw in zip(temp_w, delta_w)]
            self.b = [b- (lr / len(mini_batch)) * nb for b, nb in zip(self.b, temp_b)]
        if regularization == "L2":
            self.W = [(1 - lr * (self.l2_reg / n)) * w - (lr / len(mini_batch)) * nw 
                   for w, nw in zip(self.W, temp_w)]
        elif regularization == "L1":
            self.W = [w - lr * self.l1_reg * np.sign(w) / n - (lr / len(mini_batch)) * nw 
                   for w, nw in zip(self.W, temp_w)]
        #print(self.W)
    
    def back_propogation(self, x, y):
        temp_w = [np.zeros(w.shape) for w in self.W]
        temp_b = [np.zeros(b.shape) for b in self.b]
        activation = x
        activations = [x]
        zs = []
        #针对每个数据进行计算并保存中间数据
        for b, w, a in zip(self.b, self.W, self.activation):
            z = np.dot(activation, w) + b
            zs.append(z)
            if a == 'sigmoid':
                activation = sigmoid(z)
            elif a == 'tanh':
                activation = tanh(z)
            activations.append(activation)
        #计算最后一层的偏导
        dell = activations[-1] - y
        temp_w[-1] = np.dot(dell, activations[-2]).transpose()
        temp_b[-1] = dell
        #往前依次及计算偏导
        for l in range(2, len(self.W)+1):
            dell = np.dot(self.W[-l+1], dell) * derivative(zs[-l], self.activation[-l]).transpose()
            temp_w[-l] = np.dot(dell, activations[-l-1]).transpose()
            temp_b[-l] = dell.transpose()
        return (temp_b, temp_w)

    def predict(self, X):
        m = X.shape[0]
        Y_pred = np.zeros((m, 1))
        h = X
        for b, w, a in zip(self.b, self.W, self.activation):
            h = np.dot(h, w) + b
            if a == 'sigmoid':
                h = sigmoid(h)
            elif a == 'tanh':
                h = tanh(h)
        for i in range(m):
            if h[i, 0] > 0.5:
                Y_pred[i, 0] = 1
            else:
                Y_pred[i, 0] = 0
        return Y_pred
        
if __name__ == '__main__':
    data_path = "./data/data_vector.csv"
    df = pd.read_csv(data_path, header=None)
    df = df.sample(frac=1)
    data = df.iloc[:, 0:128].values
    label = df.iloc[:, 128].values.reshape((data.shape[0],1))
    rng = np.random.RandomState(666)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2)
    #四层网络的size分别为(n_in, 64), (64,32), (32,16), (16, 1)
    classifier = MLP(x_train.shape[1], 64, 32, 16, 1, rng)
    classifier.train(x_train, y_train, 0.00, 0.001, 2500, 20, 0.005, regularization = 'L2', print_flag = True)
    y_pred_test = classifier.predict(x_test)
    print("测试集正确率为: {} %".format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100))
    




            
        