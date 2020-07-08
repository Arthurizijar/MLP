# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:18:35 2020

@author: Niezhijie
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class logistic_regression(object):
    
    def __init__(self, n_in, n_out):
        self.W = np.zeros((n_in, n_out))
        self.b = np.zeros((n_out,))
        
    def back_propagate(self, X, Y):
        m = X.shape[1]
        h = sigmoid(np.dot(X, self.W) + self.b)
        #print(Y, Y.T)
        cost = -1/m * np.sum(Y * np.log(h) + (1-Y) * np.log(1-h))
        cost = np.squeeze(cost)
        dw = 1/m * np.dot(X.T, (h-Y))
        db = 1/m * np.sum(h-Y)
        grads = [dw, db]
        return grads, cost
    
    def train(self, X, Y, num_iter, lr, print_flag = False):
        for i in range(num_iter):
            #针对每次迭代，更新代价函数和梯度
            grads, cost = self.back_propagate(X, Y)
            self.W = self.W - lr * grads[0]
            self.b = self.b - lr * grads[1]
            if print_flag and i % 5000 == 0:
                print("{} 轮迭代损失函数为{}".format(i, cost))
            if print_flag and i % 10000 == 0:
                y_pred_train = self.predict(X)
                print("{} 轮迭代训练集正确率为: {} %".format(
                        i, 100 - np.mean(np.abs(y_pred_train - Y)) * 100))
    
    def predict(self, X):
        m = X.shape[0]
        Y_pred = np.zeros((m, 1))
        h = sigmoid(np.dot(X, self.W) + self.b)
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
    #前128列为句子向量化后数据，最后1列为标签
    data = df.iloc[:, 0:128].values
    label = df.iloc[:, 128].values.reshape((data.shape[0],1))
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2)
    classifier = logistic_regression(data.shape[1], 1)
    classifier.train(x_train, y_train, 50000, 0.001, print_flag = True)
    y_pred_test = classifier.predict(x_test)
    print("测试集正确率为: {} %".format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100))
        


    
    
    