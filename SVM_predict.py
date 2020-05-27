#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
X,y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)

#mglearn.plots.plot_2d_separator(svm, X, eps=0.5)     #画出SVM线
#mglearn.discrete_scatter(X[:,0], X[:,1], y)          #画出散点图
#画出支持向量
#sv = svm.support_vectors_
#支持向量的类别标签由dual_coef_的正负号给出
#sv_labels = svm.dual_coef_.ravel()>0
#mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=15, markeredgewidth=3)  #画出支持向量的点
#plt.xlabel("Feature 0")
#plt.ylabel("Feature 1")
#plt.show()

#修改参数c和gamma的值，观察图的变化
fig, axes = plt.subplots(3,3, figsize = (15,10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a,gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes[0,0].legend(["class 0", "class 1", "sv class 0", "sv class 1"], ncol=4, loc=(0.9, 1.2))
#plt.show()

#实际中，SVM需要对数据进行预处理
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
min_on_training = X_train.min(axis = 0)#计算训练集中的每个特征的最小值
range_on_trainging = (X_train - min_on_training).max(axis=0) #计算训练集中每个特征的范围
X_train_scaled = (X_train - min_on_training)/range_on_trainging  #数据减去最小值，然后除以范围
#检验数据是否处理为0-1之间的数
print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
print("Maxnimum for each feature\n{}".format(X_train_scaled.max(axis=0)))
X_test_scaled = (X_test - min_on_training)/range_on_trainging   #对测试集进行同样的数据预处理
svc = SVC(C=1000)  #设置正则化参数
svc.fit(X_train_scaled, y_train)
print("Training accuracy:{:.3f}".format(svc.score(X_train_scaled,y_train)))
print("Test accuracy:{:.3f}".format(svc.score(X_test_scaled,y_test)))


