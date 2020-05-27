#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X,y = make_blobs(centers=4,random_state=8)
y = y%2

#认识数据，便于检测数据该使用线性支持向量机还是非线性向量机
#mglearn.discrete_scatter(X[:,0], X[:,1], y)
#plt.xlabel("Feature 0")
#plt.ylabel("Feature 1")
#plt.show()

#测试：线性支持向量机分类。（结果表明不可以使用线性）
from sklearn.svm import LinearSVC
#linear_svm = LinearSVC().fit(X, y)
#mglearn.plots.plot_2d_separator(linear_svm, X)
#mglearn.discrete_scatter(X[:,0], X[:,1], y)
#plt.xlabel("Feature 0")
#plt.ylabel("Feature 1")
#plt.show()

#使用三维散点图展示数据
X_new = np.hstack([X, X[:,1:] ** 2])#添加第二个特征的平方，作为一个新的特征
from mpl_toolkits.mplot3d import Axes3D, axes3d
#figure = plt.figure()
#ax = Axes3D(figure, elev=-152, azim=-26) #3D可视化。创建一个新的matplotlib.figure.Figure并为其添加一个类型为Axes3D的新轴。elev=-152, azim=-26分别与坐标轴及数据点的位置有关
#不是很懂下边三行
mask = y ==0  #首先画出所有y==0的点，然后画出所有y==1的点
#ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
#ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='v', cmap=mglearn.cm2, s=60)
#ax.set_xlabel("feature0")
#ax.set_ylabel("feature1")
#ax.set_zlabel("feature1 ** 2")
#plt.show()

#使用线性模型（在三维空间中即为平面）将数据点分类
linear_svm_3d = LinearSVC().fit(X_new,y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:,0].min()-2, X_new[:,0].max()+2, 50)
yy = np.linspace(X_new[:,1].min()-2, X_new[:,1].max()+2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0]*XX+coef[1]*YY+intercept)/-coef[2]
ax.plot_surface(XX,YY,ZZ, rstride=8, cstride=8, alpha=0.3)#平滑的面
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)#scatter表示散点图
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='v', cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
plt.show()