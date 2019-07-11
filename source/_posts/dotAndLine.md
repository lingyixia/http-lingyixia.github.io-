---
layout: post
title: dotAndLine
date: 2018-12-22 21:02:26
category: 数学
tag: [点,线]
description: 关于点和直线的关系
---
关于点和直线的关系
<!--more-->

##一个知识点
定义: 平面上的三点$A(x1,y1),B(x2,y2),C(x3,y3)$的面积量:
$$S(A,B,C)=\frac{1}{2}  
\left|\begin{array}{}
    x_1 &    y_1    & 1 \\ 
    x_2 &    y_2   & 1\\ 
    x_3 & y_3 & 1 
\end{array}\right| 
$$
其中: 当A、B、C逆时针时S为正的,反之S为负的。 
证明如图: 
![](/img/dotandline.jpg)
也就是说，平面三点一定能写成一个直角梯形减两个直角三角形的形式.
即:
$$S(A,B,C)=\frac{(x_1y_2+x_3y_1+x_2y_3-x_1y_3-x_2y_1-x_3y_2)}{2}$$
正好是上诉行列式.
##一个应用
令矢量的起点为A，终点为B，判断的点为C， 
如果S(A，B，C)为正数，则C在矢量AB的左侧； 
如果S(A，B，C)为负数，则C在矢量AB的右侧； 
如果S(A，B，C)为0，则C在直线AB上