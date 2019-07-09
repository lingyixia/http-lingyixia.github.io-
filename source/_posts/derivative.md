---
title: 方向导数和梯度
date: 2019-06-14 16:54:10
category: 数学
tags:
---

#方向导数
>>方向导数是一个数值

函数$f(x,y)$在点$(x_0,y_0)$处沿$ \overrightarrow{l}$方向的**方向导数**定义为:
$$
\frac{\partial f}{\partial l}=\lim_{t \rightarrow 0^+} \frac{f(x_0+t\cos \alpha,y_0+tcos \beta) - f(x_0,y_0)}{t} \tag{1}
$$
其中$\alpha,\beta,\gamma$分别是$\overrightarrow{l}$三个相应方向的余弦值.  
(1)公是**方向导数**的定义,要证明的是该式子等于:
$$
\frac{\partial f}{\partial l}=f_x(x_0,y_0) \cos \alpha+f_y(x_0,y_0) \cos \beta     \tag{2}
$$
证明:
$$
\begin{align}
\frac{\partial f}{\partial l} &=\lim_{t\rightarrow0^+}\frac{f(x_0+t\cos \alpha,y_0+tcos \beta) - f(x_0,y_0)}{t} \\
&=\lim_{t\rightarrow 0^+} \frac{f(x_0+t\cos \alpha,y_0+t\cos\beta)-f(x_0,y_0+t\cos \beta)}{t}+\frac{f(x_0,y_0+t\cos \beta) -f(x_0,y_0)}{t}\\
&=\lim_{t\rightarrow 0^+}\frac{f_x(\xi_x,y_0+t\cos\beta)t\cos\alpha}{t} + \frac{f_y(x_0,\xi_y)t\cos\beta}{t} \qquad \xi_x\in[x_0,x_0+t\cos\alpha] \quad \xi_y\in[y_0,y_0+tcos\beta](拉格朗日中值定理) \\
&=\lim_{t\rightarrow 0 +}f_x(\xi_x,y_0+t\cos\beta) \cos\alpha + f_y(x_0,\xi_y)\cos\beta \\
&=f_x(x_0,y_0)\cos\alpha+f_y(x_0,y_0)\cos\beta
\end{align}
$$

>>方向导数的直观意义是函数在该点的偏导数向量在此方向上的投影
$$
\frac{\partial f}{\partial l} =(f_x(x_0,y_0),f_y(x_0,y_0)) ·（\cos\alpha, \cos\beta） = |(cos\alpha,cos\beta)|·|(f_x(x_0,y_0),f_y(x_0,y_0))| \cos \theta
$$

#梯度
>>是一个向量

$$
grad f(x_0,y_0)=(f_x(x_0,y_0),f_y(x_0,y_0))
$$

#方向导数和梯度
$$
\frac{\partial f}{\partial l}=gradf(x_0,y_0)· \overrightarrow{l}
$$
ss