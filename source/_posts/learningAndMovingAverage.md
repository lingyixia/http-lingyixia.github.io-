---
title: 学习率和滑动平均模型
date: 2019-02-03 19:55:20
category: 深度学习
tags: [tensorflow]
---

>[参考](https://www.cnblogs.com/wuliytTaotao/p/9479958.html)
滑动平均(exponential moving average),或者叫做指数加权平均(exponentially weighted moving average)，目的是用历史值和当前值的加权来代替当前值,这样可以使值的变化更加平滑.

# 用滑动平均估计局部均值
假设变量$v_t$表示变量$v$在时间$t$处的值,不使用滑动平均模型时的值为$\theta_t$:
$$
v_t = \beta v_{t-1}+(1-\beta)\theta_t \quad \beta \in[0,1] \tag{1} \\
$$
eg:
$$
\begin{align}
v_1 &= \theta_1 \\
v_2 &=\beta \theta_1+(1-\beta)\theta_2 \\
v_3 &= \beta v_2 + (1-\beta)\theta_3=\beta^2\theta_1+\beta(1-\beta)\theta_2+(1-\beta)\theta_3\\
v_4 &=\beta^3\theta_1+ \beta^2(1-\beta)\theta_2+\beta(1-\beta)\theta_3+(1-\beta)\theta_4
\end{align}
$$
即:
$$
v_t=(1-\beta)\theta_t+\beta(1-\beta)\theta_{t-1}+\beta^2(1-\beta)\theta_{t-2}+...+\beta^k(1-\beta)\theta_{t-k}+...+\beta^n\theta_1 \tag{2}
$$
上诉实例可以说明$v_t$是所有历史值的加权平均。
现在我们要证明:**当$\beta \rightarrow 1$时,公式(1)计算$v_t$约等于前$\frac{1}{1-\beta}$个时间点的加权.**
证明:  
我们假设$k=\frac{1}{1-\beta}$,当$\beta \rightarrow 1$时,$k \rightarrow +\infty$.只要证明$\beta^k$足够小即可.  
$$
\because \beta^{\frac{1}{1-\beta}}=(1-\frac{1}{k})^k \\
\therefore \lim_{k\rightarrow +\infty}(1-\frac{1}{k})^n=e^{-1}≈0.3679\\
\therefore \beta^k ≈ 0.3679
$$
我们认为$e^{-1}$足够小,即**当$\beta \rightarrow 1$时$k=\frac{1}{1-\beta}$之后的项都可以忽略,即可以认为公式(1)是前$\frac{1}{1-\beta}$项的加权和.**

公式(1)中,虽然可以得到上诉结论,但是当$t$比较小的时候,前面不足$\frac{1}{1-\beta}$项的时候用公式(1)得到的值和真实值差距较大,所以还要做一个调整:
$$
v_t = \frac{\beta v_{t-1}+(1-\beta)}{1-\beta^t} \qquad \beta \in[0,1] \tag{1} \\
$$



>学习率和滑动平均模型是两种神经网络的优化方式

#学习率
>>用在神经网络**训练**阶段

函数为:`def exponential_decay(learning_rate,global_step,decay_steps,decay_rate,staircase=False,name=None)`
计算公式为:
`decayed_learning_rate=learning_rate * decay_rate^(global_step/decay_steps)`
其实也就是`global_step`每过过一轮`decay_steps`,学习率就将为原来的`decay_rate`倍.
当`staircase`为`False`的时候,`global_step`每加一学习率就变一次,即学习率成平滑状下降,当为`True`时候,只有`global_step/decay_rate`为整数的时候才变,即学习率程阶梯状下降.

#滑动平均模型
>>用在神经网络**验证或测试**阶段

eg:
```
import tensorflow as tf

def fun(session):
    for v in tf.global_variables():
        print(session.run(v))

if __name__ == '__main__':
    v = tf.Variable(initial_value=1.0, dtype=tf.float32,name='v')
    step = tf.Variable(initial_value=0, trainable=False,name='step')
    ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=step)
    maintain_averages_op = ema.apply([v])
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run([v, step,ema.average(v)]))
        sess.run(tf.assign(v, 5))
        sess.run(maintain_averages_op)
        print(sess.run([v,step, ema.average(v)]))

        sess.run(tf.assign(step, 10000))
        sess.run(tf.assign(v, 10))
        sess.run(maintain_averages_op)
        print(sess.run([v,step, ema.average(v)]))

        sess.run(maintain_averages_op)
        print(sess.run([v, step,ema.average(v)]))]
输出:
[1.0, 0, 1.0]
[5.0, 0, 4.6]
[10.0, 10000, 4.654]
[10.0, 10000, 4.70746]
```

>>其基本思路是对于一个已经训练好的模型,比如其中有一个参数是`v`,在验证或测试阶段并不是直接用这个参数去直接计算,而是保存一个影子变量`shadow_variable`,使用公式:`shadow_variable=decay*shadow_variable+(1-decay)*Variable`来充当计算时候使用的参数,即在真实变量和影子变量之间做一个平滑运算,一般更接近影子变量.刚开始的时候(还没运行`maintain_averages_op`)影子变量=原始变量,此后每运行一次`maintain_averages_op`便计算一次其控制的所有变量的影子变量.然后使用`ema.average()`函数拿出计算后得到的影子变量。
衰减率的计算公式为:
$$
min\left\{ decay,\frac{1+num\_updates}{10+num\_updates} \right\}
$$
