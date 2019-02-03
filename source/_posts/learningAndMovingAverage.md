---
title: 学习率和滑动平均模型
date: 2019-02-03 19:55:20
category: 深度学习
tags: [tensorflow]
---

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
