---
title: 滑动平均模型
date: 2019-02-03 19:55:20
category: 深度学习
tags: [tensorflow]
---

>[参考](https://www.cnblogs.com/wuliytTaotao/p/9479958.html)
滑动平均(exponential moving average),或者叫做指数加权平均(exponentially weighted moving average)，目的是用历史值和当前值的加权来代替当前值,这样可以使值的变化更加平滑.

# 用滑动平均估计局部均值
>>只用在验证或测试阶段，训练时并不是使用

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
比如$\beta = 0.9$,约等于前10项的加权平均
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
v_t = \frac{\beta v_{t-1}+(1-\beta)}{1-\beta^t} \qquad \beta \in[0,1] \tag{2} \\
$$
当$t$较小的时候分母的调整作用较大,当$t$大到一定程度的时候分母接近1,调整作用越来越小。

![](\img\learningAndMovingAverage\lines.png)

#tensorflow中的滑动平均

eg:
```
# coding:utf-8

#-------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         Temp.py
# @Project       TensorflowLearn
# @Product       PyCharm
# @DateTime:     2019-06-16 15:59
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:
#-------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

BATCH_SIZE = 100
INPUT_NODE = 784
OUTPUT_NODE = 10

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001

TRAINING_STEPS = 5000

MOVING_AVERAGE_DECAY = 0.9999


def inference(input_tensor, ema, weights, biases):
    # 不使用滑动平均类
    if ema == None:
        output = tf.nn.softmax(tf.nn.xw_plus_b(input_tensor, weights, biases))
        return output
    else:
        # 使用滑动平均类
        output = tf.nn.softmax(tf.nn.xw_plus_b(input_tensor, ema.average(weights), ema.average(biases)))
        return output


def train(mnist, ifMovingAverage, ifDecayLearnRate):
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='input_x')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='input_y')

    weights = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, OUTPUT_NODE], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights=weights, biases=biases)

    global_step = tf.Variable(0, trainable=False)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    average_y = inference(x, ema, weights, biases)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights) + regularizer(weights)

    loss = cross_entropy_mean + regularization
    learning_rate = LEARNING_RATE_BASE
    if ifDecayLearnRate:
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True
        )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    if ifMovingAverage:
        correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print(f'After {i} training steps, validation accuracy is {validate_acc}')

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(f'after {TRAINING_STEPS} steps, test accuracy is {test_acc}')


if __name__ == '__main__':
    train(mnist, ifMovingAverage=True, ifDecayLearnRate=False)
```

>>其基本思路是对于一个已经训练好的模型,比如其中有一个参数是`v`,在验证或测试阶段并不是直接用这个参数去直接计算,而是保存一个影子变量`shadow_variable`,使用公式:`shadow_variable=decay*shadow_variable+(1-decay)*Variable`来充当计算时候使用的参数,即在真实变量和影子变量之间做一个平滑运算,一般更接近影子变量.刚开始的时候(还没运行`maintain_averages_op`)影子变量=原始变量,此后每运行一次`maintain_averages_op`便计算一次其控制的所有变量的影子变量.然后使用`ema.average()`函数拿出计算后得到的影子变量。
衰减率的计算公式为:
$$
min\left\{ decay,\frac{1+num\_updates}{10+num\_updates} \right\}
$$
