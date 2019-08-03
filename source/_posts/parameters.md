---
title: 请叫我调参工程师
date: 2019-07-11 19:50:42
category: 深度学习
tags:
---
>>这篇博客会一直更新.
对于初学者而言，兴趣往往都在模型构成上面,但是，只有真正成为调参工程师才能深刻体会到调参对于模型的重要性，虽然我也只能算入门级,但已经被坑了好几次了。。。。。说多了都是泪啊。
总的来说,训练神经网络模型的超参数一般可以分为两类:
* 训练参数:学习率,正则项系数,epoch数,batchsize
* 模型参数:模型层数,隐藏层参数

#调参技巧
1. 当你想实现一个模型的时候,一定要用最精简的结构去实现它,尽量不要在任何地方做其他tip,任何tip都要尽量在保证模型无误的条件下进行，否则你永远不知道你所谓的一个小tip对你的模型有多大影响(血的教训,当初在做一个文本生成模型的时候所有激活函数都用的relu,这个bug让我一顿好找。。。)

2. 保证每次随机种子不变,才能让实验更有效的对比
3.  
# 训练参数
## 学习率
作为一个调参工程师，私以为,学习率是**最重要**的参数,对于初学者来说，往往最容易忽视学习率的作用.举一个我最开始被坑的例子,当初是做了一个NER模型,写完之后各项指标增长速度特别特别慢,慢到令人发指(一小时从1%到2%),但问题是确实是不断增长，由于刚刚接触也不知道增长速度应该是怎样就一直等着，凉了一天，发现还是那个速度,等不及了开始检查模型问题,由于刚刚接触TF，对自己没把握就一直找一直找，几乎找了一个星期愣是没改出来.不得已，要不调调参数把,还好第一个改的就是学习率,其实原来是0.01我以为已经够小了,改成了0.001,刚一运行，200个step后precision直接到了20%,我他娘的就。。。。原来问题在这。所以以后我在设置学习率的时候都会从一个特别小的数开始，比如0.0001，看看指标的变化，在增大一点学习率比如0.001，再看看变化，确定模型没问题，然后在开始。当然，学习率的设置还有很多方式,比如模拟退火方式,Transform中的学习率代码就使用了这种:
```
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
```
也就是让学习率先快诉增大，当step达到warmup_steps后后在慢慢减小
bert中的学习率是这样设置的:
```
learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done
    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
```
它也是先让学习率快速增大，当step达到warmup_steps时，在安装多项式方式衰减 顺便提一下，`learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)`这行代码值得学习

##batchsize
1. batch 和 batch 之间差别太大,训练难以收敛,形成震荡
2. batchsize 增大会使梯度优化方向更准
3. 随着 batch_size 增大，处理相同数据量的速度越快。
4. 随着 batch_size 增大，达到相同精度所需要的 epoch 数量越来越多。
5. 由于上述两种因素的矛盾，batch_size 增大到某个时候，达到时间上的最优。
6. 增大 batchsize 能够有效的利用GPU并行能力
7. GPU对batchsize为2的整数次幂效果更好




#模型参数
首先需要知道的是，对于隐藏层和层数刚开始的设置要紧盯训练样本的量，要保证模型的参数量不高于样本量的一半，有权威称$\frac{1}{10}$最好.反正你不要写完模型后参数太大,你想想用一千条数据去训练好几百万的参数能学到点啥？下面给出个统计模型参数的tf代码:
```
 def __get_parametres(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        tf.logging.info("总参数量为:{total_parameters}".format(total_parameters=total_parameters))
```