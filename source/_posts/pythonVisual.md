---
title: Python可视化
date: 2019-06-16 16:07:28
category: 数据科学
tags: 
---

常用数据可视化方式[参考](https://blog.csdn.net/weixin_39739342/article/details/80008469)
<!--more-->
#折线图 Line Chart
>>用于观察数据和比较走势，比如横坐标时月份，纵坐标是销售额

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    A = [0, 10, 20, 10, 0, 10, 20, 30, 5, 0, 10, 20, 10, 0, 20]
    B = [0, 1.0, 2.9, 3.61, 3.249, 3.9241, 5.5317, 7.9785, 7.6807, 6.9126, 7.2213, 8.4992, 8.6493, 7.7844, 8.0059]
    C = [0, 10.0, 15.2632, 13.321, 9.4475, 9.5824, 11.8057, 15.2932, 13.4859, 11.2844, 11.0872, 12.3861, 12.0536,
         10.4374, 10.38077]
    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    # data.plot.line()
    sns.lineplot(data=data)
    plt.show()
```

>>注释部分是使用seaborn绘图得到

![](\img\pythonVisual\line.png)

#柱状图(条形图) Bar Chart
>>柱状图(条形图)表示数据的大小.和Line Chart其实可以是一样的数据,不过Bar Chart重在观察和比较数据之间的差距

```
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    A = [0, 10, 20, 10, 0, 10, 20, 30, 5, 0, 10, 20, 10, 0, 20]
    B = [0, 1.0, 2.9, 3.61, 3.249, 3.9241, 5.5317, 7.9785, 7.6807, 6.9126, 7.2213, 8.4992, 8.6493, 7.7844, 8.0059]
    C = [0, 10.0, 15.2632, 13.321, 9.4475, 9.5824, 11.8057, 15.2932, 13.4859, 11.2844, 11.0872, 12.3861, 12.0536,
         10.4374, 10.3807]
    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    data.plot.bar()
    plt.show()
```

![](\img\pythonVisual\bar.png)

>>`data.plot.bar()`横向柱状图,`bar(stacked=True)`堆叠柱状图

#直方图 Histgram
>>直方图表示数据的多少.和Bar Chart的区别是Histgram统计的是数据数量的多少,它的纵坐标和数据的大小无关,只和数据的多少有关.通常横坐标表示某个数据,纵坐标表示该数据**数据数量**。

```
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    A = [0, 10, 20, 10, 0, 10, 20, 30, 5, 0, 10, 20, 10, 0, 20]
    B = [0, 1.0, 2.9, 3.61, 3.249, 3.9241, 5.5317, 7.9785, 7.6807, 6.9126, 7.2213, 8.4992, 8.6493, 7.7844, 8.0059]
    C = [0, 10.0, 15.2632, 13.321, 9.4475, 9.5824, 11.8057, 15.2932, 13.4859, 11.2844, 11.0872, 12.3861, 12.0536,
         10.4374, 10.38077]
    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    data.hist()
    plt.show()

```

![](\img\pythonVisual\hist.png)

>>[其他图](https://www.cnblogs.com/fengff/p/8302315.html)

未完待续... ...

