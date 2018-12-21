---
layout: post
title: 数据科学常用之Pandas
category: 数据科学
tags: [pandas,python]
description: 数据科学常用Pandas函数
---

>包含数据科学常用Pandas函数

1.pd.set_option   
[set_option文档地址]  (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.set_option.html?highlight=set_option#pandas.set_option)  
```
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
```

