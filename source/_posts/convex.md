---
title: convex
date: 2018-12-27 20:20:55
category: 算法
tags: [C++,leetcode]
description: 对应[leetcode 587](https://leetcode.com/problems/erect-the-fence/)
---

>>There are some trees, where each tree is represented by (x,y) coordinate in a two-dimensional garden. Your job is to fence the entire garden using the minimum length of rope as it is expensive. The garden is well fenced only if all the trees are enclosed. Your task is to help find the coordinates of trees which are exactly located on the fence perimeter.
# Graham扫描法
步骤:
1. 将所有的点排序,取$y$轴最小的一个点记为$O$(如果多个则取$x$小的)
2. 平移所有的点,使$O$为原点
3. 逆时针排序