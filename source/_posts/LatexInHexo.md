---
layout: post
title: Hexo中Latex公式显示
date: 2019-01-14 11:05:00
category: 博客
tags: [Latex,Markdown,Hexo]
mathjax: true
description: Hexo中Latex公式显示问题解决
---
>Hexo中Latex公式显示问题解决

1. 卸载:
>>npm uninstall hexo-renderer-marked --save

2. 安装
>>npm install hexo-renderer-kramed --save

3. 解决冲突
`node_modules\kramed\lib\rules\inline.js`中:
[需要转义的字符太多了,直接看这里吧](https://www.jianshu.com/p/7ab21c7f0674)

4. 开关
每次写博客头部加上mathjax: true
eg:
>>\-\-\-
title: index.html
date: 2018-2-8 21:01:30
tags:
mathjax: true
\-\-\-