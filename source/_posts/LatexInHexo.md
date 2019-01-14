---
layout: post
title: Hexo中Latex公式显示
date: 2019-01-14 11:05:00
tags: [Latex,Markdown,Hexo]
mathjax: true
description: Hexo中Latex公式显示问题解决
---
>Hexo中Latex公式显示问题解决

1.Hexo中Latex公式显示问题解决 卸载:
>>npm install hexo-math --save

2. 安装
>>npm install hexo-math --save
npm install hexo-renderer-kramed --save

3. 解决冲突
`node_modules\kramed\lib\rules\inline.js中`
>>escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/
改为
escape: /^\\([`*\[\]()#$+\-.!_>])/
em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/ 
更改为 
em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/

4. 开关
每次写博客头部加上mathjax: true
eg:
>>\-\-\-
title: index.html
date: 2018-2-8 21:01:30
tags:
mathjax: true
\-\-\-