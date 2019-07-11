---
layout: post
title: Tampermonkey
date: 2019-01-14 18:25:14
category: 前端
tags: [javascript,jquery]
mathjax: true
description: Tampermonkey编程小记
---
>Tampermonkey编程小记

eg:(针对知音漫客付费漫画)
```
// ==UserScript==
// @name         知音漫客免费
// @version      0.1
// @description  知音漫客付费漫画免费看
// @author       陈飞宇
// @match        https://www.zymk.cn/*
// @require      http://code.jquery.com/jquery-1.11.0.min.js
// ==/UserScript==
$(document).ready(function(){
    $(document.body).css({ "overflow-y": "auto" });
        $("#layui-layer1").remove();
        $("div.layui-layer-shade").remove();
});
```
1. 文件头部必须包裹
```
// ==UserScript==

// ==/UserScript==
```
2. 要使用Jquery只需加入
```
// @require      http://code.jquery.com/jquery-1.11.0.min.js
```
