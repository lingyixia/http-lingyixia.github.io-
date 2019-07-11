---
layout: post
title: mongo
date: 2019-01-17 09:39:16
category: 数据
tags: [mongo,教训]
---

我日你妈卖批,手一滑训练集没了,特么的将近两个月的量,汽车之家啊大爷的，，，一定要备份，，一定要备份，，，一定要备份!!!!!!!!!!!!!!
最后,祭奠老子第一次误删数据———2019-01-16日!!!!
<!--more-->

* mongo数据备份:`mongodump -h dbhost -d dbname -o dbdirectory`
* mongo数据恢复:`mongorestore -h <hostname><:port> -d dbname <path>`