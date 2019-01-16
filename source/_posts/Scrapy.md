---
title: Scrapy
date: 2019-01-16 10:36:54
category: 爬虫
tags: [scrapy]
---

>scrapy爬虫小记

#setting文件常用配置
默认seeting文件位置:`Lib\site-packages\scrapy\settings\default_settings.py`
1. `ROBOTSTXT_OBEY = False`:默认为True,一般都需要改为False,谁会无聊到爬虫还遵循网站协议？
2. `RETRY_HTTP_CODES = [500, 502, 503, 504, 408]`,默认需要重复的码,可自行添减
3. `DOWNLOAD_DELAY`:下载间隔时间,默认为0
4. `RANDOMIZE_DOWNLOAD_DELAY=True`,为True时下载间隔时间为`0.5 * DOWNLOAD_DELAY and 1.5 * DOWNLOAD_DELAY`
5. `DEFAULT_REQUEST_HEADERS`中可以添加cookies