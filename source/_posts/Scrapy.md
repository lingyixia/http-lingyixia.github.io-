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
2. `RETRY_HTTP_CODES = [500, 502, 503, 504, 408]`,默认需要重复的码,可自行添减,比如`302`
3. `DOWNLOAD_DELAY`:下载间隔时间,默认为0
4. `RANDOMIZE_DOWNLOAD_DELAY=True`,为True时下载间隔时间为`0.5 * DOWNLOAD_DELAY and 1.5 * DOWNLOAD_DELAY`
5. `DEFAULT_REQUEST_HEADERS`中可以添加cookies

#middlewares
##DownloaderMiddleware
对于一个DownloaderMiddleware而言的生命周期函数而言,有两个主要函数需要处理
1. `process_request(self, request, spider)`
返回值有四种:
* None: 继续执行下一个`DownloaderMiddleware`(正常情况)
* Response对象: 不在调用其他`DownloaderMiddleware`的`process_request`函数,返回给`scrapy`引擎该Response对象
* Request对象: Scrapy则停止调用`process_request`方法并重新调度返回的request
* raise一个IgnoreRequest异常: 则安装的下载中间件的 process_exception() 方法会被调用
2. `process_response(self, request, response, spider)`
返回值有三种:
* Response对象:继续执行下一个`DownloaderMiddleware`(正常情况)
* Request对象:同上
* raise一个IgnoreRequest异常:调用request的errback(Request.errback)

##RetryMiddleware
scrapy默认:`from scrapy.downloadermiddlewares.retry import RetryMiddleware`