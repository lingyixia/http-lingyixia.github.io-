---
title: scrapyAndRedis
date: 2019-01-24 14:23:53
category: 爬虫
tags: [scrapy,python,redis]
---
本篇并不是要说`scrapy_redis`框架,而是要说明另外一个结构:我们有一个需要,需要一个生产者(产生url)和一个消费者(即该爬虫,消费url),消费者监听生产者,当产生url的时候需要将其接过来,并爬取该url的内容。咋一看很符合scrapy_reids结构,但生产者和消费者是约定通过`redis`发布/订阅的方式交互的,由于对scrapy_reids了解不深,没深想如何改造,便投机取巧的使用了这样一种方式.
<!--more-->
#问题
本来其实这样的代码就可以了:
```
def start_requests(self):
    while True:
        msg = self.redis_sub.parse_response()
        if msg[0] != b'message':
            continue
        data = json.loads(msg[2].decode('utf-8'))
        id = data['id']
        styleUrl = data['styleUrl']
        pageCount = data['pageCount']
        self.obi.public(json.dumps({'id': id, 'isSynchronized': 1}))
        yield SplashRequest(url=styleUrl, callback=self.specHome_parse,
                            args={'wait': 5, 'timeout': 60, 'images': 0},
                            meta={'pageCount': pageCount, 'id': id, 'dont_redirect': True})
```

>>即在`start_request(self)`函数中死循环监听`redis`,比如生产者那边产生了10个,` msg = self.redis_sub.parse_response()`这行监听的函数便在第11次循环的时候阻塞,而前面`yield`的10个`SplashRequest`会继续执行爬取任务(想的很好),但现实并不成功,当代码阻塞的时候发现原先的10个`SplashRequest`也不执行了.但是我我以前的经验中已知,下面这样的代码是可以顺利执行的.

```
def start_requests(self):
    i=0
    while True:
        yield SplashRequest(url=urls[i], callback=self.specHome_parse,
                            args={'wait': 5, 'timeout': 60, 'images': 0},
                            meta={'pageCount': 22, 'id': 1, 'dont_redirect': True})
            i++
```

>>当`urls`中的`url`很多,比如上百个的时候,该代码是可以执行的,也就是说`while True`是没问题的,但是为什么阻塞了原先的`SplashRequest`就不继续了呢?具体还不清楚,但是经验告诉我阻塞了不行,但是让程序动起来就有可能成功,于是改造如下:

#解决方法
```
def start_requests(self):
    while True:
        try:
            msg = self.redis_sub.parse_response(block=False, timeout=5)
            if msg[0] != b'message':
                continue
            data = json.loads(msg[2].decode('utf-8'))
            id = data['id']
            styleUrl = data['styleUrl']
            pageCount = data['pageCount']
            self.obi.public(json.dumps({'id': id, 'isSynchronized': 1}))
            yield SplashRequest(url=styleUrl, callback=self.specHome_parse,
                                args={'wait': 5, 'timeout': 60, 'images': 0},
                                meta={'pageCount': pageCount, 'id': id, 'dont_redirect': True})
        except Exception as e:
            yield SplashRequest()
            print(e)
```
>>即在生产者没有生产`url`的时候不断`yield`空的请求,这样让程序动起来,竟然真的可以了!!!