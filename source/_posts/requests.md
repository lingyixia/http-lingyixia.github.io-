---
title: requests
date: 2019-01-17 15:03:43
cagetory: 爬虫
tags: [requests,python]
---
requests库的一些记录
<!--more-->

1.重试次数和超时,加headers
```
import requests, time
from requests.adapters import HTTPAdapter

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36'}
if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    sess = requests.Session()
    sess.mount('http://', HTTPAdapter(max_retries=5))
    sess.mount('https://', HTTPAdapter(max_retries=5))
    try:
        response = sess.get('http://www.google.com', timeout=3)
    except Exception as e:
        print(e)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))

```
