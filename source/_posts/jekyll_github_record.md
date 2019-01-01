---
layout: post
title: github博客小记
category: 博客
tags: [坑]
description: 简单记录搭建github page的坑
---

>简单记录自己github page搭建的过程，只是为了记录自己踩一些坑，不是新手教程

本博客使用了jekyll+github page,要使用到jekyll的主题，可以先把主题下载下来，本地预览调试，在传到github上托管。也可以直接fork到自己的github中，改名字，在下载下来。我是直接fork的[作者大大的博客](http://BladeMasterCoder.github.io)，然后改名字即可。 

1.：github如果不自定义域名，会强制使用https协议，这样，如果某个页面中引用了http协议的连接会出错（似乎只能用一样的协议），这时可以在header中加入：  

```
<meta http-equiv="Content-Security-Policy"content="upgrade-insecure-requests">
```
作用是强制http使用https。

**但是！！！！！！**
当本地预览的时候地址是一定要将该句注释掉，因为本地预览是http:/127.0.0.1:4000，不支持https。

2.jekyll本地预览首先需要安装[ruby+devkit](https://rubyinstaller.org/downloads/)，接下来：gem install jekyll（安装jekyll）gem install  bundler（安装bundler）
然后到博客目录下，新建Gemfile，里面输入：

```
source 'https://rubygems.org'
gem 'github-pages', group: :jekyll_plugins
```
然后：bundle install(需要上诉Gemfile)，最后bundle exec jekyll serve启动服务器即可

3.simple-jekyll-search:最让人头疼的是找不到json,基本步骤按照[这里](https://github.com/christian-fei/Simple-Jekyll-Search)即可以，
一定不要忘了那个
```
---
layout: null
---
```
也就是说其实这根本不是一个标准的json，我就是没加上面那部分说啥也不行,坑苦了我，本来还以为不是json的一部分。