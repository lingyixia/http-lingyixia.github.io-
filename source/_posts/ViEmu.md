---
title: ViEmu免费30天破解
date: 2019-02-23 09:51:14
category: Tip
tags: [插件,破解]
---
ViEmu
<!--more-->

#目标
删除Viemu在本机上的时间记录文件

总共需要删除两个地方:
```
1. HKEY_CLASSES_ROOT\Wow6432Node\CLSID\{目录ID}的InprocServr32
2. C:\Users\用户名\AppData\Local\Identities\{ID项}
```
#方法
对于要删除的第二个很容易,直接找到删除即可,问题是第一个,该目录下的项有很多,现在需要找的是ViEmu的目录ID
##第一步
我们需要找到所ViEmu的VSHub.dll，目录ID记录在这个DLL文件里边
该文件应该在:
C:\Users\用户名\AppData\Local\Microsoft\VisualStudio\15.0_f45ae071\Extensions\h0npgwe4.q4b,也不一定,使用everything查一下即可
##第二部
1.使用Reflector（.net的反编译器，可以在网上下载）打开该DLL，找到VSHub命名空间下的Hub类，找到Initialize(RegistryKey)方法并点击进入，在对应的代码中，找到ViEmuProt.InitializeLicenseStuff(this.m_productData);这一句代码，如下图所示:
![](/img/Reflector.png)
2.点击进入ViEmuProt.InitializeLicenseStuff这个方法，找到其中的vep_WriteTrialPeriodControlItemsIfFirstTime(_productData)函数，如下图所示（这个函数就是写注册表的函数）
![](/img/LicenseStuff.png)
3.再次点击进入该函数，如下图:
![](/img/FirstTime.png)
红色框所示的函数即为写注册表的函数，可以看到，这个CreateSubKey(name)函数中对应的name参数就是我们需要的目录ID，那么这个ID是怎么来的呢？

可以看到，这个参数是通过函数的第一条语句得到的（图中蓝色框）
4.点击进入GenerateTrialControlRegKeyName(_productData)函数（上图蓝框），如下图所示:
![](/img/egKeyName.png)
VS插件对应的product是0，所以，目录ID就是最下边那个{B9CDA4C6-C44F-438B-B5E0-C1B39EA864C4}