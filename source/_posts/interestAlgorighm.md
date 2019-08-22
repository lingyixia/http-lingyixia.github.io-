---
title: 总有一个算法让你惊艳
date: 2019-07-06 13:34:05
category: 算法
tags:
---
>>总有一个算法让你惊艳

1. 随机洗牌
问:一副牌54张，如何洗牌才能让最公平。
什么叫公平?就是每张牌在每个位置的概率都一样就是公平, Knuth 老爷子给出来这样的算法:
    1. 初始化任意顺序
    2. 从最后一张牌开始,设为第K张,然后从[1,K]张中任选一张与其交换
    3. 从[1,K-1]张牌中任选一张和第K-1张交换... ...
伪代码:
```
for(int i =n;i>=1;i--)
{
    swap(array[i],random(1,i))
}
```
是不是贼简单？那为啥这个算法能做到呢？下面证明一下:
假设现在有5张牌，初始顺序为$1,2,3,4,5$
首先，从1~5(这是下标)中任选一张比如选到了2,那么用2和5交换得到$1,5,3,4,2$,也就是说，第一次交换2在最后一个位置的概率是1/5(也可以说任意一个数字在最后一个位置的概率是1/5),那么我们进行第二次交换,从1~4(这是下标)中任选一个和第4个交换,比如我们选到了1,在此之前要保证1没有在之前的步骤被选中也就是4/5,现在选中1的概率是1/4,那么两者相乘得到1/5,也就是1在第4个位置的概率是1/5也可以说任意一个数字在最后一个位置的概率是1/5),后面的不用多说了吧。
这题和[蓄水池采样算法](https://lingyixia.github.io/2019/04/14/PoolSampling/)由异曲同工之妙。
用这个算法还可以在中途随意停止,比如有54张牌,我们要找任意10张牌进行公平洗牌,那么只需要上诉步骤执行10次就可以了.

2.切分句子
在NLP中经常会有这样的需求,对于训练数据有少部分会特别长,远远超出平均长度,那么我们就需要对句子进行拆分,但是不能直接安长度切，这样很可能会切断关键词,切分方法一般是在无用的地方切分,比如标点符号,现在给出一个算法实现这个功能:
```
def data_cut(sentence, cut_chars,cut_length,min_length):
    if len(sentence) <= cut_length:
        return [sentence]
    else:
        for char in cut_chars:
            start = min_length#防止直接从头几个就找到了，这样切的太短
            end = len(sentence) - (min_length-1)#防止从最后几个找到了,这样切的也太短
            if char in sentence[start:end]:
                index = sentence[start:end].index(char)
                return data_cut(sentence[:start + index], cut_chars) + data_cut(sentence[start + index:], cut_chars)
    return [sentence]#如果没有找到切分点那就不管句子长度直接返回
```
>>
参数说明:sentence是一个list,内容是sentence每个字符
       cut_chars是一个list,内容是切分字符，按优先级排序
       cut_length是int类型，表示切分多长的句子,这个长度以及以下的直接返回
       min_length:切分后子句子的最短长度
return:二维list,注意，每个切分后的句子不一定全小于cut_length,有些另类的句子可能没有切分点,这样的需要手动处理
eg:
```
def data_cut(sentence, cut_chars, cut_length, min_length):
    if len(sentence) <= cut_length:
        return [sentence]
    else:
        for char in cut_chars:
            start = min_length
            end = len(sentence) - (min_length - 1)
            if char in sentence[start:end]:
                index = sentence[start:end].index(char)
                return data_cut(sentence[:start + index], cut_chars, cut_length, min_length) + data_cut(
                    sentence[start + index:], cut_chars, cut_length, min_length)
    return [sentence]


if __name__ == '__main__':
    sentence = "三种上下文特征：单词、n-gram 和字符在词嵌入文献中很常用。大多数词表征方法本质上利用了词-词的共现统计，即使用词作为上下文特征（词特征）。受语言建模问题的启发，开发者将 n-gram 特征引入了上下文中。词到词和词到 n-gram 的共现统计都被用于训练 n-gram 特征。对于中文而言，字符（即汉字）通常表达了很强的语义。为此，开发者考虑使用词-词和词-字符的共现统计来学习词向量。字符级的 n-gram 的长度范围是从 1 到 4（个字符特征）。"
    l = data_cut(list(sentence), ['，', '。'], 150, 5)
    print(l)
#自己运行看看吧，不好写
```