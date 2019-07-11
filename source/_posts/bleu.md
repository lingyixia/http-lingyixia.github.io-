---
title: Bleu
date: 2019-05-26 16:38:13
category: 自然语言处理
tags: [转载]
---
转载自这里: https://www.cnblogs.com/weiyinfu/p/9853769.html
<!--more-->

#理解bleu
bleu全称为Bilingual Evaluation Understudy(双语评估替换),是2002年提出的用于评估机器翻译效果的一种方法，这种方法简单朴素、短平快、易于理解。因为其效果还算说得过去，因此被广泛迁移到自然语言处理的各种评估任务中。这种方法可以说是：山上无老虎，猴子称大王。时无英雄遂使竖子成名。蜀中无大将，廖化做先锋。

#问题描述
首先，对bleu算法建立一个直观的印象。
有两类问题：
1. 给定一个句子和一个候选句子集，求bleu值，此问题称为sentence_bleu
2. 给定一堆句子和一堆候选句子集，求bleu值，此问题称为corpus_bleu

机器翻译得到的句子称为candidate，候选句子集称为references。
计算方式就是计算candidate和references的公共部分。公共部分越多，说明翻译结果越好。
##给定一个句子和一个候选句子集计算bleu值
bleu考虑1，2，3，4共4个n-gram，可以给每个n-gram指定权重。

对于n-gram：

* 对candidate和references分别分词（n-gram分词）
* 统计candidate和references中每个word的出现频次
对于candidate中的每个word，它的出现频次不能大于references中最大出现频次
* 这一步是为了整治形如the the the the the这样的candidate，因为the在candidate中出现次数太多了，导致分值为1。为了限制这种不正常的candidate，使用正常的references加以约束。
* candidate中每个word的出现频次之和除以总的word数，即为得分score
* score乘以句子长度惩罚因子即为最终的bleu分数
这一步是为了整治短句子，比如candidate只有一个词：the，并且the在references中出现过，这就导致得分为1。也就是说，有些人因为怕说错而保持沉默。

bleu的发展不是一蹴而就的，很多人为了修正bleu，不断发现bleu的漏洞并提出解决方案。从bleu的发展历程上，我们可以学到如何设计规则整治badcase。

最后,对于1-gram，2-gram，3-gram的组合，应该采用几何平均，也就是$s_1^{w_1} \times s_2^{w_2} \times s_3^{w_3}$，而不是算术平均$w_1 \times s_1+w_2 \times s_2+w_3 \times s_3$。
```
from collections import Counter

import numpy as np
from nltk.translate import bleu_score


def bp(references, candidate):
    # brevity penality,句子长度惩罚因子
    ind = np.argmin([abs(len(i) - len(candidate)) for i in references])
    if len(references[ind]) < len(candidate):
        return 1
    scale = 1 - (len(candidate) / len(references[ind]))
    return np.e ** scale


def parse_ngram(sentence, gram):
    # 把一个句子分成n-gram
    return [sentence[i:i + gram] for i in range(len(sentence) - gram + 1)]  # 此处一定要注意+1，否则会少一个gram


def sentence_bleu(references, candidate, weight):
    bp_value = bp(references, candidate)
    s = 1
    for gram, wei in enumerate(weight):
        gram = gram + 1
        # 拆分n-gram
        ref = [parse_ngram(i, gram) for i in references]
        can = parse_ngram(candidate, gram)
        # 统计n-gram出现次数
        ref_counter = [Counter(i) for i in ref]
        can_counter = Counter(can)
        # 统计每个词在references中的出现次数
        appear = sum(min(cnt, max(i.get(word, 0) for i in ref_counter)) for word, cnt in can_counter.items())
        score = appear / len(can)
        # 每个score的权值不一样
        s *= score ** wei
    s *= bp_value  # 最后的分数需要乘以惩罚因子
    return s


references = [
    "the dog jumps high",
    "the cat runs fast",
    "dog and cats are good friends"
]
candidate = "the d o g  jump s hig"
weights = [0.25, 0.25, 0.25, 0.25]
print(sentence_bleu(references, candidate, weights))
print(bleu_score.sentence_bleu(references, candidate, weights))
```

##给定一组句子和一个组候选句子集计算bleu值
>>一个corpus是由多个sentence组成的，计算corpus_bleu并非求sentence_bleu的均值，而是一种略微复杂的计算方式，可以说是没什么道理的狂想曲。一个文档包含3个句子，句子的分值分别为a1/b1，a2/b2，a3/b3。
那么全部句子的分值为：(a1+a2+a3)/(b1+b2+b3)
惩罚因子也是一样：三个句子的长度分别为l1,l2,l3，对应的最接近的reference分别为k1,k2,k3。那么相当于bp(l1+l2+l3,k1+k2+k3)。
也就是说：对于corpus_bleu不是单纯地对sentence_bleu求均值，而是基于更统一的一种方法。

```
from collections import Counter

import numpy as np
from nltk.translate import bleu_score


def bp(references_len, candidate_len):
    if references_len < candidate_len:
        return 1
    scale = 1 - (candidate_len / references_len)
    return np.e ** scale


def parse_ngram(sentence, gram):
    return [sentence[i:i + gram] for i in range(len(sentence) - gram + 1)]


def corpus_bleu(references_list, candidate_list, weights):
    candidate_len = sum(len(i) for i in candidate_list)
    reference_len = 0
    for candidate, references in zip(candidate_list, references_list):
        ind = np.argmin([abs(len(i) - len(candidate)) for i in references])
        reference_len += len(references[ind])
    s = 1
    for index, wei in enumerate(weights):
        up = 0  # 分子
        down = 0  # 分母
        gram = index + 1
        for candidate, references in zip(candidate_list, references_list):
            # 拆分n-gram
            ref = [parse_ngram(i, gram) for i in references]
            can = parse_ngram(candidate, gram)
            # 统计n-gram出现次数
            ref_counter = [Counter(i) for i in ref]
            can_counter = Counter(can)
            # 统计每个词在references中的出现次数
            appear = sum(min(cnt, max(i.get(word, 0) for i in ref_counter)) for word, cnt in can_counter.items())
            up += appear 
            down += len(can) 
        s *= (up / down) ** wei
    return bp(reference_len, candidate_len) * s


references = [
    [
        "the dog jumps high",
        "the cat runs fast",
        "dog and cats are good friends"],
    [
        "ba ga ya",
        "lu ha a df",
    ]
]
candidate = ["the d o g  jump s hig", 'it is too bad']
weights = [0.25, 0.25, 0.25, 0.25]
print(corpus_bleu(references, candidate, weights))
print(bleu_score.corpus_bleu(references, candidate, weights))
```
##简化代码
```
from collections import Counter

import numpy as np
from nltk.translate import bleu_score


def bp(references_len, candidate_len):
    return np.e ** (1 - (candidate_len / references_len)) if references_len > candidate_len else 1


def nearest_len(references, candidate):
    return len(references[np.argmin([abs(len(i) - len(candidate)) for i in references])])


def parse_ngram(sentence, gram):
    return [sentence[i:i + gram] for i in range(len(sentence) - gram + 1)]


def appear_count(references, candidate, gram):
    ref = [parse_ngram(i, gram) for i in references]
    can = parse_ngram(candidate, gram)
    # 统计n-gram出现次数
    ref_counter = [Counter(i) for i in ref]
    can_counter = Counter(can)
    # 统计每个词在references中的出现次数
    appear = sum(min(cnt, max(i.get(word, 0) for i in ref_counter)) for word, cnt in can_counter.items())
    return appear, len(can)


def corpus_bleu(references_list, candidate_list, weights):
    candidate_len = sum(len(i) for i in candidate_list)
    reference_len = sum(nearest_len(references, candidate) for candidate, references in zip(candidate_list, references_list))
    bp_value = bp(reference_len, candidate_len)
    s = 1
    for index, wei in enumerate(weights):
        up = 0  # 分子
        down = 0  # 分母
        gram = index + 1
        for candidate, references in zip(candidate_list, references_list):
            appear, total = appear_count(references, candidate, gram)
            up += appear 
            down += total 
        s *= (up / down) ** wei
    return bp_value * s


def sentence_bleu(references, candidate, weight):
    bp_value = bp(nearest_len(references, candidate), len(candidate))
    s = 1
    for gram, wei in enumerate(weight):
        gram = gram + 1
        appear, total = appear_count(references, candidate, gram)
        score = appear / total
        # 每个score的权值不一样
        s *= score ** wei
    # 最后的分数需要乘以惩罚因子
    return s * bp_value


if __name__ == '__main__':
    references = [
        [
            "the dog jumps high",
            "the cat runs fast",
            "dog and cats are good friends"],
        [
            "ba ga ya",
            "lu ha a df",
        ]
    ]
    candidate = ["the d o g  jump s hig", 'it is too bad']
    weights = [0.25, 0.25, 0.25, 0.25]
    print(corpus_bleu(references, candidate, weights))
    print(bleu_score.corpus_bleu(references, candidate, weights))
    print(sentence_bleu(references[0], candidate[0], weights))
    print(bleu_score.sentence_bleu(references[0], candidate[0], weights))
```