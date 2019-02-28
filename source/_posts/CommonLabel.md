---
title: 序列标注常用方式
date: 2019-01-28 15:11:55
tags: [Ner]
---
#BIO
>>基本标注方式为:$(B-begin,I-inside,O-outside)$.

$BIO$标注:将每个元素标注为“B_X”、“I_X”或者“O”。其中,“B_X”表示此元素所在的片段属于$X$类型并且此元素在此片段的开头,“I_X”表示此元素所在的片段属于$X$类型并且此元素在此片段的中间位置,“O”表示不属于任何类型。
eg:
  比如,我们将X表示为名词短语(Noun Phrase, NP),则BIO的三个标记为:
  1. B-NP：名词短语的开头
  2. I-NP：名词短语的中间
  3. O:不是名词短语

#BIOES
>>基本标注方式为: $(B-begin,I-inside,O-outside,E-end,S-single)$