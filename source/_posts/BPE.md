---
title: BPE 和 WordPiece
date: 2019-08-10 17:37:51
category: NLP
tags: [算法]
---
>> WordPiece 是从 BPE(byte pair encoder) 发展而来的一种处理词的技术，目的是解决 OOV 问题,以翻译模型为例,原理是抽取公共二元串(bigram),首先看下BPE(Transformer的官方代码也是使用的这种方式):

# BPE
##关键代码
```
import re, collections


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as a tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def encode(orig):
    """Encode word based on list of BPE merge operations, which are applied consecutively"""

    word = tuple(orig) + ('</w>',)
    print("__word split into characters:__ <tt>{}</tt>".format(word))

    pairs = get_pairs(word)

    if not pairs:
        return orig

    iteration = 0
    while True:
        iteration += 1
        print("__Iteration {}:__".format(iteration))
        print("bigrams in the word: {}".format(pairs))
        print(pairs)
        bigram = min(pairs, key=lambda pair: bpe_codes.get(pair, float('inf')))
        print("candidate for merging: {}".format(bigram))
        if bigram not in bpe_codes:
            print("__Candidate not in BPE merges, algorithm stops.__")
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        print("word after merging: {}".format(word))
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>', ''),)

    return word


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

if __name__ == '__main__':
    train_data = {'l o w' + re.escape('</w>'): 5,
                  'l o w e r' + re.escape('</w>'): 2,
                  'n e w e s t' + re.escape('</w>'): 6,
                  'w i d e s t' + re.escape('</w>'): 3}
    bpe_codes = {}
    bpe_codes_reverse = {}
    num_merges = 1000
    for i in range(num_merges):
        pairs = get_stats(train_data)
        if not pairs:
            break
        print("Iteration {}".format(i + 1))
        best = max(pairs, key=pairs.get)
        train_data = merge_vocab(best, train_data)
        bpe_codes[best] = i
        bpe_codes_reverse[best[0] + best[1]] = best
        print("new merge: {}".format(best))
        print("train data: {}".format(train_data))

```
输出结果:
>>Iteration 1
new merge: ('e', 's')
train data: {'l o w</w>': 5, 'l o w e r</w>': 2, 'n e w es t</w>': 6, 'w i d es t</w>': 3}
Iteration 2
new merge: ('es', 't</w>')
train data: {'l o w</w>': 5, 'l o w e r</w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
Iteration 3
new merge: ('l', 'o')
train data: {'lo w</w>': 5, 'lo w e r</w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
Iteration 4
new merge: ('n', 'e')
train data: {'lo w</w>': 5, 'lo w e r</w>': 2, 'ne w est</w>': 6, 'w i d est</w>': 3}
Iteration 5
new merge: ('ne', 'w')
train data: {'lo w</w>': 5, 'lo w e r</w>': 2, 'new est</w>': 6, 'w i d est</w>': 3}
Iteration 6
new merge: ('new', 'est</w>')
train data: {'lo w</w>': 5, 'lo w e r</w>': 2, 'newest</w>': 6, 'w i d est</w>': 3}
Iteration 7
new merge: ('lo', 'w</w>')
train data: {'low</w>': 5, 'lo w e r</w>': 2, 'newest</w>': 6, 'w i d est</w>': 3}
Iteration 8
new merge: ('w', 'i')
train data: {'low</w>': 5, 'lo w e r</w>': 2, 'newest</w>': 6, 'wi d est</w>': 3}
Iteration 9
new merge: ('wi', 'd')
train data: {'low</w>': 5, 'lo w e r</w>': 2, 'newest</w>': 6, 'wid est</w>': 3}
Iteration 10
new merge: ('wid', 'est</w>')
train data: {'low</w>': 5, 'lo w e r</w>': 2, 'newest</w>': 6, 'widest</w>': 3}
Iteration 11
new merge: ('lo', 'w')
train data: {'low</w>': 5, 'low e r</w>': 2, 'newest</w>': 6, 'widest</w>': 3}
Iteration 12
new merge: ('low', 'e')
train data: {'low</w>': 5, 'lowe r</w>': 2, 'newest</w>': 6, 'widest</w>': 3}
Iteration 13
new merge: ('lowe', 'r</w>')
train data: {'low</w>': 5, 'lower</w>': 2, 'newest</w>': 6, 'widest</w>': 3}

>>可以看到，首先输入是词典{单词:词频}的形式,在每一个轮次都会寻找一个最大的子串，上诉第一次频率最大的子串就是('e', 's'),然后把字典中所有的('e', 's')合并就得到了{'l o w <\/w>': 5, 'l o w e r <\/w>': 2, 'n e w es t <\/w>': 6, 'w i d es <\/w>': 3, 'f o l l o w <\/w>': 1},后面以此类推,直到最大的词频小于某个阈值为止，上面设置的是2，最终得到的词表是:train data: {'low<\/w>': 5, 'lower<\/w>': 2, 'newest<\/w>': 6, 'wides<\/w>': 3, 'f o l low<\/w>': 1}，

这就是处理原语料的过程，在训练的时候，首先用上诉的`encode`代码把训练数据根据`code.file`映射到'voc.txt'中的词，然后进行训练(label方面的处理方式是独立的，也不一定需要BPE处理)

##subword-nmt使用
数据准备类似：https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/tests/data/corpus.en
1. subword-nmt learn-bpe -s {num_operations} < {train_file} > {codes_file}
```
作用：生成分词器
eg: subword-nmt learn-bpe -s 30000 < corpus.en > codes_file
codes_file生成的就是接下来用到的分词器，其实就是一个词对组成的文件，其中每一行都是当时预料中词对中频率最高的一个。（上诉代码对应这部分）
```
2. subword-nmt apply-bpe -c {codes_file} < {test_file} > {out_file}
```
作用：用分词器处理预料
eg:subword-nmt apply-bpe -c codes_file < corpus.en > out_file
out_file中就是靠分词器生成的语料
这里的操作单元是对原始预料的各个单词，比如'cement'，分为'c e m e n t</w>'
1. 词对为:[<c,e>,<e,m>,<m,e>,<e,n>,<n,t</w>>],其中[<e,m>,<m,e>,<e,n>]在codes_file,并且<e,n>在codes_file排名靠前(语料中词频高),合并结果为:'c e m en t</w>'
2. 词对为:[<c,e>,<e,m>,<m,en>,<en,t</w>>],其中[<e,m>,<en,t</w>>]在codes_file,并且<en,t</w>>在codes_file中排名靠前，合并结果为:'c e m ent</w>'
.
.
.
最终合并结果为:'c ement</w>'此时只有一个词对<c,ement</w>>,并且不再codes_file中，因此合并停止，该词分为两个子词:c,ement,在预料中为:c@@ ement
```
3. subword-nmt get-vocab --train_file {train_file} --vocab_file {vocab_file}
```
作用：生成词典（训练模型要用）
eg: subword-nmt get-vocab --input out_file --output vocab_file
vocab_file就是预料对应的词典(把out_file 中的单词set一便即可)，即接下来用vocab_file作为词典，out_file作为语料训练模型即可
```
4. 模型训练完成后，在具体场景使用的时候，必定会有@（因为词典中有@，用来区分该单词是前缀还是独立单词），因此要对后缀是@的单词跟下一个单词合并。

# WordPiece
>WordPiece是Bert使用的处理方式,这个过两天在写吧，有点事。。。

#参考:
1. https://github.com/wszlong/sb-nmt
2. https://blog.csdn.net/u013453936/article/details/80878412
3. http://ufal.mff.cuni.cz/~helcl/courses/npfl116/ipython/byte_pair_encoding.html