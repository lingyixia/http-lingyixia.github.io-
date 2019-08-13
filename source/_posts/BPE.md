---
title: BPE 和 WordPiece
date: 2019-08-10 17:37:51
category: NLP
tags: [算法]
---
>> WordPiece 是从 BPE(byte pair encoder) 发展而来的一种处理词的技术，目的是解决 OOV 问题,以翻译模型为例,原理是抽取公共二元串(bigram),首先看下BPE(Transformer的官方代码也是使用的这种方式):

# BPE
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
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
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

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
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
        word = word[:-1] + (word[-1].replace('</w>',''),)

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


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def encode(orig):
    word = tuple(orig) + ('<\/w>',)
    print("__word split into characters:__ <tt>{}</tt>".format(word))
    pairs = get_pairs(word)
    if not pairs:
        return orig

    iteration = 0
    while True:
        iteration += 1
        print("__Iteration {}:__".format(iteration))
        print("bigrams in the word: {}".format(pairs))
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
    if word[-1] == '<\/w>':
        word = word[:-1]
    elif word[-1].endswith('<\/w>'):
        word = word[:-1] + (word[-1].replace('<\/w>', ''),)

    return word


if __name__ == '__main__':
    train_data = {'l o w<\/w>': 5,
                  'l o w e r<\/w>': 2,
                  'n e w e s t<\/w>': 6,
                  'w i d e s<\/w>': 3,
                  'f o l l o w<\/w>': 1}
    bpe_codes = {}
    bpe_codes_reverse = {}
    num_merges = 1000
    for i in range(num_merges):
        print("Iteration {}".format(i + 1))
        pairs = get_stats(train_data)
        # print(pairs)
        best = max(pairs, key=pairs.get)
        if pairs[best] < 2:
            break
        train_data = merge_vocab(best, train_data)
        bpe_codes[best] = i
        bpe_codes_reverse[best[0] + best[1]] = best
        print("new merge: {}".format(best))
        print("train data: {}".format(train_data))
    # print()
    # print(bpe_codes)
```
输出结果:
>>Iteration 1
new merge: ('l', 'o')
train data: {'lo w<\/w>': 5, 'lo w e r<\/w>': 2, 'n e w e s t<\/w>': 6, 'w i d e s<\/w>': 3, 'f o l lo w<\/w>': 1}
Iteration 2
new merge: ('w', 'e')
train data: {'lo w<\/w>': 5, 'lo we r<\/w>': 2, 'n e we s t<\/w>': 6, 'w i d e s<\/w>': 3, 'f o l lo w<\/w>': 1}
Iteration 3
new merge: ('lo', 'w<\/w>')
train data: {'low<\/w>': 5, 'lo we r<\/w>': 2, 'n e we s t<\/w>': 6, 'w i d e s<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 4
new merge: ('n', 'e')
train data: {'low<\/w>': 5, 'lo we r<\/w>': 2, 'ne we s t<\/w>': 6, 'w i d e s<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 5
new merge: ('ne', 'we')
train data: {'low<\/w>': 5, 'lo we r<\/w>': 2, 'newe s t<\/w>': 6, 'w i d e s<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 6
new merge: ('newe', 's')
train data: {'low<\/w>': 5, 'lo we r<\/w>': 2, 'newes t<\/w>': 6, 'w i d e s<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 7
new merge: ('newes', 't<\/w>')
train data: {'low<\/w>': 5, 'lo we r<\/w>': 2, 'newest<\/w>': 6, 'w i d e s<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 8
new merge: ('w', 'i')
train data: {'low<\/w>': 5, 'lo we r<\/w>': 2, 'newest<\/w>': 6, 'wi d e s<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 9
new merge: ('wi', 'd')
train data: {'low<\/w>': 5, 'lo we r<\/w>': 2, 'newest<\/w>': 6, 'wid e s<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 10
new merge: ('wid', 'e')
train data: {'low<\/w>': 5, 'lo we r<\/w>': 2, 'newest<\/w>': 6, 'wide s<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 11
new merge: ('wide', 's<\/w>')
train data: {'low<\/w>': 5, 'lo we r<\/w>': 2, 'newest<\/w>': 6, 'wides<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 12
new merge: ('lo', 'we')
train data: {'low<\/w>': 5, 'lowe r<\/w>': 2, 'newest<\/w>': 6, 'wides<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 13
new merge: ('lowe', 'r<\/w>')
train data: {'low<\/w>': 5, 'lower<\/w>': 2, 'newest<\/w>': 6, 'wides<\/w>': 3, 'f o l low<\/w>': 1}
Iteration 14

>>可以看到，首先输入是词典{单词:词频}的形式,在每一个轮次都会寻找一个最大的子串，上诉第一次频率最大的子串就是('e', 's'),然后把字典中所有的('e', 's')合并就得到了{'l o w <\/w>': 5, 'l o w e r <\/w>': 2, 'n e w es t <\/w>': 6, 'w i d es <\/w>': 3, 'f o l l o w <\/w>': 1},后面以此类推,直到最大的词频小于某个阈值为止，上面设置的是2，最终得到的词表是:train data: {'low<\/w>': 5, 'lower<\/w>': 2, 'newest<\/w>': 6, 'wides<\/w>': 3, 'f o l low<\/w>': 1},那么最终会生成两个文件:

>>1.code.file
version: 0.2
w e
l o
we s
wes t<\/w>
n e
ne west<\/w>
lo w<\/w>
w i
wi d
wid e
wide s<\/w>
we r<\/w>
lo wer<\/w>
2.voc.txt(在生成code.file后根据code.file生成)
low 6
newest 6
wides 3
lower 2
f@@ 1
o@@ 1
l@@ 1

这就是处理原语料的过程，在训练的时候，首先用上诉的`encode`代码把训练数据根据`code.file`映射到'voc.txt'中的词，然后进行训练(label方面的处理方式是独立的，也不一定需要BPE处理)

# WordPiece
>WordPiece是Bert使用的处理方式,这个过两天在写吧，有点事。。。

#参考:
1. https://github.com/wszlong/sb-nmt
2. https://blog.csdn.net/u013453936/article/details/80878412
3. http://ufal.mff.cuni.cz/~helcl/courses/npfl116/ipython/byte_pair_encoding.html