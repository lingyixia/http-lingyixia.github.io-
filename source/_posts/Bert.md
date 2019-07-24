---
title: Bert
date: 2019-07-22 21:55:40
category: NLP
tags: [论文]
---
>本文不讲Bert原理,拟进行Bert源码分析和应用

# Bert源码分析

## 组织结构
>> [这是Bert Github地址](https://github.com/google-research/bert),打开后会看到这样的结构:
![](img/Bert/bert.jpeg)
下面我将逐个分析上诉图片中加框的文件,其他的文件不是源码,不用分析.

## modeling.py
>>该文件是整个Bert模型源码,包含两个类:
* BertConfig:Bert配置类
* BertModel:Bert模型类
* embedding_lookup:用来返回词向量函数

### BertConfig

```
class BertConfig(object):
    def __init__(self,
                 vocab_size,#词表大小
                 hidden_size=768,#即是词向量维度又是Transform的隐藏层维度
                 num_hidden_layers=12,#Transformer encoder中的隐藏层数,普通Transform中是6个
                 num_attention_heads=12,multi-head attention 的head的数量,普通Transform中是8个
                 intermediate_size=3072,encoder的“中间”隐层神经元数,普通Transform中是一个feed-forward
                 hidden_act="gelu",#隐藏层激活函数
                 hidden_dropout_prob=0.1,#隐层dropout率
                 attention_probs_dropout_prob=0.1,#注意力部分的dropout
                 max_position_embeddings=512,#最大位置编码长度,也就是序列的最大长度
                 type_vocab_size=16,#token_type_ids的大小,所谓的token_type_ids在Bert中是0或1，也就是上句标记为0，下句标记为1，鬼知道默认为16是啥意思。。。
                 initializer_range=0.02):随机初始化的参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
```

### BertModel
>>现在进入正题,开始分析Bert模型源码

```
class BertModel(object):

    def __init__(self, config, is_training, input_ids, input_mask=None, token_type_ids=None,use_one_hot_embeddings=False, scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)#调用embedding_lookup得到初始词向量
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))
    def get_pooled_output(self):
        return self.pooled_output
    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table
```
>>参数说明:
* `config`:一个`BertConfig`实例
* `is_training`:`bool`类型,是否是训练流程,用类控制是否dropout
* `input_ids`:输入`Tensor`, `shape`是`[batch_size, seq_length]`.
* `input_mask`:`shape`是`[batch_size, seq_length]`无需细讲
* `token_type_ids`:`shape`是`[batch_size, seq_length]`,`bert`中就是0或1
* `use_one_hot_embeddings`:在`embedding_lookup`返回词向量的时候使用,详细见`embedding_lookup`函数

### embedding_lookup
>>为了得到进入模型的词向量(token embedding)

```
def embedding_lookup(input_ids,vocab_size,embedding_size=128,initializer_range=0.02,word_embedding_name="word_embeddings",use_one_hot_embeddings=False):
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])
  embedding_table = tf.get_variable(name=word_embedding_name,shape=[vocab_size, embedding_size],initializer=create_initializer(initializer_range))
  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.gather(embedding_table, flat_input_ids)
  input_shape = get_shape_list(input_ids)
  output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)
```
>>参数说明：
* input_ids：[batch_size, seq_length]
* vocab_size:词典大小
* initializer_range：初始化参数
* word_embedding_name:不解释
* use_one_hot_embeddings:是否使用one_hot方式初始化(为啥我感觉这里是True还是False结果得到的结果是一样的？？？？？)如下代码.

return:token embedding:[batch_size, seq_length, embedding_size].和embedding_table(不解释)

```
import tensorflow as tf

tf.enable_eager_execution()
flat_input_ids = [2, 4, 5]
embedding_table = tf.constant(value=[[1, 2, 3, 4],
                                     [5, 6, 7, 8],
                                     [9, 1, 2, 3],
                                     [5, 6, 7, 8],
                                     [6, 4, 78, 9],
                                     [6, 8, 9, 3]],dtype=tf.float32)
one_hot_input_ids = tf.one_hot(flat_input_ids, depth=6)
output = tf.matmul(one_hot_input_ids, embedding_table)
print(output)
print(100*'*')
output = tf.gather(embedding_table, flat_input_ids)
print(output)
```

### embedding_postprocessor
>>bert模型的输入向量有三个,embedding_lookup得到的是token embedding 我们还需要segment embedding和position embedding,这三者的维度是完全相同的(废话不相同怎么加啊。。。)本部分代码会将这三个embeddig加起来并dropout

```
def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]
    output = input_tensor
    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if""`use_token_type` is True.")
        token_type_table = tf.get_variable(name=token_type_embedding_name,
                                           shape=[token_type_vocab_size, width],
                                           initializer=create_initializer(initializer_range))
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
        output += token_type_embeddings
    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(name=position_embedding_name,
                                                       shape=[max_position_embeddings, width],
                                                       initializer=create_initializer(initializer_range))
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])
            num_dims = len(output.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings
    output = layer_norm_and_dropout(output, dropout_prob)
    return output
```
>>参数说明:
* input_tensor:token embedding[batch_size, seq_length, embedding_size]
* use_token_type是否使用segment embedding
* token_type_ids:[batch_size, seq_length],这两个参数其实就是控制生成segment embedding的,上诉代码中的` output += token_type_embeddings`就是得到token embedding+segment embedding
* use_position_embeddings:是否使用位置信息
* max_position_embeddings:序列最大长度
注:
* 本部分代码中的`width`其实就是词向量维度(换个`embedding_size`能死啊。。。)
* 可以看出位置信息跟Transform的固定方式不一样，它是训练出来的.
* ` output += position_embeddings`就得到了三者的想加结果
return :token embedding+segment embedding+position_embeddings

### create_attention_mask_from_input_mask
>>目的是将本来shape为[batch_size, seq_length]转为[batch_size, seq_length,seq_length]

```
def create_attention_mask_from_input_mask(from_tensor, to_mask):
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]
    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
    mask = broadcast_ones * to_mask
    return mask
```
>>参数说明:
* from_tensor:[batch_size, seq_length].
* to_mask:[batch_size, seq_length]
注:`Transform`中的`mask`和平常用的不太一样,这里的`mask`是为了在计算`attention`的时候"看不到不应该看到的内容",计算方式为该看到的`mask`为0，不该看到的`mask`为一个负的很大的数字,然后两者相加(平常使用`mask`是看到的为1，看不到的为0，然后两者做点乘)，这样在计算`softmax`的时候那些负数的`attention`会非常非常小,也就基本看不到了.