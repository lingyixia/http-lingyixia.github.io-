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
![](/img/Bert/bert.jpeg)
下面我将逐个分析上诉图片中加框的文件,其他的文件不是源码,不用分析.

## modeling.py
>>该文件是整个Bert模型源码,包含两个类:
* BertConfig:Bert配置类
* BertModel:Bert模型类
* embedding_lookup:用来返回函数token embedding词向量
* embedding_postprocessor:得到token embedding+segment embedding+position embedding
* create_attention_mask_from_input_mask得到mask,用来attention该attention的部分
* transformer_model和attention_layer:Transform的ender部分,也就是self-attention,不解释了，看太多遍了.

**注意上面的顺序,不是乱写的,是按照BertModel调用顺序组织的.**

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
>>目的是将本来shape为[batch_size, seq_length]转为[batch_size, seq_length,seq_length],为什么要这样的维度呢?因为.....算了麻烦不写了，去我的另一篇[Transform](等会在写)中看吧

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

### transformer_model
>>这一部分是`Transform`部分,但是只有`encoder`部分,从`BertModel`中的` with tf.variable_scope("encoder"):`这一部分也可以看出来

```
def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))
    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))
    prev_output = reshape_to_matrix(input_tensor)#这个不单独写了,就是将[batch_size, seq_length, embedding_size]的input 给reahpe为[batch_size*seq_length,embedding_size]
    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output
            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(from_tensor=layer_input,
                                                     to_tensor=layer_input,
                                                     attention_mask=attention_mask,
                                                     num_attention_heads=num_attention_heads,
                                                     size_per_head=attention_head_size,
                                                     attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                     initializer_range=initializer_range,
                                                     do_return_2d_tensor=True,
                                                     batch_size=batch_size,
                                                     from_seq_length=seq_length,
                                                     to_seq_length=seq_length)
                    attention_heads.append(attention_head)
                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    attention_output = tf.concat(attention_heads, axis=-1)
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(attention_output,
                                                       hidden_size,
                                                       kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)
            with tf.variable_scope("intermediate"):#feed-forword部分
                intermediate_output = tf.layers.dense(attention_output,
                                                      intermediate_size,
                                                      activation=intermediate_act_fn,
                                                      kernel_initializer=create_initializer(initializer_range))
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(intermediate_output,
                                               hidden_size,
                                               kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)
    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output
```
>>参数说明:
* input_tensor:token embedding+segment embedding+position embedding [batch_size, seq_length, embedding_size]
* attention_mask:[batch_size, seq_length,seq_length]
* hidden_size:不解释
* num_hidden_layers:多少个`ecncoder block`
* num_attention_heads:多少个`head`
* intermediate_size:`feed forward`隐藏层维度
* intermediate_act_fn:`feed forward`激活函数
其他的不解释了
return [batch_size, seq_length, hidden_size],

### attention_layer
>>其实就是`self-attention`,但是在计算的时候全都转换为了二维矩阵，按注释的意思是避免反复reshape,因为reshape在CPU/GPU上易于实现，但是在TPU上不易实现,这样可以加速训练.

```
def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
    if len(from_shape) != len(to_shape):
        raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")
    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")
    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)
    query_layer = tf.layers.dense(from_tensor_2d,
                                  num_attention_heads * size_per_head,
                                  activation=query_act,
                                  name="query",
                                  kernel_initializer=create_initializer(initializer_range))
    key_layer = tf.layers.dense(to_tensor_2d,
                                num_attention_heads * size_per_head,
                                activation=key_act,
                                name="key",
                                kernel_initializer=create_initializer(initializer_range))
    value_layer = tf.layers.dense(to_tensor_2d,
                                  num_attention_heads * size_per_head,
                                  activation=value_act,
                                  name="value",
                                  kernel_initializer=create_initializer(initializer_range))
    query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, from_seq_length, size_per_head)
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads, to_seq_length, size_per_head)
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))
    if attention_mask is not None:
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
        attention_scores += adder#这里就是使用mask来attention该attention的部分
    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
    value_layer = tf.reshape(value_layer, [batch_size, to_seq_length, num_attention_heads, size_per_head])
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    context_layer = tf.matmul(attention_probs, value_layer)
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    if do_return_2d_tensor:
        context_layer = tf.reshape(context_layer, [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        context_layer = tf.reshape(context_layer, [batch_size, from_seq_length, num_attention_heads * size_per_head])
    return context_layer
```
>>参数说明:
* from_tensor在Transform中被转为二维[batch_size*seq_length, embedding_size]
* to_shape:传过来的参数跟from_tensor一毛一样,在这里没什么卵用其实,因为q和k的length是一样的
* attention_mask:[batch_size, seq_length,seq_length]
* num_attention_heads:head数量
* size_per_head:每一个head维度,代码中是用总维度除以head数量得到的:attention_head_size = int(hidden_size / num_attention_heads)
return: return :[batch_size, from_seq_length,num_attention_heads * size_per_head].


### 总结一:
看完模型感觉真特么简单这模型,似乎除了self-attention就啥都没有了,但是先别着急,一般情况下模型是重点，但是对于Bert而言，模型却仅仅是开始，真正的创新点还在下面.

## create_pretraining_data.py
>>这部分代码用来生成训练样本,我们从`main`函数开始看起,首先进入`tokenization.py`

### def main
```
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))
    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)
    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(input_files,
                                          tokenizer,
                                          FLAGS.max_seq_length,
                                          FLAGS.dupe_factor,
                                          FLAGS.short_seq_prob,
                                          FLAGS.masked_lm_prob,
                                          FLAGS.max_predictions_per_seq,
                                          rng)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)
    write_instance_to_example_files(instances,
                                    tokenizer, 
                                    FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, 
                                    output_files)
```
###class TrainingInstance
>>单个训练样本类,看`__init__`就能看出来，没什么其他东西

```
class TrainingInstance(object):
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

```

### def create_training_instances
>>这个函数是重中之重，用来生成
```
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    all_documents = [[]]#外层是文档，内层是文档中的每个句子
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()
                if not line:# 空行表示文档分割
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)
    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(all_documents,
                                               document_index,
                                               max_seq_length,
                                               short_seq_prob,
                                               masked_lm_prob,
                                               max_predictions_per_seq,
                                               vocab_words,
                                               rng))
    rng.shuffle(instances)
    return instances
```
>>参数说明:
dupe_factor:每一个句子用几次:因为如果一个句子只用一次的话那么mask的位置就是固定的，这样我们把每个句子在训练中都多用几次,而且没次的mask位置都不相同,就可以防止某些词永远看不到
short_seq_prob:长度小于“max_seq_length”的样本比例。因为在fine-tune过程里面输入的target_seq_length是可变的（小于等于max_seq_length），那么为了防止过拟合也需要在pre-train的过程当中构造一些短的样本
max_predictions_per_seq:一个句子里最多有多少个[MASK]标记
masked_lm_prob:多少比例的Token被MASK掉
rng:随机率

###def create_instances_from_document
>>一个文档中抽取训练样本,重中之重

```
def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                   masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    document = all_documents[document_index]
    # 为[CLS], [SEP], [SEP]预留三个空位
    max_num_tokens = max_seq_length - 3
    target_seq_length = max_num_tokens  # 以short_seq_prob的概率随机生成（2~max_num_tokens）的长度
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        # 将句子依次加入current_chunk中，直到加完或者达到限制的最大长度
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end`是第一个句子A结束的下标
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                tokens_b = []
                is_random_next = False
                # `a_end`是第一个句子A结束的下标
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)
                    # 随机的挑选另外一篇文档的随机开始的句子
                    # 但是理论上有可能随机到的文档就是当前文档，因此需要一个while循环
                    # 这里只while循环10次，理论上还是有重复的可能性，但是我们忽略
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break
                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # 对于上述构建的随机下一句，我们并没有真正地使用它们
                    # 所以为了避免数据浪费，我们将其“放回”
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                # 如果太多了，随机去掉一些
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1
                tokens = []
                segment_ids = []
                # 处理句子A
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                # 句子A结束，加上【SEP】
                tokens.append("[SEP]")
                segment_ids.append(0)
                # 处理句子B
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                # 句子B结束，加上【SEP】
                tokens.append("[SEP]")
                segment_ids.append(1)
                # 调用 create_masked_lm_predictions来随机对某些Token进行mask
                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(tokens,
                                                                  masked_lm_prob,
                                                                  max_predictions_per_seq,
                                                                  vocab_words, rng)
                instance = TrainingInstance(tokens=tokens,
                                            segment_ids=segment_ids,
                                            is_random_next=is_random_next,
                                            masked_lm_positions=masked_lm_positions,
                                            masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    return instances
```
###def create_masked_lm_predictions
>>真正的mask在这里实现

```
def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    cand_indexes = [] # [CLS]和[SEP]不能用于MASK
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
            # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
            # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
     # 按照下标重排，保证是原来句子中出现的顺序
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)
```
>>代码流程是这样的:首先嫁给你一个句子随机打乱,并确定一个句子的15%是多少个token，设num_to_predict,然后对于[0,是多少个token，设num_to_predict]token，以80%的概率替换为[mask],10%的概率替换，10%的概率保持,这样就做到了对于15%的toke80% [mask],10%替换,10%保持。而预测的不是那15%的80（标注问题），而是全部15%。为什么要mask呢？你想啊，我们的目的是得到这样一个模型:输入一个句子，输出一个能够尽可能表示该句子的向量(用最容易理解的语言就是我们不知道输入的是什么玩意，但是我们需要知道输出的向量是什么),如果不mask直接训练那不就相当于用1来推导1？而如果我们mask一部分就意味着并不知道输入(至少不知道全部),至于为什么要把15不全部mask，我觉得这个解释很不错，但是过于专业化:
* 如果把 100% 的输入替换为 [MASK]：模型会偏向为 [MASK] 输入建模，而不会学习到 non-masked 输入的表征。
* 如果把 90% 的输入替换为 [MASK]、10% 的输入替换为随机 token：模型会偏向认为 non-masked 输入是错的。
* 如果把 90% 的输入替换为 [MASK]、维持 10% 的输入不变：模型会偏向直接复制 non-masked 输入的上下文无关表征。
所以，为了使模型可以学习到相对有效的上下文相关表征，需要以 1:1 的比例使用两种策略处理 non-masked 输入。论文提及，随机替换的输入只占整体的 1.5%，似乎不会对最终效果有影响（模型有足够的容错余量）。
通俗点说就是全部mask的话就意味着用mask来预测真正的单词,学习的仅仅是mask(而且mask的每个词都不一样，学到的mask表示也不一样，很显然不合理)，加入10%的替换就意味着用错的词预测对的词，而10%保持不变意味着用1来推导1，因此后两个10%的作用其实是为了学到没有mask的部分。
或者还有一种解释方式: 因为每次都是要学习这15%的token，其他的学不到(认识到这一点很重要)倘若某一个词在训练模型的时候被mask了，而微调的时候出现了咋办？因此不管怎样，都必须让模型好歹"认识一下"这个词.
## tokenization.py
>>按照`create_pretraining_data.py`中`main`的调用顺序，先看`FullTokenizer`类

###FullTokenizer
```
class FullTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)
```
>>在`__init__`中可以看到，又得先分析`BasicTokenizer`类和`WordpieceTokenizer`类(哎呀真烦，最后在回来做超链接吧),除此之外就是调用了几个小函数,`load_vocab`它的输入参数是bert模型的词典,返回的是一个`OrdereDict`:{词:词号}.其他的不说了，没啥意思。

###class BasicTokenizer
>>目的是根据空格，标点进行普通的分词，最后返回的是关于词的列表，对于中文而言是关于字的列表。

```
class BasicTokenizer(object):
  def __init__(self, do_lower_case=True):
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
  ##其实就是把字符串转为了list，分英文单词和中文单词处理
  ##eg:Mr. Cassius crossed the highway, and stopped suddenly.转为['mr', '.', 'cassius', 'crossed', 'the', 'highway', ',', 'and', 'stopped', 'suddenly', '.']
    text = convert_to_unicode(text)
    text = self._clean_text(text)
    text = self._tokenize_chinese_chars(text)
    orig_tokens = whitespace_tokenize(text)#无需细说，就是把string按照空格切分为list
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)#这个函数干了什么我也没看明白,但是对正题流程不重要,略过吧
      split_tokens.extend(self._run_split_on_punc(token))
    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1
    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    # 按字切分中文，其实就是英文单词不变,中文在字两侧添加空格
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    # 判断是否是汉字,这个函数很有意义，值得借鉴
    # refer：https://www.cnblogs.com/straybirds/p/6392306.html
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True
    return False

  def _clean_text(self, text): # 去除无意义字符以及空格
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)
```

###class WordpieceTokenizer
>>这个才是重点,跑test的时候出现的那些##都是从这里拿来的，其实就是把未登录词在词表中匹配相应的前缀.

```
class WordpieceTokenizer(object):
  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    text = convert_to_unicode(text)
    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue
      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end
      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens
```
>>tokenize说明: 使用贪心的最大正向匹配算法
  eg:input = "unaffable" output = ["un", "##aff", "##able"],首先看"unaffable"在不在词表中，在的话就当做一个词，也就是WordPiece，不在的话在看"unaffabl"在不在，也就是`while`中的`end-=1`,最终发现"un"在词表中,算是一个WordPiece,然后start=2,也就是代码中的`start=end`,看"##affable"在不在词表中,在看"##affabl"(##表示接着前面)，最终返回["un", "##aff", "##able"].注意，这样切分是可逆的，也就是可以根据词表重载"攒回"原词，以此便解决了oov问题.