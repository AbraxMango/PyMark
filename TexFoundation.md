# Foundation

## AutoGrad

> 在整个Pytorch框架中，所有的神经网络本质上都是一个autograd package(自动求导工具包)，其提供了一个对Tensors上所有的操作进行自动微分的功能

### 关于torch.Tensor

- torch.Tensor是整个package中的核心类，如果将属性requirets_grad设置为True，它将追踪在这个类上定义的所有操作。当代码要进行反向传播的时候，直接调用.backward()就可以自动计算所有的梯度。在这个Tensor上的所有梯度将被累加进属性.grad中
- 关于自动求导的属性设置：可以通过设置.requires_grad=True来执行自动求导，也可以通过代码块的限制来停止自动求导
- 如果想终止一个Tensor在计算图中的追踪回溯，只需执行.detach()就可以将该Tensor从计算图中撤下，在未来的回溯计算中也不会再计算该Tensor
- 除了.detach()，如果想终止对计算图的回溯，也就是不在进行反向传播求导数的过程，也可以采用代码块的方式with torch.no_grad(): ，这种方式非常适用于对模型进行检测的时候，因为预测阶段不再需要对梯度进行计算

----



# 神经网络基础

## Word2vec

### 常规训练原理

两类：

1. continuous bag-of-words(CBOW)
2. continuous skip-gram

**其原理是使用了Sliding window(滑动窗口)来浏览数据，其中CBOW是根据滑动窗口所记录的context(上下文)进行预测，而skip-gram是根据target(目标词)预测**

- CBOW：根据上下文的单词得到各自的one-hot词向量，之后算其均值，以n分类为问题使用softmax以求得预测单词的词向量
- skip-gram：取得目标词的词向量，将其转换为词表大小，再进行softmax以预测

最后加以负采样，降低计算量，使得Word2vec的训练变得高效实用

### 其他训练技巧

- sub-sampling：平衡常见词和罕见词
- soft sliding window：非固定滑动窗口

## GRU

> 门控机制：对当前信息进行筛选，决定了哪些信息可以传递到下一层。GRU存在两个门控，分别为更新门和重置门

## LSTM

> 长短期记忆网络，可看作是RNN的一种变体

1. 遗忘门：决定当前状态有哪些信息可以从cell状态中进行移除的
2. 输入门：决定哪些新信息可以被添加到cell状态中
3. 输出门：决定cell状态中的哪些信息可以作为当前时刻的输出

**这三个门控机制使得LSTM能够有效地处理长期依赖问题，选择性地记忆或遗忘信息。**

## Bidirectional RNNs

> 双向RNN

双向RNN是一种特殊的循环神经网络结构，它包含两个方向的信息流：一个从过去到未来（前向），另一个从未来到过去（后向）。这种结构允许网络在做出预测时同时考虑过去和未来的上下文信息。双向RNN在许多序列建模任务中表现出色，特别是在自然语言处理领域，如命名实体识别、词性标注和机器翻译等。

----



# Attention

> 注意力机制

注意力机制是一种强大的深度学习技术，它允许模型在处理序列数据时，根据上下文动态地关注输入的不同部分。这种机制不仅提高了模型的性能，还增强了其可解释性。在自然语言处理中，注意力机制已经成为许多先进模型的核心组件，如Transformer架构。

- 在enconder端获得隐向量
- 进行点积计算得到注意力分数
- 使用softmax函数将注意力分数转换为概率分布
- 根据概率分布对输入进行加权求和，得到上下文向量
- 将上下文向量与当前隐状态结合，用于预测下一个词或生成输出

## 注意力机制的特点

- 解决了信息瓶颈
- 有效缓解了RNN中梯度消失的问题
- 给神经网络提供了一定的可解释性（注意力可视化）

----



# Transformer

- Architecture：encoder-decoder
- input：使用byte pair encoding + positonal encoding 来对文本进行切分
- Output：Transformer生成的输出序列是通过decoder逐步预测得到的，每个时间步都会利用之前生成的token作为输入，最后是一个Linear到softmax转换\
- Loss function：使用cross-entropy来计算交叉熵以更新模型参数
- Training：使用teacher forcing策略，即在训练时用真实的目标序列作为decoder的输入，而不是使用模型生成的输出

这些特性使得Transformer成为了处理序列到序列任务（如机器翻译、文本摘要等）的强大工具。与传统的循环神经网络（RNN）相比，Transformer不仅能够捕捉更长距离的依赖关系，还能够更高效地并行计算。

Transformer模型的核心创新在于其自注意力机制（self-attention mechanism）。这种机制允许模型在处理序列数据时，动态地关注输入序列中的不同部分。自注意力机制使得Transformer能够并行处理输入，大大提高了模型的训练效率和性能。