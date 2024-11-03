:thinking:

# 文本张量的表示方法

> 将一段文本使用张量进行表示，其中一般将词汇表示成向量，称作词向量，再由各个词向量按顺序组成矩阵文本形成文本表示。其作用就是能够使语言文本可以作为计算机处理程序的输入，进行接下来一系列的解析工作

- 文本张量的表示方法
  - one-hot编码
  - Word2vec
  - Word Embedding

## one-hot编码

> 又称独热编码，将每个词表示成具有n个元素的向量，这个词向量中只有一个元素是1，其他元素都是0，不同词汇为0的位置不同，其中n的大小是整个语料中不同词汇的总数

### 编码实现

```py
# 导入用于对象保存与加载的joblib
import sklearn.externals
import joblib
# 导入keras中的词汇映射器Tokenizer
from keras.preprocessing.text import Tokenizer

# 假定vocab为语料集所有不同词汇集合
vocab = {"周杰伦", "陈奕迅", "王力宏", "蔡徐坤", "陈立农", "范丞丞"}
# 实例化一个词汇映射对象
t = Tokenizer(num_words=None, char_level=False)
# 使用映射器拟合现有文本数据
t.fit_on_texts(vocab)

for token in vocab:
    zero_list = [0] * len(vocab)
    # 使用映射器转换现有文本数据，每个词汇对应从1开始的自然数
    # 返回样式如：[[2]]，取出其中的数字需要使用[0][0]
    token_index = t.texts_to_sequences([token])[0][0] - 1
    zero_list[token_index] = 1
    print(token, "的one-hot编码为:", zero_list)
		
# 使用joblib工具保存映射器，以便之后使用
tokenizer_path = "./Tokenizer"
joblib.dump(t, tokenizer_path)

[out]:
王力宏 的one-hot编码为: [1, 0, 0, 0, 0, 0]
范丞丞 的one-hot编码为: [0, 1, 0, 0, 0, 0]
蔡徐坤 的one-hot编码为: [0, 0, 1, 0, 0, 0]
周杰伦 的one-hot编码为: [0, 0, 0, 1, 0, 0]
陈奕迅 的one-hot编码为: [0, 0, 0, 0, 1, 0]
陈立农 的one-hot编码为: [0, 0, 0, 0, 0, 1]
```

### one-hot编码器的使用

```py
import sklearn.externals
import joblib

t = joblib.load(tokenizer_path)

# 编码token为"蔡徐坤"
token = "蔡徐坤"
# 使用t获得token_index
token_index = t.texts_to_sequences([token])[0][0] - 1
# 初始化一个zero_list
zero_list = [0]*len(vocab)
# 令zero_list的对应索引为1
zero_list[token_index] = 1
print(token, "的one-hot编码为:", zero_list)
```

### 优劣势

- 优势：操作简单，容易理解
- 劣势：完全割裂了词与词之间的联系，而且在大语料集下，每个向量的长度过长，占据大量内存

##  Word2vec

> 一种流行的将词汇表示成向量的无监督训练方法，该过程将构建神经网络模型，将网络参数作为词汇向量的表示，它包含CBOW和skipgram两种训练模式

###  CBOW(Continuous bag of words)模式

> 给定一段用于训练的文本语料，再选定某段长度(窗口)作为研究对象，使用上下文预测目标词汇

![CBOW预测模式](https://img-blog.csdnimg.cn/22905411bbb14801a3c410a347a356dc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBARmVhdGhlcl83NA==,size_20,color_FFFFFF,t_70,g_se,x_16)

- 使用前后词汇对目标进行预测

### skipgram模式

> 给定一段用于训练的文本语料，再选定某段窗口作为研究对象，使用目标词汇预测上下文词汇

![skipgram](https://img-blog.csdnimg.cn/20190421163242268.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pqMTEzMTE5MDQyNQ==,size_16,color_FFFFFF,t_70)

- 使用目标词汇对前后词汇进行预测

### 使用fasttext工具实现Word2vec的训练和使用

- 获取训练数据

```py
# 在命令提示行中操作
# 通过Matt Mahoney网站下载
# 首先创建一个存储数据的文件夹data
$ mkdir data
# 使用curl下载数据的压缩包
$ curl -O http://mattmahoney.net/dc/enwik9.zip -P data -i https://pypi.tuna.tsinghua.edu.cn/simple
# 解压
# 查看原始数据
$ head -10 data/enwik9  # 可以发现原始数据中包含很多XML/HTML等格式的不需要的内容
# 使用自带的脚本以对原始数据进行处理
$ perl wikifil.pl data/enwik9 > data/fil9

```

- 训练词向量

```py
# 在训练词向量过程中，我们可以设定很多超参数来调节我们的模型效果，如：
# 无监督模式：'skipgram'或'cbow'，默认为'skipgram'，实践中skipgram模式在利用子词比cbow好
# 词嵌入维度dim：默认100，但随着语料库的增大，词嵌入的维度往往也更大
# 循环次数epoch：默认5，但当数据集足够大，可能不需要那么多次
# 学习率lr：默认0.05，根据经验，建议选择[0.01, 1]范围内
# 使用的线程数thread：默认12个线程，一般建议和cpu核数相同

model = fasttext.train_unsupervised('data/fil9', "cbow", dim=300, epoch=1, lr=0.1, thread=8)
```

- 模型效果检验

```py
# 检查单词向量质量的一种简单方法就是查看其邻近单词，通过我们主观判断这些邻近单词是否与目标
# 单词相关

# 查找"运动"的邻近单词，可以发现"体育网"，”运动汽车"，"运动服"等
model.get_nearest_neighbors('sports')

[out]:[(0.841463252325, 'sportsnet'), (0.813390420, 'sportsuit') ... ]
```

- 模型的保存和重加载

```py
# 使用save_model保存模型
model.save_model("fil9.bin")

# 使用fasttext.load_model加载模型、
model = fasttext.load_model("fil9.bin")
model.get_word_vector("the")
```

## Word Embedding

- 什么是Word Embedding(词嵌入)
  - 通过一定的方式将词汇映射到指定维度(一般是更高维度)的空间
  - 广义的word embedding包括所有密集词汇向量的表示方法，如之前学习的word2vec，即可认为是word embedding的一种
  - 狭义的word embedding是指在神经网络中加入的embedding词，对整个网络进行训练的同时产生embedding矩阵(embedding层的参数)，这个embedding矩阵就是训练过程中所有输入词汇的向量表示组成的矩阵

### word embedding的可视化分析

- 通过使用tensorboard可视化嵌入的词向量

```py
import torch
from torch.utils.tensorboard import SummaryWriter

# 实例化一个摘要写入对象
writer = SummaryWriter()

# 随机初始化一个100x50的矩阵，表示词嵌入矩阵
embedded = torch.randn(100, 50)

# 使用正确的编码打开文件
with open("D:/dev/python/pyWork/NLP/data/name/vocab100.csv", "r", encoding="utf-8") as f:
    meta = [line.strip() for line in f]

# 将嵌入和词汇添加到TensorBoard
writer.add_embedding(embedded, metadata=meta)

# 关闭writer
writer.close()
```

- 在终端启动tensorboard服务

```py
$ tensorboard --logdir runs
```

