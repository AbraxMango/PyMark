# 文本预处理

## 文本预处理的主要环节

- 文本基本处理
  - 分词
  - 词性标注命名实体识别
- 文本张量表示
  - one-hot编码
  - Word2vec
  - Word Embedding
- 文本语料的数据分析
  - 标签数量分布
  - 句子长度分布
  - 词频统计与关键词词云
- 文本特征处理
  - 添加n-gram特征
  - 文本长度规范
- 数据增强
  - 回译数据增强法

## jieba的使用

### 精确模式分词

> 试图将句子最精确的切开，适合文本分析

```python
import jieba

content = "我是理塘丁真，这是我的好朋友雪豹"

# 返回一个生成器对象
jieba.cut(content, cut_all=False)  # cut_all默认为False
<generator object Tokenizer.cut at 0x000001F360F2E980>


# 若需要直接返回列表内容, 使用jieba.lcut即可
jieba.lcut(content, cut_all=False)
['我', '是', '理塘', '丁真', '，', '这', '是', '我', '的', '好', '朋友', '雪豹']

```

### 全模式分词

> 把句子中所有可以成词的词语都扫描出来，但不能消除歧义

```py
jieba.lcut(content, cut_all=True)
```

### 搜索引擎模式分词

> 在精确模式的基础上，对长词再次切分，提高召回率

```pyt
jieba.cut_for_search(content)
```

### 用户自定义词典

> 因为存在一些专有名词或是人名，需要人为的去添加自定义词典

- 词典格式：词语，词频(可省略)，词性(可省略)，用空格隔开，顺序不可颠倒

## 各词性的缩写

**名词 (n):**  
- 普通名词 (n): 一般的名词，如 "书"、"猫"。  
- 专有名词 (nr): 特定的人名、地名，如 "北京"、"李白"。  
- 机构名 (ni): 机构、组织名，如 "中国科学院"。  
- 地名 (ns): 地理名称，如 "长江"。  
- 其他名词 (nz): 其他类型的名词，如 "第七"。  

**动词 (v):**  
- 动词 (v): 一般动词，如 "跑"、"吃"。  
- 及物动词 (vg): 及物动词，如 "看"。  
- 不及物动词 (vi): 不及物动词，如 "来"。  

**形容词 (a):**  
- 形容词 (a): 一般形容词，如 "高"、"美丽"。  

**副词 (ad):**  
- 副词 (ad): 修饰动词或形容词的词，如 "非常"、"很"。  

**介词 (p):**  
- 介词 (p): 引导介词短语，如 "在"、"从"。  

**连词 (c):**  
- 连词 (c): 用于连接词、短语或句子，如 "和"、"但是"。  

**助词 (u):**  
- 助词 (u): 语法助词，如 "的"、"了"、"着"。  

**代词 (r):**  
- 代词 (r): 代替名词的词，如 "我"、"你"、"他"。  

**数词 (m):**  
- 数词 (m): 表示数量的词，如 "一"、"二"、"百"。  

**量词 (q):**  
- 量词 (q): 与名词搭配使用的词，如 "个"、"本"、"张"。  

**叹词 (e):**  
- 叹词 (e): 表达感情、态度的词，如 "啊"、"哎"。  

**时间词 (t):**  
- 时间词 (t): 表示时间的词，如 "昨天"、"明天"。  

**其他 (x):**  
- 其他 (x): 无法归类的词，如 "呵呵"、"哈"。



- 词典样式如下

```pyt
快船队 13 nr
理塘丁真 5 nr
我测 13 v
```

- 加载

```pytho
jieba.load_userdict("... .txt")
```

## hanlp的使用

> 流行中英文处理工具包，基于tensorflow2.0

### 查看模型

```pyt
import hanlp

r = hanlp.pretrained.tok.ALL

for k in r.keys():
    print(k)
```

### 对中文分词

```pytho
import hanlp

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
r = tok(['我是理塘丁真，这是我的好朋友芝士雪豹'])

print(r)
```

### 并行分词

> 无论是CPU还是GPU，同时传入多个句子都将并行分词。也就是说，仅花费一个句子的时间就可以处理多个句子。然而工作研究中的文本通常是一篇文档，而不是许多句子。此时可以利用HanLP提供的分句功能和流水线模式优雅应对，既能处理长文本又能并行化。只需创建一个流水线pipeline，第一级管道分句，第二级管道分词

```pyto
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
HanLP = hanlp.pipeline() \
		.append(hanlp.utils.rules.split_sentence) \
		.append(tok)

r = HanLP('欢迎来到江西理工大学。我们的校训是“志存高远，责任为先”。')
print(r)
```

### 自定义字典

- 强制模式 - dict_force

```pyt
import hanlp

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tok.dict_force = {'川普'}  # 强制输出

r = tok('今天我和川普通电话')
print(r)
```

**但是有的时候强制输出会导致歧义。而自定义语句越长，越不容易发生歧义**

```py
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tok.dict_force = {'川普通电话': ['川普', '通', '电话']}  # 强制输出

r = tok('今天我和川普通电话', '银川普通人与川普通电话讲四川普通话')
print(r)
```

```py
import hanlp

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tok.dict_force = tok.dict_combine = None

r = tok(['今天我和川普通电话', '银川普通人与川普通电话讲四川普通话', '川普是美国总统'])
print(r)

```

**在上面这个例子中，并没有对词典中的’川普‘施加任何影响，但丰富的上下文促进了神经网络对语境的理解，使其得出了正确的结果**

- 合并模式 - dict_combine

> 合并模式的优先级低于统计模型，即dict_combine会在统计模型的分词结果上执行最长匹配并合并匹配到的词条。一般情况下，推荐使用该模式。比如加入”美国总统“后会合并[”美国”，“总统”]，而并不会合并[’美国‘，’总‘，’统筹部‘]为[’美国总统’，‘筹部’]

```py
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tok.dict_force = None
tok.dict_combine = {'美国总统'}

r = tok(['川普是美国总统', '美国总统筹部部长是谁'])
```

- 空格单词

> 含有空格，制表符等(Transformer tokenizer去掉的字符)的词语需要用tuple的形式提供：

```py
tok.dict_combine = {('iPad', 'Pro'), '2个空格'}  # 其中'()'内的是空格单词的内容
tok("如何评价iPad Pro?iPad  Pro有2个空格")  # 输出的'iPadPro'会是一个整体而不分开
```

### 单词位置 - config.output_spans

> HanLP支持每个单词在文本的原始位置，以便用于搜索引擎等场所。在词法分析中，非语素字符(空格，换行，制表符等)会被剔除，此时需要额外的位置信息才能定位每个单词

```py
tok.config.output_spans = True
r = tok(['欢迎来到江西理工大学。我们的校训是“志存高远，责任为先”。'])

[out]:[[['欢迎', 0, 2], ['来到', 2, 4], ['江西理工大学', 4, 10], ['。', 10, 11], 
['我们', 11, 13], ['的', 13, 14], ['校训', 14, 16], ['是', 16, 17], ['“', 17, 18], 
['志存高远', 18, 22], ['，', 22, 23], ['责任', 23, 25], ['为', 25, 26], ['先', 26, 27], 
['”', 27, 28], ['。', 28, 29]]]
```

**返回格式为三元组(单词，单词的起始下标，终止下标**

### 词性标注

> 词性标注任务的输入为已分词的一个或多个句子

```py
pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
result = pos(["我", "的", "希望", "是", "希望", "张晚霞", "的", "背影", "被", "晚霞", "映红", "。"])
```

**其中的两个”希望“的词性不同**

- 自定义词典 - dict_tags

```py
pos.dict_tags = {'HanLP': 'state-of-the-art-tool'}
pos(["HanLP", "为", ...])

[out]:['state-of-the-art-tool',
 'P',
...
]

pos.dict_tags = {('的', '希望'): ('补语成分', '名词'), '希望': '动词'}
pos(["我", "的", "希望", "是", "希望", "张晚霞", "的", "背影", "被", "晚霞", "映红", "。"])

[out]:
['PN', '补语成分', '名词', 'VC', '动词', 'NR', 'DEG', 'NN', 'LB', 'NR', 'VV', 'PU']
```

### 命名实体识别

> 命名实体识别任务的输入为已分词的句子

```py
ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
print(ner([["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"], ["阿婆主", "来到", "北京", "立方庭", "参观", "自然", "语义", "科技", "公司", "。"]], 
tasks='ner*'))

[out]:[[('2021年', 'DATE', 0, 1)], [('北京', 'LOCATION', 2, 3), 
('立方庭', 'LOCATION', 3, 4), ('自然语义科技公司', 'ORGANIZATION', 5, 9)]]
```

**每个四元组表示 [命名实体，类型标签，起始下标，终止下标] ，下标指的是命名实体在单词数组中的下标**

- 自定义词典

  - 白名单词典 - dict_whitelist

  > 白名单词典中的词语会被尽量输出。当然，以HanLP统计为主，词典的优先级很低

  ```py
  ner.dict_whitelist = {'午饭后': 'TIME'}
  ner(['2021年', '测试', '高血压', '是', '138', '，', '时间', '是', '午饭', '后', '2点45', '，', '低血压', '是', '44'])
  
  [out]:[('2021年', 'DATE', 0, 1),
   ('138', 'INTEGER', 4, 5),
   ('午饭后', 'TIME', 8, 10),
   ('2点45', 'TIME', 10, 11),
   ('44', 'INTEGER', 14, 15)]
  ```

  - 强制词典

  > 直接干预统计模型预测的标签，拿到最高优先级的权限

  ```py
  ner.dict_tags = {('名字', '叫', '金华'): ('O', 'O', 'S-PERSON')}
  ner(['他', '在', '浙江', '金华', '出生', '，', '他', '的', '名字', '叫', '金华', '。'])
  
  [out]:[('浙江', 'LOCATION', 2, 3), ('金华', 'LOCATION', 3, 4), ('金华', 'PERSON', 10, 11)]
  ```

  - 黑名单词典 - dict_blacklist

  > 黑名单中的词语绝不会被当作命名实体

  ```py
  ner.dict_blacklist = {'金华'}
  ner(['他', '在', '浙江', '金华', '出生', '，', '他', '的', '名字', '叫', '金华', '。'])
  
  [out]:[('浙江', 'LOCATION', 2, 3)]
  ```

  
