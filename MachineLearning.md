# W

## Dataset

- sklearn.datasets：加载获取流行数据集
- sklearn.datasets.load_*()：获取小规模数据集，数据包含在datasets里，’*’为数据集名称
- sklearn.datasets.fetch_*(data_home=None, subset=’’)：获取大规模数据集，需要从网上下载，data_home表示数据集下载的目录，subset表示在某些数据集中，可以指定子集的类型，例如 ’train’ 或 ’test’

```python
# 鸢尾花
iris = load_iris()
print(iris)
print(type(iris))  # type为datasets.base.Bunch(字典格式)
```

### 键值对

- data：特征数据数组，是[n_sample*n_features]二维numpy.ndarray数组
- target：标签数组，是n_samples的一维数组
- DESCR：数据描述
- feature_names：特征数据，手写数字以及回归数据集没有
- target_name：标签名

### 数据集划分

> 由于对数据训练完之后需要测试数据，所以需要对数据集进行划分，以保留数据。一般训练集占70%~80%，测试集占20%~30%

API

```python
sklearn.model_selection.train_test_split(arrays, *options)
```

参数：

- arrays：样本数组，包含特征矩阵和标签
- test_size：测试集大小，可以为浮点数（0-1之间）或整数（样本数）
- train_size：训练集大小，可以为浮点数（0-1之间）或整数（样本数）
- random_state：随机数种子，用于重复实验
- shuffle：是否在分割之前对数据进行洗牌，默认为True
- return：x_train训练集特征值，x_test测试集特征值，y_train训练集目标值，y_test测试集目标值

```python
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
```

## 特征工程

> 数据和特征决定了机械学习的上限，而模型和算法只是逼近了这个上限

### 字典特征提取（特征离散化）

- sklearn.feature_extraction.DictVectorizer(sparse=True, …)
- DictVectorizer.fit_transform(X) X：字典或包含字典的迭代器，返回的是sparse(稀疏)矩阵
- DictVectorizer.inverse_transform(X) X：array数组或者sparse矩阵，返回转换之前的数据格式
- DictVectorizer.get_feature_names()：返回类别名称

```python
from sklearn.feature_extraction import DictVectorizer

data_1 = [{"city": "北京", "temperature": 100},
          {"city": "上海", "temperature": 60},
          {"city": "深圳", "temperature": 30}]

# 1.实例化一个转换器类
d_transfer = DictVectorizer(sparse=False)

# 2.调用fit_transform()
data_new_1 = d_transfer.fit_transform(data_1)

# print("   上海  北京  深圳")
print(data_new_1)
```

<aside> 💡

应用场景：pclass，sex数据集中类别特征比较多；公司，同事合作

</aside>

### 文本特征提取

方法一：CountVectorizer

- sklearn.feature_extraction.text.CountVectorizer(stop_words=[”mamba”, “out”, …], …) stop_words 停用词，用列表传
- CountVectorizer.fit_transform(X) X：文本或者包含文本字符串的可迭代对象，返回sparse矩阵
- CountVectorizer.inverse_transform(X) X：array数组或者sparse矩阵，返回转换之前的数据格式
- CountVectorizer.get_feature_names()：返回单词列表

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

data_2_En = ["life is short, i like python very very much", "life is long, i dislike python"]
data_2_Cn = ["大烟杆 嘴里 塞, 我 只抽 第五代", "比养的 快尝尝, 我 现在 肺痒痒"]  # 中文需要空格隔开才能分为词而非句子

# 1.实例化一个转换器类
t_transfer = CountVectorizer()

# 2.调用fit_transform
data_new_2_En = t_transfer.fit_transform(data_2_En)

print(t_transfer.get_feature_names_out())  # 获取特征名字
print(data_new_2_En.toarray())  # 用toarray()方法以输出二位矩阵
data_new_2_Cn = t_transfer.fit_transform(data_2_Cn)
print(t_transfer.get_feature_names_out())
print(data_new_2_Cn.toarray())

# 中文文本特征提取,  借助jieba自动分词
import jieba

data_3 = ["我的锐刻就像一根根糖, 口味丰富到难以想象",
          "只要你往我鼻嘴里放, 我谈笑间吸干太平洋",
          "核污水不要再乱排放, 鬼子想喝就解我裤裆",
          "大家要是买不到锐刻, 往我的嘴里吸，都一样"]

fyy = []
for sen in data_3:
    fyy.append(" ".join(jieba.cut(sen)))  # 返回的是词语生成器

print(fyy)
print("--------------------")
# # 1.实例化一个转换器类
t_transfer = CountVectorizer()

# 2.调用fit_transform
data_new_3 = t_transfer.fit_transform(fyy)

print(t_transfer.get_feature_names_out())  # 获取特征名字
print(data_new_3.toarray())  # 用toarray()方法以输出二位矩阵
```

方法二：TfidfVectorizer

> Tf-idf文本特征提取，关键词对于一个文件集，或一个语料库的其中一份文件的重要程度。       词频 term frequency - tf，逆向文档率 inverse document frequency - idf ：总文件数目除以包含该词语文件的数目，将结果取10为底的对数。重要程度：tfidf(i, j) = tf(i, j) * idf(i)

```python
data_4 = ["在这么冷的天, 想抽根电子烟", "可锐克没有电, 可是雪豹已失联",
          "摘不下我虚伪的假面, 几句胡言被奉为圣谏", "尝一口你血汗的香甜, 可是钱飘进双眼"]

distance = []
for sen in data_4:
    distance.append(" ".join(jieba.cut(sen)))

# 1.实例化一个转换器类
Tf_transfer = TfidfVectorizer()

# 2.调用fit_transform
data_new_4 = Tf_transfer.fit_transform(distance)
print(data_new_4.toarray())

# ----- 图像特征提取 ----- (深度学习)

# ----- 特征预处理 -----
# 通过一些转换函数, 将特征数据转换成更加适合算法模型的特征数据过程 : 无量纲化, 不同规格的数据转换到同一规格
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
```

### 特征预处理

> 通过一些转换函数，将特征数据转换成更加适合算法模型的特征数据过程：无量纲化，不同规格的数据转换到同一规格

```python
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
```

1. 归一化 MinMaxScaler

- 健壮性较差，适合传统精确小数据场景

```python
# x' = (x - min) / (max - min) , x'' = x' * (mx - mi) + mi
# max/min 为每一列的最大/小值, x''为最终结果. mx和mi为指定区间值, 默认为1和0

# 1.获取数据
lst = np.array([[40920, 8.326976, 0.953952, 3], [14488, 7.153469, 1.673904, 2], [26052, 1.441871, 0.805124, 1],
                [75136, 13.147394, 0.428964, 1], [38344, 1.669788, 0.134296, 1], [72993, 10.141740, 1.032955, 1],
                [35948, 6.830792, 1.213192, 3], [42666, 13.276369, 0.543880, 3], [67497, 8.631577, 0.749278, 1],
                [35483, 12.273169, 1.508053, 3], [50242, 3.723498, 0.831917, 1], [63275, 8.385879, 1.669485, 1],
                [5569, 4.875435, 0.728658, 2], [51052, 4.680098, 0.625224, 1], [77372, 15.299570, 0.331351, 1],
                [43673, 1.889461, 0.191283, 1], [61364, 7.516754, 1.269164, 1], [69673, 14.239195, 0.261333, 1],
                [15669, 0.000000, 1.250185, 2], [28488, 10.528555, 1.304844, 3], [6487, 3.540265, 0.822483, 2],
                [37708, 2.991551, 0.833920, 1], [22620, 5.297865, 0.638306, 2], [28782, 6.593803, 0.187108, 3],
                [19739, 2.816760, 1.686209, 2], [36788, 12.458258, 0.649617, 3], [5741, 0.000000, 1.656418, 2],
                [28567, 9.968648, 0.731232, 3], [6808, 1.364838, 0.640103, 2]]).reshape(29, 4)
data_5 = pd.DataFrame(lst,
                      index=list(f"{i}" for i in range(1, 30)),
                      columns=list(["milage", "Liters", "Consumtime", "target"])
                      )
# print(data_5)
data_5 = data_5.iloc[:, :3]

# 2.实例化一个转换器类
pre_transfer = MinMaxScaler()  # feature_range=[]参数为指定区间, 默认为[0,1]

# 3.调用fit_transform
data_new_5 = pre_transfer.fit_transform(data_5)
print(data_new_5) # 转换完成
```

1. 标准化 StandardScaler

```python
# x' = (x - mean)/sigma
# mean 为平均值, sigma 为标准差

# 1.获取数据
# 2.实例化一个转换器类
pre_transfer = StandardScaler()

# 3.调用fit_transform
data_new_5 = pre_transfer.fit_transform(data_5)
print(data_new_5)  # 转换完成
```

### 特征降维

> 对象：二维数组 → 降低特征个数，得到一组“不相关”主变量的过程，以达到特征与特征之间不相关

1. 特征选择

- 数据中包含冗余或相关变量(特征，属性，指标)，旨在从原有特征中找出主要特征
- 1.Filter(过滤式)：探究特征本身特点，特征与特征和目标值之间关联 - 方差选择法，相关系数法
- 2.Embedded(嵌入式)：算法自动选择特征(特征与目标值之间的关联) - 决策树，正则化，卷积

API

```python
sklearn.feature_selection.VarianceThreshold(threshold = 0.0)
# 删除所有低方差特征
# Variance.fit_transform(X) X :numpy array格式的数据, 默认值是保留所有非零方差特征
```

- threshold : 临界值

```python
# # 1.获取数据
data_6 = pd.read_csv("D:/dev/resources/MLR/机器学xiday1资料/02-代码/factor_returns.csv")
data_6 = data_6.iloc[:, 1:10]

# # 2.实例化一个转换器类
fs_transfer = VarianceThreshold()

# # 3.调用fit_transform
data_new_6 = fs_transfer.fit_transform(data_6)
print(data_new_6)
```

1. 相关系数 r 0.4 | 0.7 | 1

```python
from scipy.stats import pearsonr

# 计算某两个变量之间的相关系数
r = pearsonr(data_6["pe_ratio"], data_6["pb_ratio"])
print(r)

# 特征与特征之间相关性很高 : 1.选其中一个 2.加权求和 3.主成分分析 ->
```

1. 主成分分析(PCA)

> 定义：高维数据转换为地位数据的过程，此过程可能会舍去原有数据，创造新的变量                         作用：数据维数压缩，尽可能降低原数据的维数，损失少量信息                                                                        应用：回归分析或聚类分析中

```python
from sklearn.decomposition import PCA

# API : sklearn.decomposition.PCA(n_components=None) n_components : 小数 表示保留百分之多少的信息, 整数 减少到多少特征
# PCA.fit_transform(X) X : numppy array 格式的数据, 返回指定维度的数组
data_7 = [[2, 8, 4, 5],
          [6, 3, 0, 8],
          [5, 4, 9, 1]]

# # 1.实例化一个转换器类
pca_transfer = PCA(n_components=2)  # 表示转换为两个特征

# 2.调用fit_transform
data_new_7 = pca_transfer.fit_transform(data_7)
print(data_new_7)
```

## 实战

```python
# 探究Instacart消费者将购买哪些产品

# 1.获取数据
order_products = pd.read_csv("D:/dev/resources/MLR/机器学xiday1资料/02-代码/instacart/order_products__prior.csv")
products = pd.read_csv("D:/dev/resources/MLR/机器学xiday1资料/02-代码/instacart/products.csv")
orders = pd.read_csv("D:/dev/resources/MLR/机器学xiday1资料/02-代码/instacart/orders.csv")
aisle = pd.read_csv("D:/dev/resources/MLR/机器学xiday1资料/02-代码/instacart/aisles.csv")

# 2.合并表
tab1 = pd.merge(aisle, products, on=["aisle_id", "aisle_id"])
tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])

# 3.找到user_id和aisle之间的关系
data = pd.crosstab(tab3["user_id"], tab3["aisle"])[:10000]

# 4.PCA降维
pca_transfer = PCA(n_components=0.95)
data_new = pca_transfer.fit_transform(data)

print(data_new)
```

----

# T

## 转换器和预估器

- 转换器 transformer：特征工程 - 1.实例化 2.调用fit_transform

- 预估器 estimator：是一类实现了算法的API，分类，回归，无监督学习等算法都是其子类           工作流程：实例化estimator → estimator.fit(x_test, y_test) 测试集计算，训练 → 调用完毕，模型生成 → 模型评估 1. 对比真实值和预测值 y_predict=estimator.predict(x_test), y_test==y_predict?

  ```
                                2.计算准确率 estimator.score(x_test, y_test)
  ```

## K-近邻算法 KNN

- 核心思想：根据邻居判断类别
- 距离公式：欧式距离，曼哈顿距离，明可夫斯基距离
- 无量纲化处理：标准化

API

```python
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
```

参数说明：

- n_neighbors：用于分类的邻居数量，默认为5
- algorithm：用于计算最近邻的算法，可选'auto'、'ball_tree'、'kd_tree'或'brute'

KNN算法的优缺点： 优点：简单易懂，无需训练，对异常值不敏感 缺点：懒惰算法，计算量大，内存消耗大，预测效率低，对特征数量敏感；必须指定k值，k值选择不当则分类精度不能保证

使用场景：小规模数据，范围 1000 ~ 99999

案例 ：iris分类

```python
# 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3.特征工程 : 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)  # 用transform是因为上一步已经对x_train fit 过了(求过了标准差等)

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 1)对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print(y_test == y_predict)

    # 2)计算准确率
    score = estimator.score(x_test, y_test)
    print(score)
```

## 模型选择与调优

1. cross validation 交叉验证：将拿到的训练数据，分为训练集和验证集

   如：将数据分为四份，每份取一份为验证集，经过四组测试，每次更换验证集，将四组模型结果取平均值，又称4折交叉验证

2. Grid Search 超参数搜索-网格搜索：有很多参数是要手动指定的(如KNN中的k值)，这种叫超参数，每组超参数都采用交叉验证来评估，最后选出最优参数建立模型

API

```python
sklearn.model_selection.GridSearchCV(estimator, param_grid=None, cv=None)
```

- estimator：估计器
- param_grid：估计器参数(dict){”n_neighbors”:[1, 3, 5]}
- cv：指定几折交叉验证
- fit()：输入训练数据
- score()：准确率
- 结果分析：
  - 最佳参数：best_params_
  - 最佳结果：best_score_
  - 最佳估计器：best_estimator_
  - 交叉验证结果：cv_results_

案例：iris分类，并添加网格搜索和交叉验证

```python
# 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3.特征工程 : 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)  # 用transform是因为上一步已经对x_train fit 过了(求过了标准差等)

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier()

    # 5.加入GridSearchCV
    # 参数准备
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}  # 想要测试的k值
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)

    estimator.fit(x_train, y_train)

    # 6.模型评估
    # 1)对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print(y_test == y_predict)

    # 2)计算准确率
    score = estimator.score(x_test, y_test)
    print(score)

    print("best_params_:", estimator.best_params_)
    print("best_score_:", estimator.best_score_)
    print("best_estimator_:", estimator.best_score_)
    print("cv_results_:", estimator.cv_results_)
```

## 朴素贝叶斯算法 Native Bayes

- Bayes’ theorem 贝叶斯公式： P(B|A) = P(A) / P(B)，在B已经发生的条件下，A发生的概率
- Native Bayes 朴素贝叶斯：假设特征与特征之间相互独立 P(A, B) = P(A)P(B)
- Laplacian smoothing 拉普拉斯平滑系数：P(F1|C) = (Ni + a) / (N + am), 目的是防止计算出的分类概率为零，a指定系数一般为1，m为训练文档中统计出特征词的个数

API

```python
sklearn.naive_bayes.MultinomialINB(alpha=1.0)
```

- alpha = 1.0：表示使用拉普拉斯平滑（Laplace smoothing），这是最常用的设置
- alpha < 1.0：会增强平滑效果，适用于样本量小的情况
- alpha > 1.0：减弱平滑效果，适用于样本量大的情况

实例

```python
  	# 1.获取数据
    news = fetch_20newsgroups(subset="all")

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 3.特征工程 : 文本特征抽取-tfidf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4.朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 1)对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print(y_test == y_predict)

    # 2)计算准确率
    score = estimator.score(x_test, y_test)
    print(score)
```

## 决策树 Decision_tree

> 信息论基础 : 信息的衡量 - 信息量 - 信息熵 H(x) = -∑P(xi)logb(P(xi))

1. 决策树分类器

```python
sklearn.tree.DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=None)
```

- criterion : 默认gini系数, 也可以选择信息增益的熵entropy
- max_depth : 树的深度
- random_state : 随机种子

1. 决策树可视化

```python
sklearn.tree.export_graphviz(estimator, out_file=" ", feature_names=[" ", " "]  
```

- 该文件可以导出为DOT格式
- out_file : 文件路径
- feature_names : 特征名称
- 优点 : 简单的理解和解释, 可视化 缺点 : 可能会过拟合

实例

```python
# 1.获取数据集
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3.决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 4.模型评估
    # 1)对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print(y_test == y_predict)

    # 2)计算准确率
    score = estimator.score(x_test, y_test)
    print(score)

    # 5.决策树可视化
    # export_graphviz(estimator, out_file="iris_tree.dot")  在webgraphviz.com中查看树
```

## 随机森林 Random_forest

- 两个随机 :
  - 1.训练集随机 bootstrap 随机放回抽样
  - 2.特征随机 从M个特征中随机抽取m个特征 M >> m, 起到降维的效果

API

```python
sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, bootstrap=True, random_state=None, min_samples_split=2)
```

- 随机森林分类器
  - n_estimators : 森林中的树木数量
  - criterion : 分割特征的测量方法
  - max_depth : 树的最大深度
  - bootstrap : 是否在构建树时采用放回抽样
  - min_samples_split : 节点划分的最小样本数
  - min_samples_leaf : 叶子节点的最小样本数

----

# F

## 线性回归 Linear_regression

$$ y = w{_1}x{_1}+w{_2}x{_2}+...+w{_n}x{_n}+b $$

- b为偏置

<aside> 💡

线性关系一定是线性模型

线性模型不一定是线性关系

</aside>

优化使损失减小，接近真实值：

1. 正规方程

   直接计算w

   API

   ```python
   sklearn.linear_model.LinearRegression(fit_intercept=True)
   ```

   - fit_intercept：是否计算偏置
   - LinearRegression.coef_：回归系数
   - LinearRegression.intercept_：偏置

2. 梯度下降

   > 实现了随机梯度下降学习，支持不同的的los函数和正则化惩罚项来拟合线性回归模型

   不断迭代w

   API

   ```python
   sklearn.linear_model.SGDRegression(loss="squared_loss", fit_intercept=True, 
   learning_rate="invscaling", eta0=0.01)
   ```

   - loss：损失类型，”squared_loss”普通最小二乘法

   - fit_intercept：是否计算偏置

   - learning_rate：学习率填充”constant” “optimal” invscaling”

     ```
     对一个常数值的学习率来说，可以使用learning_rate=”constant”，并通过eta0来指定学习率
     ```

   - SGDRegression.coef_：回归系数

   - SGDRegression.intercept_：偏置

3. 回归性能评估 Mean Squared Error 评价机制

   $$ MSE =1/m{\ }*{\ }Σ(y{_i}-y{_a}{_b})^2 $$

API

```python
sklearn.metrics.mean_squared_error(y_true, y_pred)
```

- y_true：真实值
- y_pred：测试值
- return：浮点数

案例：波士顿房价预测

```python
# 获取数据
    data_url = "<http://lib.stat.cmu.edu/datasets/boston>"
    raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=22)

    # 特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 预估器
    estimator = LinearRegression()  # 正规方程
    # estimator = SGDRegressor()  # 梯度下降
    estimator.fit(x_train, y_train)

    print("coef: ", estimator.coef_)
    print("intercept: ", estimator.intercept_)

    # 模型评估 (不同于之前)
    y_predict = estimator.predict(x_test)
    print("房价预测: ", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差: ", error)
```

## 岭回归 Ridge

> 欠拟合 underfitting：一个假设在训练数据上不能获得更好的拟合，并且在测试数据集上也不能很好的拟合数据，此时认为这个假设出现了欠拟合的现象。解决方法：增加数据特征数量            过拟合 overfitting：一个假设在训练数据集能够获得比其他假设更好的拟合，但是在测试数据集上却不能很好的拟合数据，这就是过拟合现象。解决方法：正则化

API

```python
sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, solver="auto", 
normalizer=False)
```

- alpha：正则化力度[0, 1] | [1, 10]
- solver：根据数据自动选择优化方法，如果数据集，特征都比较大，用SAG
- normalize：数据是否进行标准化
- estimator.coef_：回归权重
- estimator.intercept_：回归偏置

案例：波士顿房价预测

```python
# 获取数据
    data_url = "<http://lib.stat.cmu.edu/datasets/boston>"
    raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=22)

    # 特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 预估器
    estimator = Ridge()  # 正规方程
    estimator.fit(x_train, y_train)

    print("coef: ", estimator.coef_)
    print("intercept: ", estimator.intercept_)

    # 模型评估 (不同于之前)
    y_predict = estimator.predict(x_test)
    print("房价预测: ", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差: ", error)
```

## 逻辑回归  Logistic_regression

> 是一类分类模型，逻辑回归的一个输入就是线性回归的输出

API

```python
sklearn.linear_model.LogisticRegression(solver="liblinear", penalty="l2", C=1.0)
```

- solver：优化求解方法
- penalty：正则化种类
- C：正则化力度

分类的评估方法：精准率和召回率(Precisoin&Recall)

API

```python
sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None)
```

- y_true：真实值
- y_pred：估计器预估目标值
- labels：指定类别对应的数字
- target_names：目标类别名称
- return：每个类别的精准率和召回率

<aside> 💡

样本不均衡时借助ROC曲线和AUC指标。AUC[0.5, 1]，越接近1越好

</aside>

API

```python
sklearn.metrics.roc_auc_score(y_true, y_score)
```

- y_true：每个样本真实类别
- y_score：预测得分

案例：癌症分类

```python
# 读取数据
    path = "<https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data>"
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(path, names=column_name)

    # 缺失值处理
    data = data.replace(to_replace="?", value=np.nan)  # 替换为Nan
    data = data.dropna()  # 删除样本

    # 筛选特征值和目标值
    x = data.iloc[:, 1:-1]
    y = data["Class"]

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # 特征工程 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 逻辑回归
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    print("coef: ", estimator.coef_)
    print("intercept: ", estimator.intercept_)

    # 模型评估
    # 1)对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print(y_test == y_predict)

    # 2)计算准确率
    score = estimator.score(x_test, y_test)
    print("score: ", score)

    # 查看精确率, 召回率, F1-score
    report = classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"])
    print(report)

    # 查看AUC指标
    y_test = np.where(y_test > 2.5, 1, 0)

    print("AUC ", roc_auc_score(y_test, y_predict))
```

## KMeans算法

K-Means聚类步骤

- 对于其他每个点计算到K个中心的距离，未知的点选择最近的一个聚类中心点作为标记类别
- 接着对着标记的聚类中心之后，重新计算出每个聚类的新中心点（平均值）
- 如果计算得出的新中心点与原中心点一样，那么结束，否则重新进行第二步过程
- K-超参数: 看需求取值

API

```python
sklearn.clusters.KMeans(n_clusters=n, init="k-means++")
```

- n_clusters：开始的聚类中心量
- init：初始化方法
- labels：默认标记类型

<aside> 💡

KMeans性能评估指标：

"高内聚, 低耦合" -> bi 外部距离最大化, ai 内部距离最小化

轮廓系数 SCi = (bi-ai) / max(bi, ai) [-1, 1], 1效果最好

</aside>

API

```python
sklearn.metrics.silhouette_score(X, labels)
```

- X：特征值
- labels：被聚类标记的目标值

优点：采用迭代式算法，直观易懂且非常实用

缺点：容易收敛到局部最优解(多次聚类)

<aside> 💡

聚类一般在分类前

</aside>

案例：K-Means对Instacart Market用户进行聚类

```python
  	# 数据降维
    # 1.获取数据
    order_products = pd.read_csv("D:/dev/resources/MLR/机器学xiday1资料/02-代码/instacart/order_products__prior.csv")
    products = pd.read_csv("D:/dev/resources/MLR/机器学xiday1资料/02-代码/instacart/products.csv")
    orders = pd.read_csv("D:/dev/resources/MLR/机器学xiday1资料/02-代码/instacart/orders.csv")
    aisle = pd.read_csv("D:/dev/resources/MLR/机器学xiday1资料/02-代码/instacart/aisles.csv")

    # 2.合并表
    tab1 = pd.merge(aisle, products, on=["aisle_id", "aisle_id"])
    tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])

    # 3.找到user_id和aisle之间的关系
    data = pd.crosstab(tab3["user_id"], tab3["aisle"])[:10000]

    # 4.PCA降维
    pca_transfer = PCA(n_components=0.95)
    data_new = pca_transfer.fit_transform(data)

    # 预估器流程
    estimator = KMeans()
    estimator.fit(data_new)

    y_predict = estimator.predict(data_new)

    # KMeans性能指标分析结果
    print("SCi ", silhouette_score(data_new, y_predict))
```

## 模型的保存与加载

```python
    import joblib

    joblib.dump(rf, "Season1.pkl")
    estimator = joblib.load("Season1.pkl")
```

