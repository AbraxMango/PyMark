# Scraper

## Request

> request是一个简单的Python HTTP请求库

- 作用是发送请求获取响应数据

使用：

1. 导入模块
2. 发送get请求，获取响应
3. 从响应中获取数据

```python
import request

response = request.get('<http://www.baidu.com>')

# response.encoding = 'utf-8'
# print(response.txt)

print(response.content.decode()) # 自动将二进制转换为utf-8
```

## BeautifulSoup

> 可从HTML或XML文件中提取数据的Python库

```python
pip install bs4
pip install lxml
```

### BeautifulSoup对象

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup("<html>data</html>", 'lxml')
```

- BeautifulSoup对象：代表要解析整个文档树
- 支持遍历文档树和搜索文档树中描述的大部分的方法

### find方法

```python
find(self, name=None, attrs={}, recursive=True, text=None, **kargs)
```

- name：标签名
- attrs：属性字典，可以指定标签的特定属性来进行搜索
- recursive：是否递归搜索，默认为True
- text：搜索包含特定文本内容的标签
- return：返回查找到的第一个元素对象

💡若要查找某个标签的全部内容，可以使用soup.find_all(name)

```python
# Tag 对象
print('标签名', a.name)
print('标签所有属性', a.attrs)
print('标签文本内容', a.text)
```

## 正则表达式

> 正则表达式是一种字符串匹配的模式

作用：

- 检查一个字符串是否含有某种子串
- 替换匹配的子串
- 提取某个字符串中匹配的子串

```python
import re

rs = re.findall('abc', 'abcabc') # 前面abc是正则表达式，后面的是需要被查找的字符串
rs = re.findall('a.c', 'abc') # 可以匹配除换行符以外的字符
rs = re.findall('a\\.c', 'a.c') # 让'.'转换为普通的字符而非转义字符
rs = re.findall('a[bc]d', 'abd') # || ‘或’关系
```

## re.findall()方法

API

```python
re.findall(pattern, string, flags=0)
```

参数

- pattern：正则表达式
- string：从哪个字符串中查找
- flags：匹配模式
  - re.DOTALL/S：’.’匹配任意的字符
  - re.IGNORECASE/I：忽略大小写进行匹配
  - re.MULTILINE/M：多行匹配，影响 ^ 和 $ 使 ^ 和 $ 匹配每一行的开始和结束，而不仅仅是整个字符串的开始和结束
  - re.VERBOSE/X：允许正则表达式更易读，忽略空白和注释
  - re.ASCII/A：使 \w, \W, \b, 和 \B 只匹配 ASCII 字符，而不匹配 Unicode 字符
  - re.LOCALE/L：根据当前区域设置，影响\w, \W, \b, 和 \B的匹配行为
- return：扫描整个string字符串，返回所有与pattern匹配的列表

💡re.findall(”a(.+)bc”, “a\nbc”, re.DOTALL)的返回结果是”\n”，因为这里使用了()来进行分组

## r原串的使用

```python
rs = re.findall('a\\\\nbc', 'a\\\\nbc')
[out]:[]

rs = re.findall(r'a\\\\nbc', 'a\\\\nbc')
[out]:['a\\\\nbc']
```

 💡拓展：可以解决写正则的时候，不符合PEP8规范的问题

## 实例

```python
import requests
from bs4 import BeautifulSoup
import re

response = requests.get('http://ncov.dxy.cn/ncovh5/view/pneumoniaj')
page = response.content.decode()

soup = BeautifulSoup(page, 'lxml')
script = soup.find(id='getListByCountryTypeService2true')

countries_text = script.text

json_str = re.findall(r'(\[.*\])', countries_text)
```

