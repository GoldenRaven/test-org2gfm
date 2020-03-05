- [Python最佳实践指南](#orgbc54026)
  - [写出优雅的Python代码](#orgc32164c)
    - [代码风格](#org265c0d1)
    - [阅读好的代码](#orgb6e0061)
- [Python 编程经验分享](#orge88be4d)
  - [常用函数](#orgfcdfd2b)

本笔记是我在学习过程中阅读过的内容，以及自己的总结，原作者内容请跟随链接跳转。


<a id="orgbc54026"></a>

# Python最佳实践指南

本章内容是"Hitchhiker's Guide to Python"的中文翻译，来源于项目地址：[https://github.com/Prodesire/Python-Guide-CN](https://github.com/Prodesire/Python-Guide-CN)，我发现这个项目是源于 [这个有趣的项目: HelloGitHub](https://github.com/521xueweihan/HelloGitHub)的推荐。


<a id="orgc32164c"></a>

## 写出优雅的Python代码


<a id="org265c0d1"></a>

### 代码风格

如果您问Python程序员最喜欢Python的什么，他们总会说是Python的高可读性。 事实上，高度的可读性是Python语言的设计核心。这基于这样的事实：代码的 阅读比编写更加频繁。

Python代码具有高可读性的其中一个原因是它的相对完整的代码风格指引和 “Pythonic” 的习语。

当一位富有经验的Python开发者（Pythonista）指出某段代码并不 “Pythonic”时， 通常意味着这些代码并没有遵循通用的指导方针，也没有用最佳的（最可读的）方式 来表达意图。

在某些边缘情况下，Python代码中并没有大家都认同的表达意图的最佳方式，但这些情况 都很少见。

-   一般概念
    -   每行一个声明
    -   函数参数
    -   避免魔法方法
-   习语 (Idiom)
    -   解包（unpacking）
    -   创建一个被忽略的变量（如果您需要赋值（比如，在 解包（Unpacking） ）但不需要这个变量， 请使用 `__` ，而不是 `_` ）
    -   创建一个含N个列表的列表（用列表推导）
    -   根据列表来创建字符串（ `''.join(l)` ）
    -   Python事实上的代码风格指南（PEP8）
-   约定
    -   访问字典元素（ `x in d` ）
    -   过滤列表（ `filter()` ）
    -   如果只是要遍历列表，请考虑使用迭代器
    -   如果有其他变量引用原始列表，则修改它可能会有风险。但如果你真的想这样做， 你可以使用 切片赋值（slice assignment）
    -   请记住，赋值永远不会创建新对象
    -   创建一个新的列表对象并保留原始列表对象会更安全
    -   使用 `enumerate()` 函数比手动维护计数有更好的可读性。而且， 它对迭代器进行了更好的优化
    -   使用 `with open` 语法来读取文件会更好，因为它能确保您总是关闭文件
    -   行的延续(一个更好的解决方案是在元素周围使用括号。左边以一个未闭合的括号开头， Python 解释器会把行的结尾和下一行连接起来直到遇到闭合的括号)


<a id="orgb6e0061"></a>

### 阅读好的代码

成为优秀Python编写者的秘诀是去阅读，理解和领会好的代码。

良好的代码通常遵循 代码风格 中的指南,尽可能向读者表述地简洁清楚。

以下是推荐阅读的Python项目。每个项目都是Python代码的典范。

Howdoi是代码搜寻工具，使用Python编写。

Flask是基于Werkzeug和Jinja2，使用Python的微框架。它能够快速启动，并且开发意图良好。

Diamond是Python的守护进程，它收集指标，并且将他们发布至Graphite或其它后端。 它能够收集CPU,内存，网络，I/O，负载和硬盘指标。除此，它拥有实现自定义收集器的API，该API几乎能 从任何资源中获取指标。

Werkzeug起初只是一个WSGI应用多种工具的集成，现在它已经变成非常重要的WSGI实用模型。 它包括强大的调试器，功能齐全的请求和响应对象，处理entity tags的HTTP工具，缓存控制标头， HTTP数据，cookie处理，文件上传，强大的URL路由系统和一些社区提供的插件模块。

Requests是Apache2许可的HTTP库，使用Python编写。

Tablib是无格式的表格数据集库，使用Python编写。


<a id="orge88be4d"></a>

# Python 编程经验分享

内容涵盖编码技巧、最佳实践与思维模式等方面，词条[https://github.com/piglei/one-python-craftsman](https://github.com/piglei/one-python-craftsman)


<a id="orgfcdfd2b"></a>

## 常用函数

```python
%timeit func() # 考查操作的运行时间
map(func_obj, iter1, iter2) # 但是请尽量用生成器
map(lambda x:x+1, iter1, iter2) # lambda表达式
```