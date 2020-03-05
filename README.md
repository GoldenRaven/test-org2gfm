- [[Python学习笔记](Python_learning_notebook.md)](#orgf2d10c2)
- [[Python Pandas库笔记](pd_learning.md)](#org8375cd7)
- [Python Numpy库笔记](#org50c9869)
  - [Copy or View?](#org7d818f7)
  - [数组的in place改变](#orgad3dc03)
  - [常用函数](#orgef0f28a)
- [[《机器学习实战》学习笔记](handson-ml-learning.md)](#orgecc7683)
- [[二分类问题示例Kaggle Titanic](Kaggle_Titanic.md)](#org29ef383)
- [一些其他有用的东西](#org9a30135)

这个笔记是我个人在学习机器学习过程中的学习记录。其中有大量前人的总结成果，但此词条并不简单的重复 摘抄，而是我在学习中选择性地参考了不同资料的不同部分，自己学习的理解与总结。原词条都有相关链接， 请自行跳转。


<a id="orgf2d10c2"></a>

# [Python学习笔记](Python_learning_notebook.md)


<a id="org8375cd7"></a>

# [Python Pandas库笔记](pd_learning.md)


<a id="org50c9869"></a>

# Python Numpy库笔记


<a id="org7d818f7"></a>

## Copy or View?

-   vew
    -   Slice view
    -   Dtype view
-   shallow copy
-   deep copy


<a id="orgad3dc03"></a>

## 数组的in place改变

以下是不同的操作过程：

```python
# y = np.arange(8)
y += 1 # 更快
y = y + 1 # 更慢
```


<a id="orgef0f28a"></a>

## 常用函数

```python
import numpy as np
a = np.arange(3)
b = np.arange(3,6)
c = np.r_[a, b, 1, [3]] # 合并数组
d = np.c_[a, b] # 合并数组
e = np.ones((4, 1)) # 接收元组
d.shape
d.resize(2, 3) # 无返回值，将原数组形变，接收元组
f = d.reshape(（2,3）) # 返回变形后的数组，原数组不变，接收元组
```


<a id="orgecc7683"></a>

# [《机器学习实战》学习笔记](handson-ml-learning.md)


<a id="org29ef383"></a>

# [二分类问题示例Kaggle Titanic](Kaggle_Titanic.md)


<a id="org9a30135"></a>

# 一些其他有用的东西

-   [数据科学领域速查表](https://github.com/FavioVazquez/ds-cheatsheets)