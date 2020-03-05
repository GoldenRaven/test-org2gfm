- [Titanic问题](#orge140e28)
  - [加载数据](#org56c3a1b)
  - [预览数据](#org396bb9d)
  - [清理数据](#org7b0b317)
    - [Completing](#orgc606bc9)
    - [Creating](#org6270517)
    - [Converting](#orga9f06bb)
  - [从训练集中分离出测试集](#orgcae925c)
  - [分析数据，找到关联性](#org357242f)
  - [模型化数据集](#org3e98444)
    - [背景知识](#orgdba7bcd)



<a id="orge140e28"></a>

# Titanic问题

预测Titanic号上乘客在经历沉船事件后是否能够生还。并在Kaggle网站提交预测结果。

疑问：

-   分离出测试集应该在预览数据之前还是之后？
-   清理数据时，是对整个数据集操作还是只对训练集操作？
-   交叉验证总是需要的吗？只是在出现过拟合时使用？

参考Jupyter Notebook: [titanic-a<sub>data</sub><sub>science</sub><sub>framework</sub><sub>to</sub><sub>achieve</sub><sub>99</sub><sub>accuracy.ipynb</sub>](titanic-a_data_science_framework_to_achieve_99_accuracy.ipynb)


<a id="org56c3a1b"></a>

## 加载数据

Kaggle网站上Titanic竞赛中的数据集test.csv指求解时的实例，去掉了标签'Survived'.

```python
import pandas as pd
df = pd.read_csv("train.csv") # , delimiter=',')
df2 = pd.read_csv("test.csv") # , delimiter=',')
```

-   要注意Python中赋值时，引用与复制数据的区别！

```python
data1 = df.copy(deep=False) #不复制df的indices和数据，只创建一个指向原数据的引用
data1 = df.copy(deep=True) #复制df的indices和数据，并在内存中创建新的对象
```

-   引用也是很有用的，尤其是在[清理数据](#org7b0b317)时（为什么要清理data-val？）

```python
data_clearner = [data1, df2] #可以一起清理
```


<a id="org396bb9d"></a>

## 预览数据

```python
df.info()
df.head()
df.tail()
df.sample(10)
```


<a id="org7b0b317"></a>

## 清理数据

4个'C':

-   **Correcting**: 更正异常值，离群值
-   **Completing**: 补全缺失信息
-   **Creating**: 创建新的特征，用以之后的分析
-   **Converting**: 转换数据的格式，以备后续的计算与呈现


<a id="orgc606bc9"></a>

### Completing

不推荐删除记录，尤其当它占的比例大时。最好impute. 对于定性值， 一般使用mode，对于定量值一般用中值、平均值或以平均值+随机化的标准差来代替。 还有针对具体问题更特殊的处理方法，如代之以某个小类别中的中值等。

```python
df.isna().sum() # 查看数据中的空值情况
df.isnull().sum() # 查看数据中的空值情况
df.describe(include='all') #数据的简单分析
df['Age'].fillna(df['Age'].median(), inplace=True) # 用中值来补全空值（定量值）
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# 用出现最的类别来补全空值（定性值）
drop_index = ['PassengerId', 'Ticket'] # index to drop
df.drop(drop_index, axis=1, inplace=True) # drop features/columns
```


<a id="org6270517"></a>

### Creating

特征工程：用已经存在的特征来创造新的特征，以检查是否对结果预测提供新的信息。

```python
df['FamilySize'] = df.['SibSp'] + df.['Parch'] + 1 # 新建特征
df['Alone'] = 0
df['Alone'].loc[df['FamilySize'] > 1] = 1 # 选择性赋值
df['Title'] = df['Name'].str.split( # 特征中字符串截取
   ', ', expand=True)[1].str.split('.', expand=True)[0]
df['FareBins'] = pd.cut(df['Fare'], 4) # 离散化连续值到区间
df['AgeBins'] = pd.qcut(df['Age'].astype(int), 5) # 离散化连续值到区间
# 清理类别数太少的类别
title_name = df['Title'].value_counts() < 10
df['Title'] = df['Title'].apply(lambda x: 'Misc' if title_name[x] else x)
```


<a id="orga9f06bb"></a>

### Converting

如对object类型的数据格式化，使算法可以处理。

```python
from sklearn.preprocessing import LabelEncoder OneHotEncoder
# 数字型编码
encoder = LabelEncoder()
df['Sex']  = encoder.fit_transform(df['Sex'])
# 独热向量编码， 接收二维数组
encoder2 = OneHotEncoder()
df['Sex']  = encoder2.fit_transform(df['Sex'].reshape(-1,1))
```


<a id="orgcae925c"></a>

## 从训练集中分离出测试集

两种方法：

-   固定比例分离

```python
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
```

-   交叉验证（用以比较模型）

```example
model_selection.cross_val_score()
```


<a id="org357242f"></a>

## 分析数据，找到关联性

```python
df[['Sex', 'Survived']].groupby('Sex',as_index=False).mean() # 特定特征与标签的关系
# 图示某一个特征与标签的关系
plt.hist(x = [df[df['Survived']==1]['Fare'], df[df['Survived']==0]['Fare']],
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()
# 图示某两个特征与标签的关系
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')
# Pearson关联
df.corr()
```

Pearson关联的绘图函数：

```python
#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    _ = sns.heatmap(
        df.corr(),
        cmap = colormap,
        square=True,
        cbar_kws={'shrink':.9 },
        ax=ax,
        annot=True,
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data1)
```


<a id="org3e98444"></a>

## 模型化数据集


<a id="orgdba7bcd"></a>

### 背景知识

机器学习算法可以分为四个部分：

-   分类
-   回归
-   聚类
-   降维

机器学习知识：

-   [Sklearn Estimator Overview](https://scikit-learn.org/stable/user_guide.html)
-   [Sklearn Estimator Detail](https://scikit-learn.org/stable/modules/classes.html)
-   [Choosing Estimator Mind Map](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

![img](images/sklearn_mindmap.png)

-   [Choosing Estimator Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)

![img](images/cheatsheet.png)

解决有监督的分类问题的算法：

-   Ensemble Methods
-   Generalized Linear Models (GLM)
-   Naive Bayes
-   Nearest Neighbors
-   Support Vector Machines (SVM)
-   Decision Trees
-   Discriminant Analysis