- [注意](#org2ae5809)
  - [第四章 训练模型](#org75775e5)
    - [4.1 (纯)线性回归 Linear Regression](#org3260732)
      - [4.1.1 闭式解-标准方程（normal equation）](#org70b4a4f)
      - [4.1.2 梯度下降(迭代优化)](#org31d8b0c)
      - [4.1.3 标准方程与梯度下降对比](#org2ec0c54)
    - [4.2 多项式回归 Polynomial Regression](#orgb6683d7)
      - [4.2.1 训练集增广](#org06814a8)
      - [4.2.2 学习曲线](#org69232dd)
      - [4.2.3 偏差/方差权衡](#org1457685)
    - [4.3 正则线性模型（线性模型的正则化）](#org3839278)
      - [4.3.1 岭回归 Ridge Regression](#orgaf04721)
      - [4.3.2 套索回归 Lasso Regression](#orgb929440)
      - [4.3.3 弹性网络 Elastic Net](#org1ce4e9f)
      - [4.3.4 如何在线性回归和以上三种回归之中选择呢？](#orge6dc664)
      - [4.4.4 早期停止法](#org7d91570)
    - [4.4 逻辑回归 Logistic Regression](#org3ed2e64)
      - [决策边界](#org38e07a7)
      - [逻辑回归的正则化](#org98ad3ab)
    - [4.5 多元逻辑回归 Softmax Regression](#org6b9de2f)


<a id="org2ae5809"></a>

# 注意

-   对收入分层抽样，不能分太多层
-   分层方法：除以1.5，向上取整；然后合并大于5的分类
-   地理数据可视化，用其他相关属性作为颜色，和散点大小
-   寻找与标签相关性高的属性，用df.corr()['labels']
-   进一步考察高相关性属性的数据模式，并删除可能的错误数据
-   尝试不同的属性组合，以找到高相关性特征
-   将预测器与标签分离，因为可能不一定对它们使用相同的转换方式
-   特征缩放（归一化、标准化），即同比缩放所有属性
-   评估训练得的模型，对训练集求RMSE或MAE
-   误差较大则拟合不足，可以
-   误差过小？则用验证集来验证得到的模型，以检查是否过拟合
-   交叉验证，可以sklearn的K-fold功能
-   如果在验证集上得到的误差大则说明确实有过拟合，需要更换模型
-   尝试多个模型以找到2-5个有效的模型，别花太多时间去调整超参数
-   保存每个尝试过的模型，用pickel或sklearn的joblib
-   训练集分数明显低于验证集分数，则过度拟合
-   注意：目标值一般不进行绽放，并且只对训练集缩放


<a id="org75775e5"></a>

# 第四章 训练模型


<a id="org3260732"></a>

## 4.1 (纯)线性回归 Linear Regression

用以描述线性化数据集，模型或假设（hypothesis）是特征（x）的线性函数,或者写成向量形式，令x<sub>0</sub> = 1:

![img](images/linear_hypothsis.png)

上面的表达式也称之为回归方程（regression equation），\theta为回归系数。 成本函数，MSE函数：

![img](images/MSE.png)


<a id="org70b4a4f"></a>

### 4.1.1 闭式解-标准方程（normal equation）

即直接通过解析表达式计算得到参数向量&theta;:

![img](images/normal_equation.png)

可以使用Numpy的线性代数模块np.linalg中的inv()函数来求矩阵逆，用dot()方法计算内积。 特征数量大时标准方程计算极其缓慢，此时可以用迭代优化法。

    注意：书中有误，Scikit-Learn的LinearRegression类并不是标准方程的实现，而是基于X的SVD分解。其时间复杂度为O(n^2)，在m<n或特征线性相关时依然可以工作（标准方程不行，因为不满秩）。
    LinearRegression类不需要对特征进行标度。

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # 基于scipy.linalg.lstsq()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_ # 偏置\theta_0与权重\theta_i
lin_reg.predict(X_new) # 预测
# 可能直接调用lstsq()，意为最小平方
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
```


<a id="org31d8b0c"></a>

### 4.1.2 梯度下降(迭代优化)

从随机值开始，每一步降低成本函数，直到成本函数最小值。每一步的步长取决于超参数: *学习率* *&eta;* ( *learning rate* ).

注意：

1.  线性回归模型的MSE是凸函数，没有局部最小，只一个全局最小。
2.  应用梯度下降时要保证所有特征数值大小比例差不多，即要先进行特征缩放！
3.  特征缩放主要有两种方式：standerization和normalization，见第二章，68页。
4.  可以使用sklearn的StandardScaler类。
5.  学习率的选取很关键，可以限制迭代次数进行网格搜索。

1.  4.1.2.1 批量梯度下降

    在计算梯度下降的每一步时，都基于整个训练集。训练集庞大时很耗时，但随特征数增大时，算法表现良好。

2.  4.1.2.2 随机梯度下降

    在计算梯度下降的每一步时，只随机地使用一个训练集实例。训练集庞大时很耗时，但随特征数增大时，算法表现良好。

    -   当成本函数有局部最小时，可以跳出局部最小，找到全局最小
    -   设定 *学习计划* ，开始时大步长，最后小步长（模拟退火）
    -   乱序训练集使一个接一个地使用实例，反而会导致收敛更慢！

    ```python
    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42)
    sgd_reg.fit(X, y.ravel())
    sgd_reg.intercept_, sgd_reg.coef_
    ```

3.  4.1.2.3 小批量梯度下降

    在计算梯度下降的每一步时，只随机地使用一个小的实例集。主要优势在于可以用GPU加速计算。


<a id="org2ec0c54"></a>

### 4.1.3 标准方程与梯度下降对比

| 梯度下降（Gradient descending） | 标准方程（Normal equation）      |
| 需要选择适当的学习率&eta; | 不需要学习率&eta;                |
| 需要多次迭代              | 直接解析求解                     |
| 在特征很多时仍工作很好    | 复杂度O(n<sup>3</sup>)，特征矩阵维度大时不宜考虑 |
| 能应用在更复杂的算法中（如逻辑回归） | 需要矩阵可逆（满秩）             |


<a id="orgb6683d7"></a>

## 4.2 多项式回归 Polynomial Regression

也称为多元线性回归，所以也属于线性回归，即使用以拟合非线性数据集。从参数\theta的角度看，这个模型将线性回归特征的高次幂项作为新的特征，并将它们线性组合起来，所以依然属于线性模型。


<a id="org06814a8"></a>

### 4.2.1 训练集增广

将原特征的次幂项作为新的特征加入训练集，在这个拓展过的特征集上训练线性模型。可以使用sklearn的PolynomialFeatures类来进行：

```python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
```

    注意：
    1. 高次幂项也包括特征的交叉项
    2. 作用PolynomialFeatures类要小心特征数量爆炸！


<a id="org69232dd"></a>

### 4.2.2 学习曲线

在使用模型时要经常判断：模型是否过度拟合或者拟合不足？

-   一种是第二章中学习的，使用交叉验证来评估模型的泛化性能。如果在训练集上表现比交叉验证的泛化表现好很多，则是过度拟合。如果两者表现都不佳，则拟合不足。
-   还有一种，即观察学习曲线。

曲线绘制的是模型在训练集和验证集上，关于训练集大小的性能函数。要绘制这个函数，要在不同大小的训练集上多次训练模型。

**判断标准** ：

-   拟合不足：两线均到达高地，十分接近，且相当高。
-   过度拟合：训练集误差远小于一般标准，且两条线之间有一定差距。

**改进方法** :

-   拟合不足：增加模型复杂程度
-   过度拟合：提供更多数据，或约束模型（正则化）


<a id="org1457685"></a>

### 4.2.3 偏差/方差权衡

增加模型复杂度会显著减少模型的偏差，增加拟合的方差;相反，降低模型复杂度会显著提升模型的偏差，降低拟合的方差。


<a id="org3839278"></a>

## 4.3 正则线性模型（线性模型的正则化）

对多项式模型来说，正则化的简单方法是降低多项式除数;对线性模型来说，正则化通常通过约束模型的权重来实现，比如有如下三种不同的实现方法：岭回归、套索回归、弹性网络。


<a id="orgaf04721"></a>

### 4.3.1 岭回归 Ridge Regression

也叫吉洪诺夫正则化，在成本函数中添加一个正则项 &alpha;/2 &sum;<sub>i=1</sub><sup>n</sup> &theta;<sub>i</sub><sup>2</sup>。

    注意：正则化只能在训练时添加到成本函数，完成训练后要用未经正则化的性能指标来评估模型性能。

岭回归的成本函数：

<div class="org-center">
J(&theta;) = MSE(&theta;) + &alpha;/2\*&sum;<sub>i=1</sub><sup>n</sup> &theta;<sub>i</sub><sup>2</sup>
</div>

超参数&alpha; 控制正则化程度，&alpha;=0时回复到线性模型，&alpha; 非常大时所有权重都接近于零，结果是一条穿过数据平均值的水平线。正则项是权重向量 **&theta;** 的l<sub>2</sub>范数平方的一半。

    注意：
    1. 求和从i=1开始，对偏置项不正则化。
    2. 执行岭回归前，要对数据进行缩放（大多数正则化模型都需要）。

与线性回归相同，可以直接闭式解，也可以使用随机梯度下降。sklearn的Ridge执行闭式解法， 利用Andre-Louis Cholesdy的矩阵因式分解：

```python
from sklearn.linear_model import Ridge
# ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```

使用随机梯度下降的代码如下：

```python
sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty="l2", random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
```

其中的penalty参数为惩罚的类型。


<a id="orgb929440"></a>

### 4.3.2 套索回归 Lasso Regression

套索回归是另一种正则化方法，也叫最小绝对收缩和选择算子回归（Least Absolute Shrinkage and Selection Operator Regression），简称Lasso。它为成本函数增加的一项是权重向量的l<sub>1</sub>范数。Lasso回归的成本函数为：

<div class="org-center">
J(&theta;) = MSE(&theta;) + &alpha; &sum;<sub>i=1</sub><sup>n</sup> |&theta;<sub>i</sub>|
</div>

Lasso回归倾向于完全消除最不重要特征的权重，换句话说，它会自动执行特征选择并输出一个稀疏模型（即只有少量特征的权重非零）。sklearn的Lasso类 ~~实现的是什么算法？~~

```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
```

与岭回归一样，也可以使用随机梯度下降，代码如下：

```python
sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty="l1", random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
```


<a id="org1ce4e9f"></a>

### 4.3.3 弹性网络 Elastic Net

弹性网络是岭回归和Lasso回归的中间地带，其正则项是它们正则项的混合，比例由r来控制。r=0时相当于岭回归，r=1时相当于Lasso回归。其成本函数为：

<div class="org-center">
J(&theta;) = MSE(&theta;) + r&alpha; &sum;<sub>i=1</sub><sup>n</sup> |&theta;<sub>i</sub>| + (1-r)&alpha;/2\*&sum;<sub>i=1</sub><sup>n</sup> &theta;<sub>i</sub><sup>2</sup>
</div>

sklearn的ElasticNet类代码如下：

```python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```

同样可以用随机梯度下降来实现弹性网络正则化，如下：

```python
sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty="elasticnet", random_state=42)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```


<a id="orge6dc664"></a>

### 4.3.4 如何在线性回归和以上三种回归之中选择呢？

通常而言，有正则化总比没有强，所以大多数时候应该避免使用纯线性回归。岭回归是个不错的默认选择，但如果你觉得实际用到的特征只有少数几个，那就应该更倾向于Lasso或弹性网络，因为它们可以对特征进行自动选择。一般而言，弹性网络优于Lasso回归，因为当特征数大于训练实例数或特征强相关时，Lasso回归可能非常不稳定。


<a id="org7d91570"></a>

### 4.4.4 早期停止法

对于梯度下降等迭代算法，还有一个正则化方法，就是在验证误差达到最小误差时停止训练。（可以先观察是否真正达到最小误差）

```python
from sklearn.base import clone
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None,
                       learning_rate="constant", eta0=0.0005, random_state=42)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
```


<a id="org3ed2e64"></a>

## 4.4 逻辑回归 Logistic Regression

一些回归算法也被用于分类任务，反之亦然。逻辑回归依然是线性模型。 逻辑回归，也叫罗吉思回归，被广泛用于估算一个实例属于某个特定类别的概率。如果预概率测超过50%，则判定为正类，反之则为负类。这样它就成一个二元分类器。 与线性回归不同的是，它用 **&theta;<sup>T</sup>&sdot; X** 的sigmoid函数值作为概率值，而不是 **&theta;<sup>T</sup>&sdot; X** 本身：

![img](images/logistic.png)

&sigma;(t)是sigmoid函数：

![img](images/sigmoid.png)

成本函数为log损失函数：

![img](images/cost_log.png)

这个函数没有闭式解，只能迭代优化，而且它是个凸函数。可以用随机梯度下降等优化算法求解。 如下：

```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="liblinear", random_state=42)
log_reg.fit(X, y)
```


<a id="org38e07a7"></a>

### 决策边界

决策边界，顾名思义，就是用来划清界限的边界，边界的形态可以不定，可以是点，可以是线， 也可以是平面。Andrew Ng 在公开课中强调：“决策边界是预测函数h<sub>&theta;</sub>(x)的属性， 而不是训练集属性”，这是因为能作出“划清”类间界限的只有h<sub>&theta;</sub>(x)，而训练集只是用来 训练和调节参数的。

决策边界由h<sub>&theta;</sub>(x) = &theta;<sup>T</sup> &sdot; X = 0定义，所以如果h<sub>&theta;</sub>(x) 函数是线性的，那么决策边界就是线性的;如果h<sub>&theta;</sub>(x)是非线性的，那么决策边界就是非 线性的。

    注意： 与上述多项式回归同理，虽然决策边界是非线性的，但是模型依然是线性的。


<a id="org98ad3ab"></a>

### 逻辑回归的正则化

与其他线性模型一样，逻辑回归也可以用“l<sub>1</sub>”, “l<sub>2</sub>”或“elasticnet”惩罚函数来正则化， 默认是l<sub>2</sub>函数。sklearn的LogisticRegression类中控制正则化程度的超参为C， 是&alpha; 的逆反，（其他线性模型为&alpha; ），C越 ~~大~~ 小，正则化程度越大。


<a id="org6b9de2f"></a>

## 4.5 多元逻辑回归 Softmax Regression

对于多分类问题，如前所述，可以采用OvA策略，也可采用OvO策略。OvA指为每个类别分别训练一 个二分类器，用以识别是否是该类别，对于特定实例取最近的类别为预测类别。即将多分类转化成 多次二分类问题。OvO策略指任何两个类别训练一个二分类器，如MNIST中，要训练C<sub>10</sub><sup>2</sup>=45 个二分类器。识别时对一个实例运行C<sub>10</sub><sup>2</sup>个二分类器，最后以获胜次数多的类别作用预测 结果。OvO的优点在于，训练时只需要对部分训练数据进行（只需要在需要区分的两个类别的训练集上 进行）。

    注意：只有对于在大数据集上表现糟糕的算法（SVM），OvO是优先的选择;对于大多数二元分类器来说，OvA策略更好。

Softmax回归是逻辑回归的推广，可以直接支持多类别，不需要训练并组合多个二元分类器。 对于一个特定实例 **x**, Softmax 回归会计算出每个类别k的分数s<sub>k</sub>(**x**), 然后应用 softmax函数（也叫归一化指数），估算每个类别的概率。softmax分数：

![img](images/softmax.png)

每个类别都有自己的权重向量 **&theta;<sub>k</sub>**, 所有这些向量通常作为行，存贮在参数矩阵 &Theta; 中。

有了类别分数后，实例 **x** 属于类别k的概率被定义为：

![img](images/softmax2.png)

预测类别\hat{y} 是概率\hat{p}<sub>k</sub> 最大的类别k：

![img](images/softmax-pred.png)

    注意： Softmax回归器每次只能预测一个类别，也就是说它是多类别，但不是多输出。所以仅适用于互斥的类别。

我们已经知道了模型怎么估算概率，并做出预测，那怎么训练呢？需要最小化的成本函数（交叉熵）：

![img](images/cross-entropy.png)

它来源于信息理论，描述的是多类别预测的准确性。两个离散概率分布p和q之间的交叉熵定义为：

![img](images/cross2.png)

对于这个成本函数可以作用随机梯度下降或其他优化算法找到最优解参数矩阵&Theta; （每个类别 的权重向量 **&theta;<sub>k</sub>** ）。在sklearn中，当对多个类别进行训练时 LogisticRegression会默认选择OvA策略。将参数multi<sub>class设置为</sub>"multinomial" 可以将其切换成Softmax回归。还要指定一个支持Softmax回归的求解器。默认使用l<sub>2</sub>正则化， 用超参C控制。代码如下：

```python
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X, y)
y_proba = softmax_reg.predict_proba(X_new) # 预测特定实例的每个类别概率
y_predict = softmax_reg.predict(X_new) # 预测特定实例的类别
```

-   此时的决策边界如何理解？

![img](images/decision_boundary.png)