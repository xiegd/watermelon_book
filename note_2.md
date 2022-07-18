# Chapter_2 线性模型

## 线性模型简介

机器学习的三要素：

1、模型：根据具体问题，确定假设空间

2、策略：根据评价标准，确定选取最优模型的策略（通常是会得到一个“损失函数”

3、算法：求解损失函数，确定最优模型 

线性模型的基本形式

$f(\pmb{x}) = \pmb{w}^{T} \pmb{x} + b$

示例$\pmb{x} = (x_{1}; x_{2}; ...;x_{d})$，$x_{i}$是$\pmb{x}$在第$i$个属性上的取值

$\pmb{w}= (w_{1};w_{2};...;w_{d})$，在确定参数$\pmb{w}$和$b$就可确定模型

## 一元线性回归模型

考虑输入的属性只有一个，此时数据集$D = \{(x_i, y_i)\}^{m}_{i = 1} ,\space x_i \in R$

在一元线性模型中我们试图寻找一个模型使得$f(x_i)$近似等于$y_i$

为了衡量我们学得的模型$f(x_i)$与标记$y_i$之间的误差，我们使用欧式距离，即$|f(x_i) - y_i|$，而我们最终要寻找的最优模型就是取得所有instance的欧式距离之和最小的的那个模型。

即$\min \sum^{m}_{i=1} (f(x_i) - y_i)^2$

而由一元线性回归模型的一般形式为$f(x_i) = w^{T} x_i + b$

要确定模型则需要确定参数$w$和$b$的取值，而最优模型的参数值在$\min \sum^{m}_{i=1} (f(x_i) - y_i)^2$取得

即损失函数 $(w^*, b^*) = \arg_{(w,b)} \min \sum^{m}_{i=1}(y_i - wx_i -b)^2$

### 通过极大似然估计导出损失函数

对于一批观测样本$x_1, x_2,...,x_n$，假设这批样本服从$X \sim N(\mu, \sigma^2)$,其中$\mu, \sigma$为待估计的参数

则$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$

对应的似然函数，$L(\mu, \sigma^2) = \prod^{n}_{i=1}p(x_i; \mu, \sigma^2) = \prod^{n}_{i=1}\frac{1}{\sqrt{2\pi \sigma}}\exp(-\frac{(x_i-\mu)^2}{2\sigma^2})$

两边取对数得：$\ln L(\mu, \sigma^2) = \sum^{n}_{i=1}\ln \frac{1}{\sqrt{2\pi \sigma}}\exp(-\frac{(x_i-\mu)^2}{2\sigma^2})$

对于一元线性回归模型，设其基本形式为$y = wx + b + \epsilon$，其中$\epsilon$为不受控制的误差，一般设其符合正态分布$\epsilon \sim N(0,\sigma^2)$

$\epsilon$的概率密度函数为$p(\epsilon) = \frac{1}{\sqrt{2\pi \sigma}}\exp(-\frac{\epsilon^2}{2\sigma^2})$

则综合两式得：
$p(y) = \frac{1}{\sqrt{2\pi \sigma}}\exp(-\frac{(y-(wx-b))^2}{2\sigma^2})$????

即，$y \sim N(wx+b, \sigma^2)$

则最大似然函数：$L(w, b) = \prod^{m}_{i=1}p(y_i) = \prod^{m}_{i=1}\frac{1}{\sqrt{2\pi \sigma}}\exp(-\frac{(y_i-(wx_i+b))^2}{2\sigma^2})$

两边取对数得：$\ln L(w, b) = \sum^{m}_{i=1}\ln \frac{1}{\sqrt{2\pi \sigma}}\exp(-\frac{(y_i-wx_i-b))^2}{2\sigma^2})$

化简得：$\ln L(w, b) = m\ln \frac{1}{\sqrt{2 \pi}\sigma} - \frac{1}{2 \sigma^2}\sum^{m}_{i=1}(y_i-wx_i-b)^2$

要使得$\ln(w,b)$最大，即$\sum^{m}_{i=1}(y_i-wx_i-b)^2$最小

等价于，$(w^*, b^*) = \arg_{(w,b)} \min \sum^{m}_{i=1}(y_i - wx_i -b)^2$

### 参数$w$和$b$的求解

前置知识：凸函数的定义，梯度，Hessian矩阵，根据Hessian矩阵判断凸函数,矩阵求导公式，凸充分定理

首先证明$E(w,b) = \sum^{m}_{i=1}(y_i - wx_i -b)^2$是关于$w$和$b$的凸函数，再根据凸函数求解$w$和$b$

证明：

首先解出函数对应的Hessian矩阵为：

$\nabla^2 E(w,b) = \begin{bmatrix}
  2 \sum_{i=1}^{m}x_i^2 & 2 \sum_{i=1}^{m}x_i \\
  2 \sum_{i=1}^{m}x_i^2 & 2m
\end{bmatrix}$

计算得出Hessian矩阵的顺序主子式均大于零，

一阶顺序主子式为平方项，大于等于0。二阶顺序主子式可转化为完全平方（利用$\bar x$），大于等于0。则该矩阵为半正定矩阵

则$E(w,b)$是关于$w$和$b$的凸函数，根据凸充分定理，$\nabla E(w,b) = \pmb0$的点即为最小值点。

得出：

$\frac{\partial E(w,b)}{\partial w} = 0$

$\frac{\partial E(w,b)}{\partial b} = 0$

解得：$b = \frac{1}{m}\sum_{i=1}^{m}(y_i-wx_i)$

化简为：$b = \bar y -w \bar x$

$w \sum_{i=1}^{m}x_i^2 = \sum_{i=1}^{m} y_ix_i - \sum_{i=1}^{m} bx_i$

将上面解得的$b$带入整理得到：

$w = \frac{\sum_{i=1}^{m}y_ix_i-\bar y \sum_{i=1}^m x_i}{\sum_{i=1}^mx_i^2 - \bar x \sum_{i=1}^m x_i}$

再恒等变换$\bar x \bar y$整理得到：

$w = \frac{\sum_{i=1}^{m}y_i(x_i-\bar x)}{\sum_{i=1}^mx_i^2 - \frac{1}{m}(\sum_{i=1}^m x_i)^2}$

## 多元线性回归模型

## 对数几率回归模型

矩阵微分公式




