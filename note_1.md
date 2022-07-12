# task_1

## chapter_1

### 机器学习的基本术语

data set(数据集): 对事件或者对象的描述的一组记录的集合

sample/instance(样本/示例)：对于一个事件或对象的描述

feature/attribute(特征/属性)：对于一个事件或对象的某一方面的表现或者性质的反映

attribute value(属性值)：feature的具体取值

attribute space/sample space(属性空间/样本空间)：以feature作为维度，将某一事件或者对象对应到attribute space的一个具体坐标

feature vector(特征向量)：一个 sample，即一个特征向量对应属性空间中的一个点

dimensionality(维数)：sample中feature的个数

training/learning(训练/学习)：从数据中学到模型的过程

training data(训练数据)：训练过程中使用到的数据

training sample(训练样本)：训练过程中使用到的样本

training set(训练集)：training sample 的集合

learner(学习器)：即模型

prediction(预测)：对于训练样本的标记(label)（即训练样本的结果信息）

example(样例)：带有label的instance

label space(标记空间)：label的集合

对于学习任务的分类：

- **classification**(分类)：预测的结果是离散值
  
- **regression**(回归)： 预测的结果是连续值
  
- **clustering**(聚类)：将训练集中的sample划分为若干组，每组具有相同的一些事先不知道的特性，此类任务中，一般无label

根据训练数据是否有标记信息，将学习任务划分为：

- supervised learning(监督学习)：classification and regression
  
- unsupervised learning(无监督学习)：clustering

generalization(泛化)能力：学习得到的模型对新样本的适用能力

## chapter_2 模型评估与选择

### 误差与过拟合

error rate(错误率)：分类错误的样本占样本总数的比例，错误率$E = a/m$，$a$:分类错误的样本数，$m$:样本总数

accuracy(精度)：精度$ = 1 = E$

error(误差)：学习器的实际预测输出与样本真实输出之间的差异

training error/empricial error(训练误差/经验误差)：学习器在训练集上的误差

generalization error(泛化误差)：在新样本上的误差

overfitting(过拟合)：模型在训练集上表现好，但在测试集上表现不好，泛化性能差(将训练样本自身的一些特点作为了所有潜在样本的一般性质)，**过拟合无法避免，只能缓解**

underfitting(欠拟合)：学习器对于训练样本的一般性质尚未学好

### 评估方法

由于训练误差无法说明学习器的性能（存在过拟合），而泛化误差又无法测量得到，所以需要有一个测试集(testing set)来测试学习器对新样本的判别能力，将在测试集上得到的测试误差(testing error)作为泛化误差的近似。注意：**测试集与训练集尽可能的互斥**

### 划分测试集与训练集的方法

- 留出法(hold out)

直接将数据集$D$划分为两个互斥的集合，一个集合作为训练集$S$，另一个集合作为测试集$T$。为保证数据分布的一致性，一般采用分层采样。为减少不同划分方式对模型评估结果的影响，一般采用若干次随机划分，重复实验评估后取平均值作为留出法的评估结果

- 交叉验证法(cross validation)

将数据集$D$划分为k个大小相似的互斥子集，同样采用分层采样的方法得到。每次用$k - 1$个子集作为训练集，剩下的$1$个子集作为测试集，然后依次取$k$个不同的子集作为测试集，这样进行$k$组训练和测试，最终的评估结果为这$k$组测试的平均值

在交叉验证法中，对样本进行不同的划分得到的评估结果会有差别，一般采用随机适用不同的划分方法重复$p$次，最终将这$p$次评估结果的均值作为$p$**次**$k$**折叉验证**的结果

在留一法中样本划分不会带来影响，留一法，即每个子集包含一个样本

- 自助法

在包含$m$个样本的数据集$D$中进行如下采样得到数据集$D^{'}$：随机在数据集$D$中选取一个样本，将其复制到数据集$D^{'}$中，然后放回，重复m次上述操作。

然后将数据集$D^{'}$作为训练集，数据集$D\\D^{'}$作为测试集，由概率论和极限得数据集$D\\D^{'}$中得样本数大约占数据集$D$中样本数的$36.8\%$，且数据集$D\\D^{'}$与数据集$D$互斥

分层采样(stratified sampling)：在采样过程中保留类别比例

### 性能度量(performance measure)

性能度量：用来衡量模型泛化能力

对于样例集$D$

- mean squared error(均方误差)：$E(f; D) = \frac{1}{m} \sum^{m}_{i = 1}(f(x_{i})-y_{i})^{2}$，回归中常用

-错误率：$E(f; D) = \frac{1}{m} \sum^{m}_{i = 1}(f(x_{i})-y_{i})^{2}$