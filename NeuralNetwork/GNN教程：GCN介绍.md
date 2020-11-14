## 引言

这是我们介绍图神经网络的第一篇文章，取自Kipf et al. 2017，文章中提出的模型叫**Graph Convolutional Network**(GCN)，个人认为可以看作是图神经网络的“开山之作”，因为GCN利用了近似的技巧推导出了一个简单而高效的模型，使得图像处理中的卷积操作能够简单得被用到图结构数据处理中来，后面各种图神经网络层出不穷，或多或少都受到这篇文章的启发。

## 1. 问题定义

考虑图（例如引文网络）中节点（例如文档）的分类问题，假设该图中只有一小部分节点标签（label）是已知的，我们的分类任务是想通过这部分已知标签的节点和图的结构来推断另一部分未知标签的节点的标签。这类问题可以划分到基于图结构数据的半监督学习问题中。半监督学习（semi-supervised learning）是有监督学习的一个分支，主要研究的是如何利用少量的有标签数据学习大量无标签数据的标签。

为了对节点进行分类，首先我们可以利用节点自身的特征信息，除此之外，我们还可以利用图结构信息，因此一个典型的图半监督学习问题可以对两个损失函数一起优化： 

$$ \mathcal{L}=\mathcal{L}*{labeled}+\lambda \mathcal{L}*{\mathrm{reg}} \tag{1} $$

其中，$\mathcal{L}*{labeled}$基于基于标签数据的损失函数（loss function），$\mathcal{L}*{reg}$ 代表基于图结构信息的损失函数，$\lambda$ 是调节这两种损失函数相对重要性的超参（hyperparameter）。

一般来说，基于图结构信息的损失函数可以表示成： 

$$ \mathcal{L}*{\mathrm{reg}}=\sum*{i, j} A_{i j}\left|f\left(X_{i}\right)-f\left(X_{j}\right)\right|^{2}=f(X)^{\top} \Delta f(X) \tag{2} $$

其中，$f(\cdot)$ 是类似神经网络的可微分函数， 在无向图$\mathcal{G}=(\mathcal{V}, \mathcal{E})$ 中，假设有$N$个节点$v_{i} \in \mathcal{V}$，$X$为节点特征向量构成的矩阵，其中$X_i$表示节点$v_i$的特征向量，边$\left(v_{i}, v_{j}\right) \in \mathcal{E}$， 邻接矩阵表示为$A \in \mathbb{R}^{N \times N}$（可以是二值的(bianry)，也可以是加权的(weighted)），度矩阵$D_{i i}=\sum_{j} A_{i j}$。$\Delta=D-A$表示无向图$\mathcal{G}=(\mathcal{V}, \mathcal{E})$的未正则化图拉普拉斯算子。这样的损失函数希望对相邻节点的特征向量做限制，希望它们能尽量相近。

显然，这样的学习策略基于图中的相邻节点标签可能相同的假设（因为损失函数要求相邻节点的特征向量尽量相似，如何他们的标签不相似的话，那么不能学习到一个从特征向量到标签的有效映射）。然而，这个假设可能会限制模型的能力，因为图的边在语义上不一定代表所连接节点相似。还有一种可能是图中有大量的噪声边。

因此，在这个工作中，作者不再显示的定义图结构信息的损失函数 $\mathcal{L}*{reg}$ , 而是使用神经网络模型$f(X, A)$直接对图结构进行编码，训练所有带标签的结点$\mathcal{L}*{0}$，来避免损失函数中的正则化项$\mathcal{L}_{reg}$。

这篇文章的主要贡献是为图半监督分类任务设计了一个简单并且效果好的神经网络模型，且这个模型由**谱图卷积**(spectral graph convolution)的一阶近似推导而来，具有理论基础。

## 2. 图上的快速卷积近似

这一节介绍如何从谱图卷积推导出GCN的逐层更新模型，涉及到一些谱图理论的知识，可以安全的跳过这一节，后面我们会为谱图卷积专门出一个专栏的文章，将详细讨论它们。

这一节主要介绍图卷积网络GCN逐层更新(propagation)的理论推导。多层图卷积网络(Graph Convolutional Network, GCN)的逐层传播公式：
$$
H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A} \tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)
$$
其中，$\tilde{A}=A + I_N$表示无向图$\mathcal{G}$加上了自连接后的邻接矩阵，$$\tilde{D}*{ii}=\sum_j\tilde{A}*{ij}$$是每个节点的度，$W^{(l)}$是$l$层可学习的参数。$H^{(l)}\in\mathbb{R}^{N\times D}$是第$l$层激活后的节点Embedding值，输入层$H^{(0)}=X$. 下面我们会重点介绍这个逐层的更新规则可以由**图上的局部谱滤波(Localized spectral filters)的一阶近似**推导而来。(另一种解释是基于Weisfeiler-Lehman算法的，见论文的附录A，之后也会有一篇文章详细讨论这个算法)

### 谱图卷积

谱图卷积是这样定义的：信号$x\in\mathbb{R}^N$(对每个节点来说是一个标量，可以看成是节点的标签)通过一个傅里叶域的滤波器$g_\theta=\text{diag}(\theta), \theta\in\mathbb{R}^N$得到的结果，即：
$$
g_{\theta} \star x=U g_{\theta} U^{\top} x
$$
左式表示信号$x$经傅里叶域滤波器$g_\theta$变换，右边是将这个变换过程用矩阵的乘法表示。这里要引入一个叫归一化的图拉普拉斯矩阵(normalized graph Laplacian)的概念，它可以表示为 
$$
L=I_{N}-D^{-\frac{1}{2}} A D^{-\frac{1}{2}}=U \Lambda U^{\top}
$$
其中$U$是$L$的特征向量矩阵，而$\Lambda$是对应的特征值矩阵(特征值在对角线上)，$U^\top x$是$x$在图上的傅里叶变换。我们能将$g_\theta$理解为$L$特征值的一个函数，即$g_\theta(\Lambda)$。

上面的等式算起来非常耗时，因为特征向量矩阵的乘法是$\mathcal{O}(N^2)$阶的。再者，$L$的特征分解在大图上也很低效。

为了克服这样的问题 [Hammond et al. (2011)](https://hal.inria.fr/inria-00541855/document)指出$g_{\theta}(\Lambda)$能够被切比雪夫多项式(Chebyshev polynomials) $T_k(x)$所近似，$K$阶多项式就能达到很好的近似效果：
$$
 g_{\theta^{\prime}}(\Lambda) \approx \sum_{k=0}^{K} \theta_{k}^{\prime} T_{k}(\tilde{\Lambda}) 
$$
其中$\tilde{\Lambda}=\frac{2}{\lambda_{\max }} \Lambda-I_{N}$， $\lambda_{\max}$表示$L$的最大特征值。

$\theta'\in\mathbb{R}^k$是切比雪夫因子，切比雪夫多项式是由$T_k(x)=2xT_{k-1}(x)-T_{k-2}(x)$递归定义的，其中$T_0(x)=1, T_1(x)=x$。

将切比雪夫近似带入到谱图卷积的公式里：
$$
g_{\theta^{\prime}} \star x \approx \sum_{k=0}^{K} \theta_{k}^{\prime} T_{k}(\tilde{L}) x
$$
其中$\tilde{L}=\frac{2}{\lambda_{\max}}L-I_N$, 这个表达式现在是K-localized的，因为我们仅用$K$阶切比雪夫多项式近似$g_\theta(\Lambda)$。即每个节点只受$K$步以内的其他节点影响，评估上式的时间复杂度是 $\mathcal{O}(\vert\mathcal{E}\vert)$ , 即线性于边的数量。[Defferrard et al.2016](https://arxiv.org/abs/1606.09375) 使用 K-localized 卷积定义图上的卷积操作。

### 逐层线性模型

现在假设我们限制$K=1$，即谱图卷积近似为一个关于$L$的线性函数。这种情况下，我们仍能通过堆叠多层来得到卷积的能力，但是这时候，我们不再受限于切比雪夫多项式参数的限制。我们期望这样的模型能够缓解当图中节点度分布差异较大时对局部结构过拟合问题，比如社交网络，引文网络，知识图谱等。另一方面，从计算的角度考虑，逐层线性模型使我们可以构建更深的模型。

在GCN模型中， 我们做了另一个近似$\lambda_{\max}\approx 2$。

我们期望神经网络的参数能够在训练过程中适应这种变化。在这些近似的条件下，我们得到：
$$
g_{\theta^{\prime}} \star x \approx \theta_{0}^{\prime} x+\theta_{1}^{\prime}\left(L-I_{N}\right) x=\theta_{0}^{\prime} x-\theta_{1}^{\prime} D^{-\frac{1}{2}} A D^{-\frac{1}{2}} x
$$
上式有两个参数 $\theta'_0$ 和 $\theta'_1$ 。这两个参数可以在整个图所有节点的计算中共享。在实践中，进一步限制参数的个数能够一定程度上避免过拟合的问题，并且减少计算量。因此我们引入$$\theta=\theta'_0=-\theta'_1$$做进一步的近似：

$$
 g_{\theta} \star x \approx \theta\left(I_{N}+D^{-\frac{1}{2}} A D^{-\frac{1}{2}}\right) x 
$$
注意$I_{N}+D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$的特征值现在被限制在了$[0, 2]$中。重复这样的操作将会导致数值不稳定、梯度弥散/爆炸等问题。为了缓解这样的问题，我们引入了这样的再正则化(renormalization)技巧：

$$
I_{N}+D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \rightarrow \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}
$$
其中$\tilde{A}=A+I_{N}, \tilde{D}{i i}=\sum{j} \tilde{A}_{i j}$。

我们把上述定义进一步泛化：输入信号$X\in \mathbb{R}^{N\times C}$, 每个输入信号有$C$个通道(channels, 即每个图节点有$C$维特征)，卷积包含$F$个滤波器(filters)或特征映射(feature maps), 如下：

$$
 Z=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X \Theta 
$$
其中 $\Theta\in \mathbb{R}^{C\times F}$ 是filters的参数矩阵, $Z\in \mathbb{R}^{N\times F}$ 是卷积之后的信号矩阵。这样的转化操作的时间复杂度为 $\mathcal{O}(\vert\mathcal{E}\vert FC)$, 因为$\tilde{A}X$能够被高效的以稀疏矩阵和稠密矩阵相乘的形式实现。

$X$是输入向量，神经网络通常有多层，我们习惯于把$X$变换后的向量记做$H$，表示节点在隐藏层的embedding, 其次，我们习惯将神经网络的参数表示为$W$而非$\Theta$，输出$Z$会变成下一层的输入，做完这些符号替换后，多层图神经网络的Embedding更新公式为 
$$
 H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A} \tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right) 
$$

## 3. 半监督学习节点分类

### 传播公式解释

上一节中，我们从谱图卷积理论中推导得到了GCN是如何逐层更新节点embedding的

$$ H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A} \tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right) $$

首先我对这个公式做一下形象的解释：**每个节点拿到邻居节点信息然后聚合到自身embedding上**。在上面的公式中$D^{-\frac{1}{2}}\tilde{A}D^{-\frac{1}{2}}$可以看成是归一化后的邻接矩阵，$H^{(l)}W^{(l)}$相当于给$l$层所有节点的embedding做了一次线性变换，左乘邻接矩阵表示对于每个节点来说，该节点的特征变为邻居节点特征相加后的结果。

这个形象的解释对理解GNN非常重要，希望大家能仔细想一下是不是懂了。

**例子**

下面，让我们用一个两层GCN的例子阐述GCN是如何对节点进行分类的，令$\hat{A}=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$。根据逐层传播公式，这个两层的GCN输出embedding计算为：

$$ Z=f(X, A)=\operatorname{softmax}\left(\hat{A} \operatorname{ReLU}\left(\hat{A} X W^{(0)}\right) W^{(1)}\right)\tag{9} $$

这里，$W^{(0)} \in \mathbb{R}^{H \times C}$是权重矩阵, 目的是对节点的输入embedding$X$做线性变换(从输入层到隐藏层的变换)。$W^{(1)} \in \mathbb{R}^{H \times F}$是另一个权重矩阵，目的是对第一层变化后的节点embeding再做一次变换(从隐藏层到输出层)。

变换结果经过softmax 激活函数后输出作为节点的分类结果。对于半监督多分类问题，我们在所有带标签的样本上评估交叉熵：

$$ \mathcal{L}=-\sum_{l \in \mathcal{Y}*{L}} \sum*{f=1}^{F} Y_{l f} \ln Z_{l f}\tag{10} $$

这里$\mathcal{Y}_L$是所有带标签节点的集合。最后通过利用梯度下降法训练神经网络权重$W^{(0)}$和$W^{(1)}$ 就可以了。

## 后话

实现时，由于GCN需要输入整个邻接矩阵$A$和特征矩阵$X$, 因此它是非常耗内存的，论文中作者做了优化，他们将$A$作为稀疏矩阵输入，然后通过实现稀疏矩阵和稠密矩阵相乘的GPU算子来加速计算，然而，即使这样，整个矩阵仍然存在这要被塞进内存和显存中的问题，当图规模变大的时候，这种方法是不可取的，在下一篇GraphSAGE的博文中，我们将会介绍如何巧妙的克服这样的问题。

### 参考文献
[1] Semi-Supervised Classification with Graph Convolutional Networks