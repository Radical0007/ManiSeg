# 方法与算法

[返回首页](../README.md) · [实验](experiments.md) · [实验流程](reproducibility.md)

## 1 输入与实例级特征

设共有 $N$ 个目标实例、$C$ 个类别，实例特征维数为 $d$。输入包括原始图像、真实目标框 $b_i$、类别标签及 YOLO 网络。框架先利用复合损失 $L_{\mathrm{total}}$ 进行联合训练与流形对齐，得到 $f_{\theta'}$。

对每个目标框裁剪并调整目标区域，将其输入网络，在目标层提取局部特征图 $H_i$，再进行全局平均池化，形成实例特征 $x_i$。同时取得模型的预测概率向量 $p_i$：

$$
X=[x_1,\ldots,x_N]^\top\in\mathbb{R}^{N\times d},\qquad
P=[p_1,\ldots,p_N]^\top\in\mathbb{R}^{N\times C},\qquad
p_i\in\Delta^{C-1}.
$$

$X$ 表示实例级特征，$P$ 表示对应的预测概率。实验提取 FPN 的 P5 层 C3K2 模块输出作为高维语义表示。

## 2 相关概念

### 2.1 普通粒球

用超球体 $B_j=(c_j,r_j)$ 表示一组样本。若成员索引集合为 $S_j$，普通几何定义为：

$$
c_j=\frac{1}{|S_j|}\sum_{i\in S_j}x_i,\qquad
r_j^{\mathrm{max}}=\max_{i\in S_j}\|x_i-c_j\|_2.
$$

本文方法在此基础上采用分位数覆盖半径，并通过语义熵决定是否继续分裂。

### 2.2 语义熵

利用高斯核与类别标签，可以构造局部语义概率：

$$
p_k^{\mathrm{kernel}}(B)=
\frac{\sum_{i\in S_B,\,y_i=k}\exp\!\left(-\frac{\|x_i-c_B\|_2^2}{2\sigma^2}\right)}
{\sum_{i\in S_B}\exp\!\left(-\frac{\|x_i-c_B\|_2^2}{2\sigma^2}\right)},
\qquad
H(B)=-\sum_{k=1}^{C}p_k(B)\log_2 p_k(B).
$$

高斯核通过距离加权描述局部语义分布；粒球构建阶段使用样本预测概率的均值计算语义熵。

### 2.3 传统三支决策

传统概率三支决策在 $0\leq\beta<\alpha\leq1$ 时定义接受、延迟和拒绝区域：

$$
\begin{aligned}
\mathrm{POS}(X)&=\{B:P(X\mid B)\geq\alpha\},\\
\mathrm{BND}(X)&=\{B:\beta<P(X\mid B)<\alpha\},\\
\mathrm{NEG}(X)&=\{B:P(X\mid B)\leq\beta\}.
\end{aligned}
$$

MC3WD-SEGBC 通过粒球覆盖与异类碰撞关系构建三支空间。

## 3 语义熵驱动的粒球构建

### 3.1 鲁棒覆盖半径

对节点集 $S_j$ 计算质心和各成员的欧氏距离，并使用分位数截断：

$$
c_j=\frac{1}{|S_j|}\sum_{i\in S_j}x_i,\qquad d_i=\|x_i-c_j\|_2,
$$

$$
r_j=\min\!\left(\operatorname{Percentile}(\{d_i:i\in S_j\},q),\ \max_{i\in S_j}d_i\right).
$$

实验采用 $q=95$。逻辑成员集 $S_j$ 用于计算均值和语义分布，实际覆盖由 $\|x-c_j\|_2\leq r_j$ 决定。样本的三支区域归属由全部终端粒球的覆盖关系共同确定。

### 3.2 粒球语义分布与熵

$$
\bar p_{B_j}=\frac{1}{|S_j|}\sum_{i\in S_j}p_i,
$$

$$
H(B_j)=-\sum_{k=1}^{C}\bar p_{B_j,k}\log_2(\bar p_{B_j,k}+\varepsilon),\qquad
L(B_j)=\operatorname*{arg\,max}_k\bar p_{B_j,k}.
$$

$\varepsilon$ 为数值稳定项。低熵表示球内预测信念较一致；高熵提示语义混杂。主导语义 $L(B_j)$ 用于后续碰撞检测。

### 3.3 分裂准则

当 $H(B_j)>H_{\mathrm{th}}$ 且当前深度未达到上限时，尝试局部欧氏二元分裂，得到两个子节点 $S_1,S_2$。信息增益为：

$$
\Delta H=H(B_j)-\left[\frac{|S_1|}{|S_j|}H(B_1)+\frac{|S_2|}{|S_j|}H(B_2)\right].
$$

当 $\Delta H\geq\Delta H_{\min}$、两个子节点样本数均不少于 $N_{\min}$，且满足深度约束时接受分裂；否则保留当前节点。信息增益、子节点规模与递归深度共同控制粒化过程。

### 算法 2 语义熵引导的粒球构建

```text
输入：特征 X，预测概率 P，粒化参数
输出：终端粒球集合 B，各球主导语义 L

初始化待处理队列为全部样本，B 为空
确定 H_th、最小信息增益、最小子节点数、分位数 q 和深度上限
while 存在待处理节点 S:
    计算质心 c、分位数半径 r、平均预测分布 p_bar 和语义熵 H
    if H > H_th 且深度未超限:
        执行局部欧氏二元分裂，得到 S1 和 S2
        计算两个子节点的熵和信息增益
        if 信息增益达标且两个子节点样本数均达标:
            将两个子节点加入待处理队列
            continue
    将当前粒球加入 B
对每个终端球计算 L(B) = argmax(p_bar)
返回 B 与 L
```

粒球构建与三支划分的参数见[实验参数表](experiments.md#粒球与三支划分参数)。

## 4 异类流形碰撞与三支空间

### 4.1 语义支撑区

$$
\mathcal B_k=\{B_j\in\mathcal B:L(B_j)=k\},\qquad
\mathcal S_k=\bigcup_{B_j\in\mathcal B_k}B_j,\qquad
\Omega=\bigcup_{k=1}^{C}\mathcal S_k.
$$

### 4.2 碰撞对

对于主导语义不同的两个粒球，使用带松弛因子的碰撞判据：

$$
L(B_u)\ne L(B_v),\qquad
\|c_u-c_v\|_2<\gamma(r_u+r_v).
$$

满足条件的球对记入 $\mathcal C_{\mathrm{pairs}}$。实验设置 $\gamma=1.1$，通过松弛因子描述相邻异类粒球的碰撞关系。样本同时处于异类碰撞球对的实际覆盖范围内时，归入边界域。

### 4.3 样本覆盖与赋域

对每个样本 $x_n$，定义实际覆盖它的粒球索引集：

$$
A_n=\{j:\|x_n-c_j\|_2\leq r_j\}.
$$

| 区域 | 判定规则 | 含义 |
| --- | --- | --- |
| NEG | $\lvert A_n\rvert=0$ | 未被任何终端粒球覆盖 |
| BND | 覆盖球中存在异类碰撞对，且该样本同时位于两球内 | 异类覆盖冲突区域 |
| POS | 其余已覆盖样本 | 单球覆盖、同类多球覆盖或未满足 BND 的已覆盖样本 |

边界域的集合表达为：

$$
x\in\mathrm{BND}\iff
\exists u\ne v:\ x\in B_u\cap B_v,\quad
L(B_u)\ne L(B_v),\quad (u,v)\in\mathcal C_{\mathrm{pairs}}.
$$

上述几何等价关系将边界域成员资格与异类粒球的共同覆盖位置相联系：属于 BND 的样本存在相应异类覆盖对，满足该覆盖条件的样本也归入 BND。三支划分由此将语义边界映射为高维空间中的几何重叠区域。

### 算法 3 碰撞与三支空间划分

```text
输入：X，终端粒球 B，主导语义 L，松弛系数 gamma
输出：碰撞对 C_pairs，以及 POS、BND、NEG 三个样本集合

初始化碰撞对和三个区域为空
遍历不同主导语义的粒球对 (u, v):
    若中心距离 < gamma * (r_u + r_v)，记录该碰撞对
遍历每个样本 x:
    A = 实际覆盖 x 的粒球索引集合
    if A 为空:
        将 x 放入 NEG
    else if A 中存在主导语义不同且属于 C_pairs 的球对:
        将 x 放入 BND
    else:
        将 x 放入 POS
返回碰撞对和三支区域
```

## 5 视觉–语义保真度

设低维投影为 $Y\in\mathbb R^{N\times d_{\mathrm{low}}}$，通常 $d_{\mathrm{low}}=2$。高维和低维空间分别计算完整邻域秩矩阵 $R_{\mathrm{high}}$ 与 $R_{\mathrm{low}}$，自身秩为 0。

### 5.1 几何拓扑保真度 GTF

令 $N_k^X(i)$ 与 $N_k^Y(i)$ 分别为高维和低维空间中样本 $i$ 的前 $k$ 个邻居，不包含自身：

$$
U_i(k)=N_k^Y(i)\setminus N_k^X(i),\qquad
V_i(k)=N_k^X(i)\setminus N_k^Y(i).
$$

$U_i(k)$ 为投影后错误挤入的邻居，$V_i(k)$ 为投影后错误挤出的邻居。可信度与连续性为：

$$
C_{\mathrm{norm}}=\frac{2}{Nk(2N-3k-1)},
$$

$$
T(k)=\max\!\left(0,1-C_{\mathrm{norm}}\sum_{i=1}^{N}\sum_{j\in U_i(k)}[R_{\mathrm{high}}[i,j]-k]\right),
$$

$$
C(k)=\max\!\left(0,1-C_{\mathrm{norm}}\sum_{i=1}^{N}\sum_{j\in V_i(k)}[R_{\mathrm{low}}[i,j]-k]\right),
$$

$$
\operatorname{GTF}(k)=\frac{2T(k)C(k)}{T(k)+C(k)+\varepsilon}.
$$

GTF 使用调和平均共同约束错误邻域引入与原有邻域丢失，综合刻画结构坍塌与拓扑撕裂。

### 5.2 语义邻域一致性 SNH

在低维邻域内计算预测概率向量的平均余弦相似度：

$$
\operatorname{SNH}(k)=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{k}\sum_{j\in N_k^Y(i)}
\frac{p_i\cdot p_j}{\|p_i\|_2\|p_j\|_2}.
$$

SNH 以预测概率向量的余弦相似度描述低维邻域内语义信念的一致性。

### 5.3 复合指标与多尺度积分

$w_{\mathrm{sem}}$ 控制 GTF 在复合指标中的权重：

$$
\operatorname{VSF}(k)=w_{\mathrm{sem}}\operatorname{GTF}(k)+(1-w_{\mathrm{sem}})\operatorname{SNH}(k),
\qquad w_{\mathrm{sem}}\in[0,1].
$$

对邻域尺度集合 $\mathcal K$，采用梯形积分计算归一化 AUC：

$$
\operatorname{VSF}_{\mathrm{AUC}}=
\frac{\int_{\min(\mathcal K)}^{\max(\mathcal K)}\operatorname{VSF}(k)\,\mathrm dk}
{\max(\mathcal K)-\min(\mathcal K)+\varepsilon}.
$$

多尺度积分汇总不同邻域规模下的 VSF，形成整体投影质量评价。

## 6 整体算法

### 算法 1 MC3WD-SEGBC 整体流程

```text
输入：带标注框的数据集 D，YOLO 网络 f_theta，损失 L_total，目标层 l_conv
输出：训练与对齐后的网络 f_theta_prime，以及 VSF

1. 利用 L_total 进行联合训练和流形对齐
2. 遍历所有标注实例，裁剪并调整目标区域
3. 从目标层提取特征图，全局平均池化，得到 X
4. 收集每个实例的预测概率，得到 P
5. 调用算法 2，得到终端粒球及主导语义
6. 调用算法 3，得到异类碰撞对及三支区域
7. 将 X 降维，得到 Y
8. 根据 X、Y、P 计算 GTF、SNH 和 VSF
9. 输出网络与评估结果，展示类别、三支区域和碰撞点
```

粒球与碰撞分析刻画高维特征空间的语义边界，VSF 评价高维至低维的结构保持与语义一致性。完整评估步骤见[实验流程](reproducibility.md)。
