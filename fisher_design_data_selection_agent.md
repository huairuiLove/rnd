# 基于 Fisher 实验设计的残差需求匹配数据选择智能体

> 暂定名称：**RND-Agent（Residual-Need Design Agent）**
>
> 文档性质：方法推导初稿。第 2--9 节是可验证的数学推导，第 10 节起是尚未验证的方法设计与实验协议。
>
> 定位：本文档是 `influence_prototype_cluster_mab_method.md`（IPB-MAB）的替代设计。IPB-MAB 将
> influence 边界利用、原型外扩、rank 融合与 Cluster-UCB 四个机制并列拼接；本文档证明其中前三者
> 是同一个 c-最优实验设计目标的不同近似，因而可以合并为单一打分式，第四者可降级为工程手段。
>
> **本版本为 CoMAL 适配修订版。** 相对初稿的实质改动：
> 1. 第 2 节结构前提改指 CoMAL 的 `networks.py` 单 Linear 头，并补上 bias 增广 $z=[h,1]$（初稿遗漏）；
> 2. 第 9.1 节以**精确分块对角**取代 K-FAC + rank-$r$ 截断，连带删除第 5 节的粗筛/精算两段式——
>    在单 sigmoid 头设定下这些近似是多余的，不是"更省"而是"本不需要"（式 14/14'）；
> 3. 第 12 节查询算法据此重写，并显式给出阻尼 $\delta=2\,\mathrm{wd}\,N/d$ 的推导来源与
>    `GV_eff` 的必要性（两处会静默破坏正确性的实现陷阱）；
> 4. 第 13 节 A1 已在 AAPD-54 上验证通过，并记录了"真值选择本身影响结论"这一测量教训；
> 5. 第 15 节仓库落点全部重映射到 CoMAL（初稿指向另一个仓库，无法直接使用）。

## 1. 目的与要回答的问题

IPB-MAB 的结构性缺陷不在于任一模块，而在于**相似度核不完整**：

- `S_prototype` 在标签轴上细（逐标签 $c$ 分解），在实例轴上粗（把已标注集合压成 $M_c$ 个均值）；
- 通常的梯度/特征相似度在实例轴上细，在标签轴上粗（拍平成一个标量内积）。

两者互不包含，所以只能靠 `rank()` 归一化与 $\lambda_t$ 硬缝，批内冗余无法在分数层面表达，于是再外挂
Cluster-MAB 补多样性。本文档的解法不是二选一，而是构造一个**两个轴都细**的核，并证明：

1. 在只用模型自身分布、不引入伪标签的约束下，signed 线性 influence 的期望恒为零而退化（命题 1；这是 score-function identity，**不构成"线性 influence 是错误目标"的论断**）；
2. 正确的打分是一个**二次型**，它同时是 c-最优设计的边际增益线性化（命题 2）与标签边缘化后的
   influence 二阶矩（命题 3），两条路线恒等；
3. 批内去冗余由目标函数自带的 deflation 给出（式 9），不需要 MAB；
4. prototype 在标签轴细、实例轴粗，故式 (13) 的表达力不低于它；"$M_c$ 就是截断秩"是解释性猜想而非恒等（第 8 节已按证明强度分级）；
5. 完整核可**精确因式分解**，复杂度由 $O(Cd)$ 降到 $O(C+d)$（命题 4）。

净结果：删除 $S_{\mathrm{boundary}}$、$S_{\mathrm{prototype}}$、PFD、SLCI、多原型库、$\tau_{t,c}$、
rank 融合、$\lambda_t$、Cluster-UCB 及其超参 $\{\lambda_{\min},\lambda_{\max},\gamma,\alpha,k_H,M,\eta,M_c\}$，
保留一个核、一条选择式、一条更新式，并在其上叠加一个可学习策略。

## 2. 设定与符号

本文档的全部闭式结论依赖一个结构前提：**可训练部分是冻结特征之上的单个 Linear 头**。

CoMAL 满足这一前提：`networks.py:16` 的 `BackBone_No_GCN_No_Atten.clf = nn.Linear(hidden_size, label_num)`
直接作用于 BERT 的 CLS 输出，`Encoder_No_GCN_No_Atten.encoder_init()` 在 `--freeze_bert` 下冻结前
`freeze_layer_num` 层（`scripts/aapd/main_our.sh` 已默认使用）。**若解冻整个 backbone，第 5--8 节的闭式
结论不再成立**，此时须退回 K-FAC 近似；那不是本适配的目标场景。

一处必要修正：$h(x)$ 必须增广为 $z(x)=[h(x),1]$，否则 `nn.Linear` 的 bias 不进入 Fisher 几何，
式 (1) 的秩一梯度恒等式会漏掉 bias 分量。下文的 $h$ 在实现中一律指增广后的 $z$，$d\to d+1$。

- 特征 $h(x)\in\mathbb R^{d}$，取 $\lVert h(x)\rVert_2=1$；
- 头部参数 $W\in\mathbb R^{C\times d}$，$\theta=\operatorname{vec}(W)\in\mathbb R^{Cd}$；
- $p_c(x)=\sigma(\langle W_c,h(x)\rangle)$，$s_c(x)=\sqrt{p_c(x)(1-p_c(x))}$，$S(x)=\operatorname{diag}(s_c^2)$；
- 参考集 $V$（只用于设计方向、超参与模型选择，test 全程封闭）；
- $A_t$ 累计标注档案，$C_t$ 固定大小训练核心集，$U_t$ 未标注池，$Q$ 当前轮已选批次。

**逐样本梯度是秩一的。** 多标签 BCE $\ell(x,y)=\sum_c\mathrm{BCE}(p_c,y_c)$ 满足
$\nabla_{W_c}\ell=(p_c-y_c)h(x)$，故

$$
G(x,y)=(p-y)\,h(x)^\top\in\mathbb R^{C\times d},
\qquad
g(x,y)=\operatorname{vec}(G)=\sum_{c=1}^{C}(p_c-y_c)\,\psi_c(x).
\tag{1}
$$

**标签块基是正交归一的。** 定义 $\psi_c(x)=e_c\otimes h(x)\in\mathbb R^{Cd}$，则

$$
\langle\psi_c,\psi_{c'}\rangle=(e_c^\top e_{c'})\lVert h\rVert_2^2=\delta_{cc'}.
\tag{2}
$$

式 (2) 是后续所有闭式成立的原因：$\Psi(x)=[\psi_1,\ldots,\psi_C]\in\mathbb R^{Cd\times C}$ 是一组
正交归一列，$\Psi^\top\Psi=I_C$。**梯度表示天然按标签分块**，因此不需要另建 label-wise latent 分支。

**Fisher 是标签块上的对角二次型。** 设给定 $x$ 时标签条件独立（交叉项见第 6 节），

$$
F(x)=\mathbb E_{y\sim p}\big[gg^\top\big]
=\sum_{c}p_c(1-p_c)\,\psi_c\psi_c^\top
=\Psi S\Psi^\top
=S\otimes h h^\top .
\tag{3}
$$

**残差需求.** 记 $g_V=\nabla_\theta L_V(\theta_t)$，$A_0=H_0$ 为阻尼 Fisher/GGN 先验，

$$
r_b=A_b^{-1}g_V\in\mathbb R^{Cd},
\qquad
R_b\in\mathbb R^{C\times d}\ \text{为}\ r_b\ \text{的矩阵形},
\qquad
v_c^{(b)}(x)=\langle\psi_c(x),r_b\rangle=\langle R_{b,c},h(x)\rangle .
\tag{4}
$$

$v^{(b)}(x)=\Psi(x)^\top r_b\in\mathbb R^{C}$ 是候选样本在各标签方向上与当前残差需求的对齐向量。

## 3. 命题 1：signed 线性 influence 的期望在模型分布下为零

参考集 influence（忽略符号约定）为 $\mathcal I_b(x,y)=\langle r_b,g(x,y)\rangle=\sum_c(p_c-y_c)v_c^{(b)}$。

**命题 1.** 若候选标签服从 $y\sim q(\cdot\mid x)$，则

$$
\mathbb E_q\big[\mathcal I_b(x,y)\big]=\sum_{c=1}^{C}\big(p_c(x)-q_c(x)\big)\,v_c^{(b)}(x)
\;\xrightarrow{\;q=p\;}\;0 .
\tag{5}
$$

*证明.* 对式 (1) 取期望，$\mathbb E_q[p_c-y_c]=p_c-q_c$，代入内积即得。$\square$

**推论 1.1（作用范围）.** 式 (5) 就是 score-function identity（$\mathbb E_q[\nabla\log p]$ 在
$q=p$ 处为零）在本设定下的直接实例。它的结论**仅限于 signed 期望**：以 $\mathbb E_q[\mathcal I_b]$
为打分函数时，良好校准下期望信号为零，经验值主要由标签抽样噪声驱动，故该特定形式退化。

**必须明确不能由此推出的三件事**（初稿此处越界，本版收缩）：

1. 不能推出"线性 influence 是错误目标"。$\lvert\mathcal I_b\rvert$、
   $\operatorname{Var}_y(\mathcal I_b)$、单边效用 $\max(\mathcal I_b,0)$ 以及任何非线性泛函
   都不受式 (5) 约束——零均值不等于零信息。事实上第 6 节的式 (10) 正是
   $\operatorname{Var}_y(\mathcal I_b)$，本文的打分式本身就是"线性 influence 的一个二阶泛函"。
2. 不能推出"伪标签打分不合理"。取 $q\ne p$（伪标签、邻域传播、TTA 一致性）得到的是另一个估计量，
   其合理性取决于 $q$ 对真实条件分布的近似质量，属独立的经验问题，不由式 (5) 判定。
3. 不能推出 IPB-MAB §5.4 的锚点传播无效，只能说它额外依赖"邻域内 $q$ 与锚点标签一致"这一未验证
   假设；本文档不对该假设的成立与否表态。

**本文的实际主张，收缩为：** 在"不引入任何伪标签、只用模型自身分布"这一约束下，signed 一阶量退化，
因此需要一个二阶量；式 (10) 给出的方差是该约束下最自然的非退化选择。**这是一个存在性论证，不是
唯一性论证。**

**关于 BADGE（本版更正）.** 初稿称"BADGE 是对目标量选错的补偿"，此说不准确，应删除。BADGE 的实际
目标是在硬伪标签梯度嵌入 $g_x=\nabla_{W}\ell(x,\hat y)$ 上做 k-means++ 采样，其两个组成部分是
**梯度幅值**（$\lVert g_x\rVert$ 大者被 k-means++ 以更高概率选中）与**批内多样性**（k-means++ 的
$D^2$ 采样），二者都不是标量 signed influence，也不由式 (5) 覆盖。可以准确陈述的对比只有：

- BADGE 的幅值项与式 (9) 都含 $s_c$ 型的 Fisher 尺度，方向上一致；
- BADGE 的多样性来自候选点在梯度空间的几何（k-means++），本文的批内去冗余来自目标函数的 deflation
  （式 11）。两者机制不同，**孰优孰劣是待验证的经验问题（见 A1/A2），不是可由本节推出的结论。**

## 4. 目标函数：c-最优实验设计

在 Laplace 近似下，采集批次 $Q$ 后头部参数后验协方差近似为 $A_Q^{-1}$，其中

$$
A_Q=H_0+\sum_{x\in Q}F(x).
\tag{6}
$$

参考集损失的一阶泛函 $\langle g_V,\theta\rangle$ 的后验方差为 $g_V^\top A_Q^{-1}g_V$。于是定义

$$
\boxed{\;
\min_{Q\subset U_t,\;\lvert Q\rvert=B}\;
\Phi(Q)=g_V^\top\Big(H_0+\sum_{x\in Q}F(x)\Big)^{-1}g_V
\;}
\tag{7}
$$

这是以 $c=g_V$ 为方向的**c-最优实验设计**。式 (7) 给了 $r_b$ 严格身份：**当前后验下的残差需求方向**，
而不是类比。它也给出一条自然的停止规则（第 9.3 节）。

## 5. 命题 2：精确边际增益与其线性化

**命题 2.** 令 $M_b(x)=\Psi(x)^\top A_b^{-1}\Psi(x)\in\mathbb R^{C\times C}$，则将 $x$ 加入批次的
精确目标下降量为

$$
\Delta_b(x)=\Phi_b-\Phi_{b+1}
=v^{(b)}(x)^\top\big(S(x)^{-1}+M_b(x)\big)^{-1}v^{(b)}(x),
\tag{8}
$$

且满足上界

$$
\Delta_b(x)\;\le\;v^{(b)}(x)^\top S(x)\,v^{(b)}(x)
=\sum_{c=1}^{C}p_c(1-p_c)\,v_c^{(b)}(x)^2
\;=:\;\Delta_b^{\mathrm{lin}}(x).
\tag{9}
$$

*证明.* 由式 (3)，$A_{b+1}=A_b+\Psi S\Psi^\top$。Woodbury 恒等式给出

$$
A_{b+1}^{-1}=A_b^{-1}-A_b^{-1}\Psi\big(S^{-1}+\Psi^\top A_b^{-1}\Psi\big)^{-1}\Psi^\top A_b^{-1}.
$$

两侧左乘 $g_V^\top$、右乘 $g_V$，并代入 $v^{(b)}=\Psi^\top A_b^{-1}g_V$ 即得式 (8)。
对上界：$M_b\succeq0\Rightarrow S^{-1}+M_b\succeq S^{-1}\Rightarrow(S^{-1}+M_b)^{-1}\preceq S$，
对同一向量取二次型即得式 (9)。$\square$

**推论 2.1（饱和是自带的）.** $\Delta_b^{\mathrm{lin}}$ 是 $\Delta_b$ 的**上界**，二者差距随 $M_b$ 增大。
$M_b(x)$ 度量 $x$ 的标签块方向在当前后验下已被多大程度覆盖：近重复样本、高密度区域重复采样会使
$M_b$ 变大，从而使真实增益远低于线性估计。因此 IPB-MAB 中用 $\rho_t(x)$、批内相似度惩罚等手工项
压制的"重复/冗余"现象，在式 (8) 中是**目标函数自带的饱和**，不需要外挂。保留 $\rho_t$ 仅作为策略特征
（第 10 节），不再进入主打分。

**实现注记.** $M_b$ 是 $C\times C$，逐候选精确求逆代价高。默认对全池用式 (9) 粗筛，再对 top 候选用
式 (8) 精算（$C$ 大时对 $v$ 支撑集上的活跃标签子块求逆）。这是本方法唯一的近似-精算两阶段结构。

## 6. 命题 3：两条路线恒等

**命题 3.** 标签边缘化后的 influence 二阶矩等于式 (9)：

$$
\mathbb E_{y\sim p}\big[\mathcal I_b(x,y)^2\big]
=v^{(b)\top}\Sigma(x)\,v^{(b)}
\;\xrightarrow{\ \text{条件独立}\ }\;
\sum_{c}p_c(1-p_c)\,v_c^{(b)2}
=\Delta_b^{\mathrm{lin}}(x),
\tag{10}
$$

其中 $\Sigma(x)_{cc'}=\operatorname{Cov}(y_c,y_{c'}\mid x)$。

*证明.* $\mathcal I_b=\sum_c(p_c-y_c)v_c$，故
$\mathbb E[\mathcal I_b^2]=\sum_{c,c'}\mathbb E[(p_c-y_c)(p_{c'}-y_{c'})]v_cv_{c'}=v^\top\Sigma v$。
条件独立时 $\Sigma=S$，与式 (9) 右端相同。$\square$

**推论 3.1.** 由命题 1，$\mathbb E[\mathcal I_b]=0$，故式 (10) 同时就是 $\mathcal I_b$ 的**方差**。
方法的语义因此非常直白：**选择那些无论标签实现为何，都会对参考集需求方向产生大幅度影响的样本**。
这正是未标注候选上唯一不依赖伪标签的、非退化的一阶泛函。

**推论 3.2（标签相关性如何进入打分式，及其与第 9.1 节的互斥性）.** 若不做条件独立近似，交叉项
$\Sigma_{cc'}$ 直接进入式 (10)，标签相关性以**协方差加权**的形式出现，无需另设标签关系项。

**但必须写明代价：这与第 9.1 节的"精确分块对角"不能同时成立。** 分块对角性的来源正是条件独立
$\Sigma=S$（对角）。一旦以经验标签共现 $\widehat\Sigma$ 替代 $S$，
$F(x)=\Psi(x)\widehat\Sigma\Psi(x)^\top$ 在 $\widehat\Sigma_{cc'}\ne0$ 处产生**跨标签块**
$\widehat\Sigma_{cc'}hh^\top$，于是：

- $A$ 不再分块对角，式 (14) 的逐标签 $A_c$ 分解失效；
- 式 (14') 的闭式对角形式失效，$M_b$ 退回 $C\times C$ 稠密矩阵；
- 第 9.1 节据此删除的 K-FAC / rank-$r$ 截断**重新变得必要**。

因此这是二选一，当前实现选择前者：

| | 条件独立（当前实现） | 经验 $\widehat\Sigma$ |
|---|---|---|
| $A$ 结构 | 精确分块对角 | 稠密跨标签耦合 |
| 式 (14') 闭式 | 成立 | 失效 |
| K-FAC / rank-$r$ | 不需要 | 需要 |
| 标签相关性 | 仅经 $H^{-1}$ 与 $R$ 行间几何间接体现 | 显式建模 |

`rnd/scoring.py` 的模块 docstring 记录了该互斥关系。引入 $\widehat\Sigma$ 属未实现扩展，
**不得在同一处同时宣称"建模了标签相关性"与"精确分块对角"。**

## 7. deflation：批内去冗余的正确形式

**精确形式.** 由命题 2 的 Woodbury 展开，

$$
\boxed{\;
r_{b+1}=r_b-A_b^{-1}\Psi(x_b)\big(S^{-1}+M_b\big)^{-1}v^{(b)}(x_b)
\;}
\tag{11}
$$

**受控近似.** 若在新增标签块方向上近似 $A_b^{-1}\Psi\approx\delta^{-1}\Psi$（阻尼主导、且该方向尚未被
先前选择显著覆盖），则式 (11) 化为逐标签的秩一更新

$$
R_{b+1,c}=R_{b,c}-\beta_c(x_b)\,v_c^{(b)}(x_b)\,h(x_b)^\top,
\qquad
\beta_c=\frac{p_c(1-p_c)}{\delta+p_c(1-p_c)}\in[0,1).
\tag{12}
$$

**更正.** 本项目早期草稿中写作"$\beta_c\propto p_c(1-p_c)$"，该式仅在 $p_c(1-p_c)\ll\delta$ 时成立，
不是恒等式；正确表达为式 (12) 的饱和形式，$\beta_c$ 关于 $p_c(1-p_c)$ 单调递增但有界于 1。
实现中不应使用比例形式，而应让 $\Psi$ 通过已有的 $H^{-1}$ low-rank sketch（`active_learning/influence.py`）。

**语义.** 式 (12) 的 deflation 是**逐标签**的：一个样本只按它实际激活该标签的边界程度
$\beta_c$ 去消耗该行的残差需求。这恰是 prototype 想做而做不到的细粒度，且现在是从目标函数导出的，
不含 $M_c$、$\tau_{t,c}$、PFD、SLCI 等超参。

**猜想 7.1（探索-利用调度可能不必显式存在）——待验证，非定理.** 直觉是：第一步 $R_0$ 秩较高、需求
分散，argmax 倾向选彼此正交的样本（探索）；随 deflation 进行 $R_b$ 收窄为少数方向，argmax 转为在剩余
窄需求上局部加密（利用）。若成立，IPB-MAB 的 $\lambda_t$ 调度及其三个超参可以删除。

**但这只是对贪心动力学的定性描述，本文档未予证明**："$R_b$ 的有效秩随 $b$ 单调下降"并非式 (11) 的
推论——deflation 减小 $\lVert R\rVert$ 但并不保证有效秩单调，且"正交性偏好"依赖候选池的几何。
可验证形式：记录 $\operatorname{rank}_{\mathrm{eff}}(R_b)$ 与已选批次的平均成对余弦随 $b$ 的轨迹，
检验是否确实由分散转向集中（消融矩阵中列为独立条目）。

**猜想 7.2（稀有标签覆盖可能部分涌现）——待验证，非定理.** 直觉是：稀有标签 $c$ 在被覆盖前
$\lVert R_{b,c}\rVert$ 持续偏高，使能激活该标签的候选得分更高。若成立，IPB-MAB 手工逆频率权重
$w_c$ 的**部分**作用由 $\lVert R_{b,c}\rVert$ 承担。

**不能宣称完全替代**：$\lVert R_{b,c}\rVert=\lVert A_c^{-1}(g_V)_c\rVert$ 由参考集梯度与 Fisher
共同决定，而稀有标签在 $V$ 中正例同样稀少，$(g_V)_c$ 本身可能很小——两个效应方向相反，净结果是经验
问题。第 9.2 节保留的行范数下界正是为此：**如果该涌现是完备的，那条下界就不该有效果**；实测在
$\kappa=1$ 时抬升了 9/54 行，说明涌现并不完备。这条 floor 因此是对猜想 7.2 不成立部分的补偿。

## 8. 命题 4：核的精确因式分解，及与已有准则的关系分级

**命题 4.** 取 $u(x)=\sum_c\sqrt{w_c}\,s_c(x)\,\psi_c(x)$，则

$$
\boxed{\;
K(x,x')=\langle u(x),u(x')\rangle
=\underbrace{\Big[\sum_{c=1}^{C}w_c\,s_c(x)s_c(x')\Big]}_{K_{\mathrm{lab}}}
\cdot
\underbrace{\big\langle h(x),h(x')\big\rangle}_{K_{\mathrm{sem}}}
\;}
\tag{13}
$$

*证明.* 由式 (2)，$\langle\psi_c(x),\psi_{c'}(x')\rangle=\delta_{cc'}\langle h(x),h(x')\rangle$，
代入双线性展开即得。$\square$

**推论 4.1（复杂度）.** 式 (13) 只需缓存 $d$ 维语义向量与 $C$ 维标签 profile，代价 $O(C+d)$，
而 IPB-MAB §5.3 显式物化的 $\phi_t(x)\in\mathbb R^{Cd}$ 为 $O(Cd)$。原文档"用随机投影或按标签分块
控制维度"的补丁可以删除。

**推论 4.2（Fisher 权重不是启发式）.** $s_c=\sqrt{p_c(1-p_c)}=\big(\mathbb E_y[(p_c-y_c)^2]\big)^{1/2}$，
即对标签假设取期望后的梯度均方幅度。两个候选标签假设 $y_c\in\{1,0\}$ 给出方向 $(p_c-1)h$ 与 $p_ch$
——这就是 CoMAL 的"双极性"两极，而其二阶矩即 $s_c^2$。因此 IPB-MAB §6.5 额外的 $U^{\mathrm{bipolar}}$
项被式 (13) 吸收，不再需要作为独立消融项。

**对照表（区分"已证恒等"与"结构类比"）.** 初稿把整张表称为"归约表"，暗示每一行都是已证明的严格
特例。本版按证明强度分级——只有第一档可称归约：

**A 档：代数恒等，本文档已证。**

| 已有准则 | 关系 | 依据 |
|---|---|---|
| CoreSet / ProbCover | $K_{\mathrm{lab}}\equiv1$，标签轴退化的纯语义核 | 式 (13) 直接代入 |
| BAIT | 同为 Fisher 设计；目标函数不同（A-最优 trace vs 本文 c-最优参考方向） | 定义对比 |

**B 档：在附加假设下成立的特例，假设未验证。**

| 已有准则 | 关系 | 所需附加假设 |
|---|---|---|
| BALD / epistemic 不确定性 | 式 (9) 的特例 | $R_b$ 各行等权**且** $v_c$ 退化为常数——真实运行中均不成立 |

**C 档：结构类比或待验证猜想，不是归约。**

| 已有准则 | 可准确陈述的内容 | 不能宣称的内容 |
|---|---|---|
| BADGE | 幅值项与式 (9) 同含 Fisher 尺度 $s_c$ | 不能说 BADGE"归约为"本式：其 k-means++ 批多样性项在本框架中无对应物（见第 3 节更正） |
| CoMAL PFD $1-k^+_{t,c}$ | 二者都在"$R_c$ 相对已覆盖方向的剩余分量"这一层面起作用，形式上类似投影补 | **未证明** PFD 等于任何精确投影补：PFD 用余弦相似度的 $\max$，不是 $\operatorname{span}$ 上的正交投影，二者只在特殊配置下重合 |
| 多原型数 $M_c$ | 可解释为实例轴上的截断秩 | 这是**解释而非恒等**：CoMAL 的 $M_c$ 个 k-means 均值不是 $\operatorname{span}$ 的最优 $M_c$ 秩基，$M_c$ 与截断秩只是量纲一致 |
| SLCI | 行范数分布 $\lVert R_{b,c}\rVert$ 承载相近的信息 | **未证明**可完全替代，反证见猜想 7.2：第 9.2 节的 floor 在 $\kappa=1$ 时仍抬升 9/54 行 |
| Cluster-UCB | 式 (11) 的 deflation 在目标函数层面提供批内去冗余 | 不能说 UCB 的探索项被"归约"；两者机制不同，孰优是 A2 的经验问题 |

**关于 prototype 的收缩陈述.** 初稿称"prototype 是式 (13) 在实例轴上的低秩截断，用精确投影后它自动
消失"。可以严格支持的只有前半句的**方向**：prototype 在标签轴细、实例轴粗，而式 (13) 两轴都细，
所以式 (13) 表达力不低于 prototype。"$M_c$ 的真实身份是截断秩"与"精确投影后自动消失"属**解释性猜想**
——前者要求证明 k-means 均值构成最优低秩基（不成立），后者是关于 A3 实验结果的预测，尚未验证。
**因此第 1 节第 4 条与第 16 节中的"归约"措辞对 prototype 一项不成立，已相应改写。**

## 9. 低秩维护、复杂度与停止规则

### 9.1 精确分块对角：K-FAC 与 rank-$r$ 截断在本设定下均不必要

**这是本次 CoMAL 适配相对原方案的主要简化，且它是精确的而非近似的。**

原方案在此处引入 K-FAC $H\approx\Sigma_{\mathrm{out}}\otimes\Sigma_{\mathrm{in}}$ 与 rank-$r$ 截断
（$r\sim16\text{--}32$），以回避 $Cd\times Cd$ 规模的求逆。但在第 2 节的结构前提下这层近似是多余的：
由式 (3)，$F(x)=\sum_c s_c^2\,\psi_c\psi_c^\top$，而式 (2) 给出 $\langle\psi_c,\psi_{c'}\rangle=\delta_{cc'}$，
因此累计 Fisher **按标签精确分块对角**，无跨标签耦合：

$$
A=\delta I+\sum_{x}F(x)
\;\Longrightarrow\;
A_c=\delta I_d+\sum_{x}s_c(x)^2\,z(x)z(x)^\top\in\mathbb R^{d\times d},
\qquad c=1,\ldots,C.
\tag{14}
$$

于是 $R_c=A_c^{-1}(g_V)_c$ 逐标签独立求解，$m_c(x)=z^\top A_c^{-1}z$ 亦逐标签独立，
式 (8) 中的 $(S^{-1}+M_b)^{-1}$ 退化为**对角**矩阵，精确边际增益成为闭式标量和：

$$
\Delta_b(x)=\sum_{c=1}^{C}\frac{v_c^{(b)}(x)^2}{1/s_c(x)^2+m_c(x)}.
\tag{14'}
$$

数值验证：$A$ 的跨标签块最大元素为 $0.0$（`rnd/_check_blockdiag.py`），
非“接近零”而是构造上恒为零。

**由此删除三处结构：**（i）K-FAC 近似；（ii）rank-$r$ 截断与原式 (14) 的因子形式维护；
（iii）第 5 节“式 (9) 粗筛 + 式 (8) 精算”的两段式——既然式 (14') 已是闭式，全池直接精算即可；
式 (9) 仅作为消融项保留（用于检验推论 2.1 的饱和效应是否真的在起作用）。

代价：$C$ 个 $d\times d$ Cholesky。CoMAL/AAPD 规模（$C=54$，$d=769$）下建表 $0.65$s、
$150$ 候选打分 $0.2$s（CPU，float64）。**仅当 $C\cdot d^2$ 超出显存时才需退回原方案的低秩路径**，
本项目不触发。

### 9.2 唯一保留的手工先验：稀有标签行下界

本方法相对 PFD 丢失了一件事：**只有参考集 $V$ 有需求的新颖性才被认为有价值**。落在 $R$ 完全没有分量
方向上的新语义会被打零分。方向上这是正确的（无需求的新颖 = 猎奇），但 $\lvert V\rvert$ 小时稀有标签
的行会被系统性低估。因此对稀有标签行施加先验下界

$$
\lVert R_{b,c}\rVert_2\;\ge\;\kappa\,w_c,
\qquad w_c=(\pi_{t,c}+\epsilon)^{-1/2}\ \text{归一化并裁剪}.
\tag{15}
$$

式 (15) 是全案**唯一**的手工先验，也是"标签最低覆盖约束"唯一该存在的位置，必须单独消融
（$\kappa=0$ vs 调优 $\kappa$）。$V$ 较小时同时使用多个 reference folds 交叉拟合 $g_V$。

**两点实现约束（实测所迫）。**

1. **$\kappa$ 必须相对化。** $\lVert R_c\rVert$ 与 $\lVert g_V\rVert/\delta$ 同阶，跨数据集与跨轮次
   相差若干数量级，绝对阈值要么对所有标签生效、要么对任何标签都不生效。实现取
   $\text{target}_c=\kappa\,w_c\cdot\operatorname{median}_{c'}\lVert R_{c'}\rVert$，
   $\kappa=1$ 意为"把最稀有标签抬到中位数"。实测 $\kappa\in\{0,0.25,0.5,1,2\}$
   分别抬升 $0/0/3/9/23$ 行（共 54）。
2. **$\pi_{t,c}$ 必须取自 $V$ 而非 $L$。** $\lvert L\rvert\sim100$、$C=54$ 时多数标签在 $L$ 中零正例，
   $w_c$ 退化为常数，floor 静默失效。$V$ 属设计侧信息，允许使用；须加 Laplace 平滑。

**关键：flooring 改变了目标函数本身（初稿未写明，本版补正）。**

实现在 flooring 后令 $\mathrm{GV_{eff}}=A\,R^{\text{floored}}$，使 $R=A^{-1}\mathrm{GV_{eff}}$ 在整个
deflation 过程中保持成立，**代数一致性因此是精确的**（实测预测 $\Delta_b$ 与 $\Phi$ 实际降幅相对误差
$\sim10^{-13}$）。但代数一致 $\ne$ 目标未变：

$$
\kappa>0\;\Longrightarrow\;\mathrm{GV_{eff}}\ne g_V,
\qquad
\Phi_b=\mathrm{GV_{eff}}^\top A_b^{-1}\mathrm{GV_{eff}}\ne g_V^\top A_b^{-1}g_V .
$$

即 $\kappa>0$ 时贪心优化的是**以 $\mathrm{GV_{eff}}$ 为方向的设计目标**，而非式 (7) 以 $g_V$ 为方向的
原始 c-最优目标。因此：

- 该配置应称为**稀有标签稳健化目标（rare-label robustified objective）**，
  **不得**称为"原 c-最优目标的精确实现"；
- 一切最优性陈述（$\Phi$ 单调非增、$\Delta_b$ 精确、贪心性质）均**相对 $\mathrm{GV_{eff}}$** 成立，
  相对 $g_V$ 不成立；
- $\kappa=0$ 是唯一与式 (7) 严格等同的配置，消融必须含它作为"纯 c-最优"基准；
- 日志打印 $\lVert\mathrm{GV_{eff}}-g_V\rVert/\lVert g_V\rVert$ 与是否发生 flooring，
  使读者可判断目标偏离幅度（`rnd/acquire.py` 的 `objective:` 行）。

### 9.3 停止规则

$\Phi_b$ 单调非增且有下界，故可定义相对停止条件
$\big(\Phi_{b}-\Phi_{b+1}\big)/\Phi_0<\varepsilon$，即"本轮剩余候选已无法显著降低参考方向后验方差"。
这替代了固定 $B$ 的工程约定，可用于变预算实验（但主实验仍固定 $B$ 以便与基线可比）。

### 9.4 复杂度汇总

**初版的复杂度不可接受，本版重做（实测 564s $\to$ 11.8s，约 48 倍）。**

初版有三处叠加的问题，在 CoMAL 真实规模（$C=54$、$d+1=769$、$B=100$、$\lvert U\rvert=2\times10^4$）
下合计 $\sim2.5\times10^{12}$ 次立方级运算：

1. 全池精算式 (14')：$O(\lvert U\rvert Cd^2)$，且需 $(C,\lvert U\rvert,d)$ 的中间张量（float64 下 332 MB）；
2. 每步 deflation 后重做 $C$ 个 $769\times769$ Cholesky：$O(BCd^3)$；
3. 每步在工作集上重算 $m_c$：$O(BMCd^2)$。

三者分别对应三个可用的恒等式：

| 阶段 | 初版 | 本版 | 依据 |
|---|---|---|---|
| 特征 $z(x)$ | 一次前向 | 同 | 冻结 backbone，全程缓存 |
| $A_c$ 与求逆 | $O(C(Nd^2+d^3))$ | 同（仅一次） | 式 (14)，$C$ 个独立块 |
| 全池筛选 | 精算 $O(\lvert U\rvert Cd^2)$ | **上界** $O(\lvert U\rvert Cd)$ | $m_c\ge\lVert z\rVert^2/\operatorname{tr}A_c$ |
| $A_c^{-1}$ 维护 | 重分解 $O(Cd^3)$/步 | **Sherman--Morrison** $O(Cd^2)$/步 | $A_c$ 增量为秩一 |
| $m_c$ 维护 | 重算 $O(MCd^2)$/步 | **秩一更新** $O(MCd)$/步 | 同一 SM 量：$m_c\!-\!=\!\text{coef}_c\langle w_c,z\rangle^2$ |

**筛选上界（本版新增）.** 式 (9) 完全丢弃 $m_c$，在 $A_c$ 已吸收数据后极松。任何 $m_c$ 的**下界**都能
收紧它，而有一个是免费的：$A_c\preceq\lambda_{\max}I$ 且 $\lambda_{\max}\le\operatorname{tr}A_c$，故

$$
m_c(x)=z^\top A_c^{-1}z\;\ge\;\frac{\lVert z\rVert^2}{\operatorname{tr}A_c},
\qquad
\Delta^{\mathrm{ub}}(x)=\sum_c\frac{v_c^2}{1/s_c^2+\lVert z\rVert^2/\operatorname{tr}A_c}
\;\ge\;\Delta(x).
\tag{16}
$$

式 (16) 仍是上界，故 top-$M$ 截断**可采纳**（被丢弃者的精算值也必低于截断线），
且比式 (9) 紧得多。数值验证 $\Delta^{\mathrm{lin}}\ge\Delta^{\mathrm{ub}}\ge\Delta$ 逐点成立
（`rnd/_check_cost.py`）。

**关于"精确"的准确表述.** 贪心每步对**整个工作集 $S$** 精确计算式 (14')（缓存 $m$ 使其为 $O(MCd)$），
因此 argmax 在 $S$ 内是精确的、无 shortlist、不丢弃任何候选。唯一的近似是构造 $S$ 的那次上界筛选；
日志打印**筛选证书**（被丢弃者的最大上界 vs 保留者的最大精算值），据此判断首个选择是否可证明为
全池 argmax。$M$ 的截断量必须打印——否则覆盖率看起来会像是全池的。

**秩一累积误差是实测量，不是假设.** SM 更新与缓存 $m$ 都会累积浮点误差。批末重算若干候选的 $m$
与缓存值比对，实测 $B=100$ 后漂移 $4.2\times10^{-14}$；预测/实际 $\Delta$ 相对误差 $6.4\times10^{-13}$；
SM 与从头重分解的差 $4.7\times10^{-16}$（`rnd/_check_deflate.py`）。三者都进日志。

## 10. 智能体层：把解析式降级为先验

第 5--8 节的 argmax 仍是**固定手工规则**：它依赖条件独立近似、rank-$r$ 截断、$\delta$ 阻尼、
$\Sigma$ 收缩等一串近似，且 c-最优目标本身只是真实泛化增益的代理。智能体化的含义是保留解析式作为
**行为克隆先验与 shaping reward**，让策略去修正这些近似误差。

**MDP.** 一轮采样 = 一条长度 $B$ 的轨迹。

- 状态 $s_b=\big(\{\lVert R_{b,c}\rVert\}_c\ \text{的摘要},\ \Phi_b/\Phi_0,\ b/B,\ t/T,\ \text{标签覆盖缺口},\ \text{已选批次熵}\big)$；
- 动作：从 $U_t$（或其聚类子采样）选一个样本；
- 策略 $\pi_\psi(x\mid s_b)\propto\exp\big(\psi^\top f(x,s_b)/\tau\big)$。

**特征 $f(x,s_b)$（逐标签聚合，非拍平标量）.** 对 $\{(v_c^{(b)},s_c)\}_{c=1}^{C}$ 做置换不变的
DeepSets 式聚合（sum/max/top-k 分位），再拼接标量项：

$$
f=\Big[\ \text{agg}_c\{s_c^2v_c^2\},\ \text{agg}_c\{\lvert v_c\rvert\},\ \Delta_b^{\mathrm{lin}},\ \Delta_b\ (\text{若精算}),\ \operatorname{tr}M_b,\ \max_{x'\in Q_b}K(x,x'),\ \max_{x'\in A_t}K(x,x'),\ \rho(x),\ q(x),\ \widehat L(x)-\bar L_t\ \Big].
$$

刻意保持小规模：$\psi$ 只有几十个参数、只看无量纲特征、对候选池与标签集均置换不变，因此
**可跨数据集迁移**（如 ODIR 上训练、直接用于 FFAIR/BRSET）。这是"数据选择智能体"相对手工分数的
实质卖点，且可验证（迁移增益 vs 目标域重训策略）。

**训练：critic-free group-relative.** 真实奖励极稀疏（一轮一个标量）、轮数仅约 20，学 value function
必然过拟合，故用组内基线，不用 critic：

1. 同一状态下从 $\pi_\psi$ 温度采样 $G$ 个批次 $Q^{(1)},\ldots,Q^{(G)}$；
2. 每个批次在 **proxy 环境**中廉价重训（冻结 backbone + 线性头/LoRA，Selection-via-Proxy），
   在 reference fold 上得 $R^{(i)}=\Delta\text{macro-AUPRC}$；
3. 组内优势 $\hat A^{(i)}=(R^{(i)}-\operatorname{mean})/\operatorname{std}$，均摊给该轨迹的 $B$ 个决策；
4. 正则：对行为克隆先验策略的 KL（防坍缩）+ 熵奖励（保批内多样性）；
5. **dense shaping（本版更正：初稿的 potential-based 论证是错的）**。每步内在奖励取式 (8) 的
   $\Delta_b(x)$，$\sum_b\Delta_b=\Phi_0-\Phi_B(Q)$。初稿据此称"这是 potential-based shaping，
   不改变最优策略"——**该论证不成立**：

   Ng-Harada-Russell 的不变性要求 shaping 取 $F(s,s')=\gamma\Phi(s')-\Phi(s)$ 且**在终止状态强制
   $\Phi(s_{\mathrm{term}})=0$**。这里 $\Phi_B(Q)$ 依赖最终选中的批次 $Q$，是策略可以影响的量，
   并非常数；因此总 shaping 量 $\Phi_0-\Phi_B(Q)$ 随策略变化，直接叠加到 macro-AUPRC 奖励上会
   **改变最优策略**——策略可以通过压低 $\Phi_B$ 换取 shaping 收益而牺牲真实 AUPRC。

   两种正确做法，任选其一：

   **(a) 严格 potential-based。** 令 $\Phi(s_b)=-\Phi_b$（负号使 $\Phi$ 随进展递增），
   $F(s_b,s_{b+1})=\Phi(s_{b+1})-\Phi(s_b)=\Phi_b-\Phi_{b+1}=\Delta_b$，并在终止转移上取
   $F(s_B,s_{\mathrm{term}})=0-\Phi(s_B)=\Phi_B$。即**最后一步必须额外补上 $+\Phi_B$**，
   使整条轨迹的 shaping 总和恒为 $\Phi_0$（与策略无关的常数），不变性才成立。初稿遗漏的正是这一项。

   **(b) 不声称不变性，只当作偏置。** 保留 $\Delta_b$ 作为 dense 信号，但明确它是一个
   **有偏 shaping**，并以系数 $\eta$ 退火至 0（或只在行为克隆/预热阶段使用）。此时不得宣称
   "不改变最优策略"，只能宣称"加速早期信用分配，末期偏置消失"。

   **本文档采用 (a)**，因为它零成本且可验证：$\sum_b\Delta_b+\Phi_B\equiv\Phi_0$ 是一个可在实现中
   assert 的恒等式。若实现选择 (b)，必须在日志中打印 $\eta$ 的退火曲线。

**冷启动与下界.** 先行为克隆模仿式 (9)/(8) 的解析 argmax，再上 RL。最坏情况策略退化为解析规则，
方法有明确下界。

**推理时搜索.** 不做单条贪心：从 $\pi_\psi$ 采 $G$ 条 rollout 或 beam search，用**闭式**代理
$\Phi_0-\Phi(Q)$ 打分（**无需重训模型**）取最优批次提交专家。训练时用真实重训奖励、推理时用闭式代理
搜索，成本可控。

**超参数对比.** IPB-MAB 需调 $\{K_c,M,k_H,\alpha,\lambda_{\min},\lambda_{\max},\gamma,\eta,M_c,\delta,\tau_{t,c}\}$（11+）；
本方法需调 $\{r,\delta,\kappa\}$ 与标准 RL 超参 $\{\tau,G,\text{KL 系数}\}$。

## 11. 理论边界（不可越界宣称）

- **单调性成立**：$A'\succeq A\Rightarrow g^\top A'^{-1}g\le g^\top A^{-1}g$，故 $f(Q)=\Phi(\varnothing)-\Phi(Q)$ 单调非降。
- **子模性一般不成立**：c-最优/方差缩减目标不具备一般子模性，**不得宣称 $(1-1/e)$ 保证**。
  只能走 Das--Kempe 的 weak submodularity：若子模比 $\gamma>0$，贪心有 $(1-e^{-\gamma})$ 保证，
  而 $\gamma$ 的下界依赖 $\{\psi_c\}$ 的受限特征值/相干性条件——本文档**不声称已证明**该条件在真实
  数据上成立，只将其列为待检验项。
- **与 BAIT 的区别必须明写**：同为 Fisher 实验设计，BAIT 为 A-最优，本文为以 $g_V$ 为方向的 c-最优
  并配合逐标签 deflation。"用 Fisher 做主动学习设计"本身不是创新点。
- **Laplace / GGN 近似**：式 (6) 忽略高阶项，且假设采集后头部重训收敛到近似 MAP。
- **仅头部子空间**：全部闭式依赖第 2 节的线性头结构；backbone 端到端微调时式 (1) 的秩一性质失效，
  需退回块对角近似并重新验证。

## 12. 查询算法

```text
输入：已标注集 L, 未标注池 U, 参考集 V, 预算 B, deflation 子集大小 M, 稀有下界 kappa
输出：查询批次 Q

1.  读取实时模型的 CLS 特征与 logits。**必须同坐标系**：特征即头部实际消费的原始 CLS，
    不得再归一化（否则 (p-y)z 不是产生该 logits 的梯度）。日志打印 max|Z W^T - logits|。
2.  z(x) = [h(x), 1]（第 2 节的 bias 增广），全池一次前向后缓存。
3.  delta <- 阻尼超参。**这是调节量，不是推导量**：CoMAL 用 AdamW，没有可对齐的 L2 项，
    2*wd*|L|/d 只是继承自代理头的约定。建议用 --rnd_delta_rel 相对 Fisher 尺度指定，
    并打印 delta 与 fisher_scale 之比。它的实际作用是设定零空间尺度（|L| << d 时 A_c 秩亏）。
4.  A_c <- delta*I + sum_{x in L} s_c(x)^2 z z^T                      # 式 (14)
    一次 Cholesky 求逆得 A_c^{-1}；R_c <- A_c^{-1} (g_V)_c            # 式 (4)
5.  施加式 (15) 稀有标签行下界；随后记录 GV_eff = A R。
    **kappa>0 时目标已改变**（见 9.2）：此后 Phi 是对 GV_eff 而非 g_V 的设计目标，
    日志打印 ||GV_eff - g_V|| / ||g_V||。
6.  全池用**上界**打分（O(|U| C d)，绝不用式 (14') 精算——那是 O(|U| C d^2) 且要
    (C,|U|,d) 的中间量），取 top-M 为工作集 S。上界逐点支配式 (14')，故截断是
    **可采纳的**而非仅是省事；打印被丢弃的候选数与筛选证书。
7.  在 S 上一次性计算 m_c(x) = z^T A_c^{-1} z 并缓存。
8.  重复 B 次：
    8.1 Delta_b(x) = sum_c v_c^2 / (1/s_c^2 + m_c)，对**整个 S** 精确计算（O(M C d)）；
    8.2 若启用智能体：按 pi_psi(.|s_b) 采样/argmax（第 10 节）；否则取 argmax Delta_b(x)；
    8.3 x_b <- 选中样本；Q <- Q + {x_b}；S <- S - {x_b}；
    8.4 秩一 deflation，**Sherman-Morrison 更新 A_c^{-1}，不重分解**：            # 式 (11)
        w_c <- A_c^{-1} z；A_c^{-1} <- A_c^{-1} - [s_c^2/(1+s_c^2 z^T w_c)] w_c w_c^T；
        R <- A^{-1} GV_eff；并以同一秩一量更新缓存 m：m_c(x) -= coef_c <w_c, z(x)>^2。
    8.5 记录 Phi_b = <GV_eff, R>，验证单调非增与预测/实际 Delta 一致性；
    8.6 若式 (9.3) 停止条件满足则提前结束（仅变预算实验启用）。
9.  批末重算若干候选的 m 与缓存值比对，打印漂移上界（秩一累积误差的实测量）。
10. 返回 Q 提交标注；CoMAL 侧调用 dataset.update_data(Q) 并重建 labeled_loader。

**三处易错点，全部有对应日志行。**

1. **坐标系（第 1 步）.** 读 logits 后再缩放特征，会让 (p-y)z 变成另一个头的梯度。此错误不报异常、
   不影响单调性，只是让整套 Fisher 几何描述一个从未训练过的头。自检量：max|Z W^T - logits|。
   本项目实测曾犯此错（归一化 CLS 而 logits 取自原始 CLS），故列为第一位。
2. **GV_eff（第 5 步）.** flooring 后若不重算 GV_eff，deflation 不再精确（R != A^{-1} GV_eff），
   Phi 轨迹失去单调性。自检量：预测 Delta 与实际 Phi 降幅的相对误差（应 ~1e-13）。
3. **复杂度（第 6--8 步）.** 三个独立的坑：(i) 全池精算式 (14')；(ii) 每步重做 Cholesky；
   (iii) 每步重算 m。本项目初版三者皆犯，实测 C=54, d=769, B=100, |U|=2e4 下耗时 564s；
   改为上界筛选 + Sherman-Morrison + 缓存 m 后为 11.8s（约 48 倍），
   预测/实际误差 6.4e-13、m 漂移 4.2e-14、Phi 仍严格单调。见 `rnd/_check_cost.py`。

**关于 delta 的定位（本版更正）.** 初稿在第 3 步标注"与拟合头的 L2 项一致，非手调"，
**该表述已撤回**。它仅在用 `rnd/head.py` 的 L2 目标拟合代理头时成立；接入 CoMAL 实时模型后，
优化器是 AdamW，没有可对齐的 L2 项，因此 delta 是一个**普通超参**，须进入消融、
不得宣称"由拟合目标严格导出"。
```

## 13. 待验证假设

替代 IPB-MAB 的 H1--H4。前三项是**推导的经验前提**，第四项是方法卖点。

- **A1（核与打分的有效性，基石）— 已在 AAPD-54 上验证通过**。
  式 (14')/(9) 与真实重训增益的 Spearman 应显著高于线性 influence 与 BADGE；同时直接检验命题 1。
  实测（8 seeds，n_lab=20 / n_ref=4000 / n_cand=100，真值 = 揭示真标签重训后参考集损失下降）：

  | 打分器 | mean rho |
  |---|---|
  | `rnd_exact` 式 (14') | **+0.313**（8/8 seeds 为正，范围 +0.192~+0.508） |
  | `rnd_linear` 式 (9) | +0.282 |
  | `entropy` | +0.017 |
  | `random` | -0.040 |
  | `influence`（线性，y~p） | **0.000（恒为零）** |

  三点结论：（i）命题 1 得到精确验证——线性 influence 在 y~p 下恒等于零（max|.| = 0.0），
  不是接近零，因而完全无法排序候选；（ii）`rnd_exact > rnd_linear` 证实推论 2.1 的饱和项
  $M_b$ 确有贡献；（iii）**真值的选择本身是 A1 成败的一部分**：若改用 n_lab=200 + macro-AUPRC
  作真值，同一打分器只读出 +0.08（random +0.04），看似失败——实为噪声主导，eval 集 AP 的
  bootstrap 噪声底约为候选间信号的 5 倍。参考集损失下降既是 $\Phi$ 实际建模的量，
  信噪比也高约 3 倍。复现：`python -m rnd.a1`。
- **A2（deflation 替代 MAB）**：式 (11) 的批内去冗余在批内平均相似度、覆盖簇数、相同预算性能上
  应不劣于 Cluster-UCB。
- **A3（残差需求替代原型外扩）**：新增稀有标签正例数、新表型覆盖应不劣于 PFD/SLCI；
  并单独消融式 (15) 的 $\kappa$ 以量化"只有 $V$ 有需求的新颖性才有价值"这一行为变化的代价。
- **A4（智能体增益与迁移）**：学到的 $\pi_\psi$ 应显著超过解析 argmax，且跨数据集迁移有正增益。
  **若 A4 不成立，只发表第 2--9 节的统一与简化结论（本身已是干净贡献），不宣称智能体。**

辅助检查：rank-$r$ 截断误差 vs $r$；$\Sigma$ 收缩强度敏感性；$\Delta_b^{\mathrm{lin}}$ 与 $\Delta_b$
的排序差异（验证推论 2.1 的饱和确实在起作用）；$\gamma$（子模比）的经验估计。

## 14. 消融矩阵

| 消融 | 要回答的问题 | 对应命题 |
|---|---|---|
| 线性 influence 打分替代式 (9) | 二次型是否必要？ | 命题 1、3 |
| 硬伪标签（BADGE 式）替代期望 | 伪标签是否只是补偿目标量错误？ | 推论 1.2 |
| $K_{\mathrm{lab}}\equiv1$（纯语义核） | 标签轴细粒度是否有用？ | 命题 4 |
| 式 (9) 上界替代式 (14') 精算 | 饱和项 $M_b$ 是否重要？ | 推论 2.1（A1 已初步证实：+0.282 vs +0.313） |
| 无 deflation（top-$B$） | 批内去冗余是否来自目标函数？ | 式 (11) |
| top-$M$ 工作集大小 $M$ | deflation 子集截断是否损失覆盖？ | 9.4 |
| 恢复 PFD/SLCI 分支 | 原型外扩是否已被残差需求涵盖？ | 归约表、A3 |
| $\kappa=0$ | 稀有标签先验下界是否必需？ | 式 (15) |
| 条件独立 vs 经验 $\Sigma$ | 标签相关性是否需显式建模？ | 推论 3.2 |
| 解析 argmax vs $\pi_\psi$ | 智能体是否有净增益？ | A4 |
| $\pi_\psi$ 零样本迁移 vs 目标域重训 | 策略是否跨数据集可迁移？ | A4 |
| 单条贪心 vs 推理时搜索 | 闭式代理搜索是否有增益？ | 第 10 节 |
| A-最优（BAIT）替代 c-最优 | 参考方向是否优于 trace 目标？ | 第 11 节 |

## 15. 仓库落点（CoMAL）

原文本节指向的是另一个仓库（`models/vision_multicare.py`、`active_learning/influence.py`、
`active_learning/cluster_mab.py` 等，医疗多模态项目），与 CoMAL 无对应关系，**整节按下表重映射**。

新增模块（均已落地）：

| 文件 | 职责 |
|---|---|
| `rnd/features.py` | 冻结 BERT CLS 特征提取与缓存；`z=[h,1]` 增广在 `scoring.augment` |
| `rnd/head.py` | 凸线性头（LBFGS）+ `fisher_damping()` 给出 delta = 2*wd*N/d |
| `rnd/scoring.py` | `ResidualNeed`：式 (14) 分块 $A_c$、式 (14') 精算、式 (11) deflation、式 (15) flooring |
| `rnd/a1.py` | A1 离线回放验证 |
| `rnd/_check_*.py` | 秩一梯度、分块对角、deflation 精确性、Hessian/damping 一致性的数值自检 |

CoMAL 侧改动点：

- `selection_methods.py`：新增 `query_samples_rnd()`，与既有 `query_samples()`（CoMAL 原型 + PFD/SLCI）、
  `query_samples_other()`（core_set/badge/lloss/...）并列。**不改动既有分支**，使其保持为可比基线；
- `main.py`：`--method_type rnd` 走新分支；新增 `--rnd_*` 超参；
- 原型与 PFD/SLCI（`query_samples()`）**降级为消融基线**，不在主路径；
- Cluster-MAB 在 CoMAL 中本就不存在，故原方案"降级 cluster_mab.py"一条自动消解——
  其功能位（批内去冗余）由式 (11) deflation 承担；
- 诊断输出直接写入既有的 `sampler_record.log`（`utils.print_and_write_2_file`），
  记录 $\Phi_b$ 轨迹、$\lVert R_{b,c}\rVert$ 演化、每步预测 $\Delta_b$ 与实际 $\Phi$ 降幅之差。

仍需新增（尚未实现）：

1. 多 reference folds 交叉拟合 $g_V$（$\lvert V\rvert$ 小时）；
2. proxy 重训环境与 group-relative 策略训练循环（第 10 节智能体层）；
3. 推理时 rollout 搜索与闭式 $\Phi$ 代理评分。

原方案列出的"$R_b$ 低秩因子表示与 rank-$r$ 管理""式 (13) 因式分解核""式 (8) 活跃标签子块精算"
三项**不再需要**，理由见 9.1。

**实施顺序：A1 已通过，下一步是 A2/A3（把 `query_samples_rnd` 接入 AL 主循环跑真实曲线），
再写第 10 节的智能体层。** 若后续假设不成立，不应通过增加特征或权重掩盖，
而应回到第 3 节重新检查目标量的定义。

## 16. 可声明的贡献

不可把 influence、Fisher 设计、聚类、原型、RL 任一单点作为创新点。可声明的表述为：

> 证明多标签主动学习中的 influence 打分与样本相似度可统一为线性分类头诱导的单一 Fisher 几何：
> 未标注候选上的线性 influence 期望为零（其值只反映校准误差），正确的准则是标签边缘化后的
> influence 二阶矩，而该二次型恒等于以参考梯度为方向的 c-最优实验设计目标的边际增益线性化；
> 由此批内去冗余、探索-利用调度与原型外扩分别归约为目标函数自带的逐标签 deflation、残差秩的
> 自然收缩与低秩投影补，使原本并列的四个机制合并为单一打分式，且核可精确因式分解将复杂度从
> $O(Cd)$ 降到 $O(C+d)$；在此解析解之上以行为克隆与 potential-based shaping 训练一个小参数量、
> 标签与候选置换不变、因而跨数据集可迁移的选择策略，并在推理时用闭式目标代理做批次搜索。
