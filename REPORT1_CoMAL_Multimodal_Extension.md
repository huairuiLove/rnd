# CoMAL-SMI：基于样本级多模态交互与贝叶斯信息增益的主动学习

> 理论完善版。本文首先纠正早期版本中不成立的 CCA/PID 与“风险支配”结论，再给出一个可实现、可证伪、理论边界明确的多模态扩展。本文不是把 BERT 换成 CLIP：新增的核心对象是每个“样本—标签”的多模态交互估计，以及它与可约 epistemic uncertainty 的联合采集。

---

## 1. 结论摘要

CoMAL 可以扩展到多模态，但合理的研究问题不是“如何融合图像与文本”，而是：

> 在有限标注预算下，哪些样本—标签对同时具有可被标注消除的模型不确定性，并且确实需要两种模态联合解释？

本文提出 CoMAL-SMI（Sample-wise Multimodal Interaction）：

1. 保留 CoMAL 的 label-wise representation：
   \[
   z_{il}\in\mathbb R^p,\qquad i=1,\ldots,N,\ l=1,\ldots,L.
   \]
2. 为每个样本—标签估计 redundancy、两种 uniqueness 和 synergy：
   \[
   \widehat{\mathbf m}_{il}
   =(\hat r_{il},\hat u^v_{il},\hat u^t_{il},\hat s_{il}).
   \]
3. 用 BALD 型后验互信息估计标注能够消除的 epistemic uncertainty：
   \[
   \hat b_{il}=I(Y_{il};\Theta\mid v_i,t_i,\mathcal D).
   \]
4. 定义交互调制的信息增益分数：
   \[
   \hat a_{il}=w_l\,\bar b_{il}\bigl(1+\lambda\bar s_{il}\bigr).
   \]
5. 通过 label-wise coverage 与 facility-location 构成单调子模 batch 目标，贪心选样具有 \(1-1/e\) 近似保证。

这里的 synergy 不是由 CCA 或分类性能差“定义”出来，而是由明确选定的 PID/样本级交互估计器给出。理论保证针对 acquisition estimation regret 和 batch optimization，不声称 CoMAL-SMI 在任意数据分布上必然优于 CoMAL。

---

## 2. 源码事实：CoMAL 实际实现了什么

### 2.1 任务编码器

`networks.py` 中 `Encoder_No_GCN_No_Atten` 使用 BERT pooler 输出：
\[
h_i=f_\theta(x_i)\in\mathbb R^d.
\]
分类头为
\[
o_i=W_ch_i+b_c\in\mathbb R^L,
\]
训练损失是逐标签 BCE。

### 2.2 label-wise deterministic autoencoder，而不是严格意义上的 VAE

类名虽为 `MLP_VAE`，但其 `forward` 没有计算 \(\mu,\log\sigma^2\)，也没有调用文件中定义的 `reparameterize`；训练路径没有 KL 项。因此源码实现更准确地说是：

\[
h_i
\xrightarrow{\texttt{fc0}}
\mathbb R^{L\cdot512}
\xrightarrow{\mathrm{reshape}}
(q_{i1},\ldots,q_{iL}),
\]
\[
z_{il}=\texttt{fc1}(q_{il})\in\mathbb R^p.
\]

它是带 label-wise latent slots、prototype memory 和重构头的确定性自编码模块。后续报告统一称为 **label-wise prototype autoencoder**。

### 2.3 原型与监督对比

正标签原型以累积均值形式更新并归一化：
\[
P_l\leftarrow
\operatorname{norm}\left(
n_lP_l+\sum_{i\in\mathcal B}y_{il}w_{il}\hat z_{il}
\right).
\]

`controler.py` 将 \(z_{il}\) 展平后使用 `SupConLoss`，正负关系由已观测标签与 `cl_neg_mode` 决定。原型相似度路径使用 `sub_rep.data`，因而与该路径相关的原型更新和距离本身不向编码器反传；梯度主要来自监督对比和重构头。

### 2.4 原采集函数

以仓库脚本采用的 `cl_neg_mode=1` 为例，正样本相似度阈值为：
\[
\tau_l=
\frac{
\max_{j:y_{jl}=1}D_{jl}
+
\min_{j:y_{jl}=1}D_{jl}
}{2}.
\]

然后：
\[
\hat c_i=\sum_l\mathbf 1[D_{il}>\tau_l],\qquad
\delta_i=|\hat c_i-\bar c|,
\]
\[
d_i=
\left[
\sum_l\mathbf 1[\sigma(o_{il})\ge 0.5]\frac{D_{il}+1}{2}
\right]^{-1},
\qquad
q_i=\sqrt{d_i\delta_i}.
\]

该分数是启发式的“正特征多样性 × 标签基数不一致性”，不是 Bayesian information gain，也没有给出风险最优性保证。

---

## 3. 对早期理论版本的审计与撤回

### 3.1 CCA 不能识别 PID

CCA 最大化投影后的线性相关：
\[
\rho_k=\max_{a_k,b_k}
\operatorname{corr}(a_k^\top V,b_k^\top T).
\]

它描述两模态的二阶相关结构，却没有使用目标 \(Y_l\) 的完整联合分布。PID 分解的是
\[
I(V,T;Y_l),
\]
必须依赖 \((V,T,Y_l)\) 的分布和一个明确的 redundancy/unique information 定义。因此：

- 高 CCA 相关不等于 PID redundancy；
- 低 CCA 相关不等于 uniqueness；
- “去掉高、低相关方向后的残差”不等于 synergy。

CCA 可作为表示诊断或白化工具，但不能作为 PID 的可识别性定理。

### 3.2 性能差不是一般意义上的 synergy 一致估计

旧定义
\[
\mathrm{perf}(g^{vt})-\max\{\mathrm{perf}(g^v),\mathrm{perf}(g^t)\}
\]
混合了模型容量、优化误差、校准误差和数据量。即使联合模型更优，也不能据此唯一分解 redundancy、uniqueness 与 synergy；即使模型类无限大，该差值也不自动等于任一 PID 定义下的 synergy。

### 3.3 “协同样本必然更值得标”不成立

高 synergy 可以完全是 aleatoric、已经学会或与评价任务无关；低 synergy 样本也可能位于决策边界并具有很高 epistemic value。因此 synergy 只能作为任务交互结构，不能单独推出 expected error reduction，更不能无条件支配 CoMAL。

### 3.4 本文保留的理论承诺

本文只证明：

1. acquisition 分数估计误差如何传播为 top-\(B\) selection regret；
2. 所定义 batch 目标的子模性及贪心近似比；
3. 在明确的表示、后验和估计误差假设下，上述界成立。

模型最终性能是否提升必须通过实验验证。

---

## 4. 多模态交互的严格定义

### 4.1 Bivariate PID

令 \(X_1=V_l\)、\(X_2=T_l\)、目标 \(Y=Y_l\)。PID 写成：
\[
I(V_l,T_l;Y_l)=R_l+U^v_l+U^t_l+S_l.
\]
并满足：
\[
I(V_l;Y_l)=R_l+U^v_l,
\]
\[
I(T_l;Y_l)=R_l+U^t_l.
\]

四项并不由这三个等式唯一确定，必须额外选择 PID 定义。本文建议使用 Bertschinger–Rauh–Olbrich–Jost–Ay（BROJA）定义作为总体统计的基准：

\[
\Delta_p=
\left\{
q:q(v,y)=p(v,y),\ q(t,y)=p(t,y)
\right\},
\]
\[
U^v_l=\min_{q\in\Delta_p}I_q(V_l;Y_l\mid T_l),
\qquad
U^t_l=\min_{q\in\Delta_p}I_q(T_l;Y_l\mid V_l),
\]
\[
R_l=I_p(V_l;Y_l)-U^v_l,
\]
\[
S_l=I_p(V_l,T_l;Y_l)-R_l-U^v_l-U^t_l.
\]

不同 PID 定义可能给出不同数值；因此论文必须声明使用哪一种定义，不能把“synergy”当作无歧义的天然量。

### 4.2 为什么需要 sample-wise interaction

总体 \(S_l\) 只能说明标签 \(l\) 平均需要多少联合信息，不能区分未标注池中的具体样本。主动学习需要：
\[
s_{il}=s(v_i,t_i,y_{il}),
\]
即 pointwise/sample-wise synergy。

本文采用 Yang、Wang、Hu 在 ICML 2025 提出的 LSMI 作为主要工程估计器。LSMI 从 pointwise information 出发，输出：
\[
(\hat r_{il},\hat u^v_{il},\hat u^t_{il},\hat s_{il}).
\]

重要限定：

- LSMI 是已有估计方法，不是本文贡献；
- 本文不声称它对任意高维连续分布都无偏或一致；
- 其误差需在 XOR、AND/OR、redundant-copy 和 unique-only 合成数据上校准；
- 低维离散场景用 BROJA/CVX 作为参考真值；
- 高维表示场景报告 bootstrap 区间与估计稳定性。

---

## 5. CoMAL-SMI 表示结构

### 5.1 双模态 label-wise slots

冻结或低学习率更新两个编码器：
\[
e_i^v=f_v(v_i)\in\mathbb R^{d_v},
\qquad
e_i^t=f_t(t_i)\in\mathbb R^{d_t}.
\]

为每个标签构造三类 slots：
\[
z^v_{il}=\phi_l^v(e_i^v),\qquad
z^t_{il}=\phi_l^t(e_i^t),
\]
\[
z^{vt}_{il}=\psi_l(z^v_{il},z^t_{il}).
\]

其中 \(\psi_l\) 必须具有非加性交互能力，例如低秩双线性层：
\[
\psi_l(z^v,z^t)
=W_l[z^v;z^t;(A_lz^v)\odot(B_lz^t)].
\]

如果只使用线性加和，XOR 型 synergy 无法表示。

### 5.2 损失

\[
\mathcal L=
\mathcal L_{\mathrm{BCE}}
+\lambda_p\mathcal L_{\mathrm{proto}}
+\lambda_c\mathcal L_{\mathrm{SupCon}}
+\lambda_r\mathcal L_{\mathrm{recon}}
+\lambda_m\mathcal L_{\mathrm{marginal}}.
\]

其中：

- \(\mathcal L_{\mathrm{BCE}}\)：联合分类头的多标签 BCE；
- \(\mathcal L_{\mathrm{proto}}\)：正样本靠近 label prototype；
- \(\mathcal L_{\mathrm{SupCon}}\)：沿用 CoMAL 的 label-wise 监督对比；
- \(\mathcal L_{\mathrm{recon}}\)：防止 slots 退化；
- \(\mathcal L_{\mathrm{marginal}}\)：保证单模态头可独立校准，以便估计交互。

建议对 `loss_weight` 做严格 mask：
\[
\mathcal L_{\mathrm{BCE}}
=
\frac{
\sum_{i,l}w_{il}\,
\ell_{\mathrm{BCE}}(o_{il},y_{il})
}{
\sum_{i,l}w_{il}+\varepsilon
}.
\]

---

## 6. 可约不确定性：BALD 而不是 predictive entropy

高预测熵可能来自不可约噪声。主动学习应优先选择能通过标注减少的 epistemic uncertainty。令 \(\Theta\) 是模型参数后验：

\[
b_{il}
=I(Y_{il};\Theta\mid v_i,t_i,\mathcal D)
\]
\[
=
H\!\left[
\mathbb E_{p(\theta\mid\mathcal D)}
p_\theta(Y_{il}\mid v_i,t_i)
\right]
-
\mathbb E_{p(\theta\mid\mathcal D)}
H[p_\theta(Y_{il}\mid v_i,t_i)].
\]

用 \(K\) 个 deep ensembles、MC-dropout 或 Laplace 样本近似：
\[
\hat b_{il}
=
H\!\left(\frac1K\sum_{k=1}^Kp_{\theta_k}(Y_{il}=1\mid x_i)\right)
-
\frac1K\sum_{k=1}^KH(p_{\theta_k}(Y_{il}=1\mid x_i)).
\]

对二元标签，\(0\le b_{il}\le\log2\)。归一化：
\[
\bar b_{il}=\hat b_{il}/\log2\in[0,1].
\]

---

## 7. 交互调制 acquisition

LSMI 的 sample-wise interaction 可能有估计噪声或负的 pointwise 值。采用校准函数：
\[
\bar s_{il}
=
\operatorname{clip}
\left(
\frac{\hat s_{il}-q_{0.05,l}}
{q_{0.95,l}-q_{0.05,l}+\varepsilon},
0,1
\right).
\]

标签权重 \(w_l\) 可取逆频率的截断版本：
\[
w_l=
\min\left(w_{\max},
\frac{1}{\sqrt{n_l^++\varepsilon}}
\right).
\]

定义：
\[
a_{il}
=w_l\bar b_{il}(1+\lambda\bar s_{il}),
\qquad
a_i=\sum_{l=1}^La_{il}.
\]

解释：

- \(\bar b_{il}=0\)：模型没有可约参数不确定性，即使 synergy 高也不优先标；
- \(\bar s_{il}=0\)：退化为 label-wise BALD；
- \(\lambda=0\)：得到不含多模态交互的主动学习基线；
- synergy 只调制信息增益，不被宣称为信息增益本身。

这是风险下降的 surrogate，不等同于精确 EER。若算力允许，应把小池上的重训练式 EER 作为 oracle baseline。

---

## 8. Batch 目标与子模保证

仅取 top-\(B\) 容易重复选择相似样本。定义候选池 \(\mathcal U\)，batch \(A\subseteq\mathcal U\)。

### 8.1 label-wise saturation coverage

\[
F_{\mathrm{lab}}(A)
=
\sum_{l=1}^L
\alpha_l
\left[
1-\exp\left(-\sum_{i\in A}a_{il}\right)
\right].
\]

每个标签的边际收益随已选质量增加而递减。

### 8.2 facility-location diversity

用 CoMAL 的联合 label-wise slots 构造：
\[
k(i,j)
=
\frac1L\sum_l
\max\{0,\cos(z^{vt}_{il},z^{vt}_{jl})\}.
\]

\[
F_{\mathrm{div}}(A)
=
\sum_{j\in\mathcal U}
\max_{i\in A}k(i,j),
\]
约定空集最大值为 0。

总目标：
\[
F(A)=F_{\mathrm{lab}}(A)+\mu F_{\mathrm{div}}(A),
\qquad |A|\le B.
\]

### 定理 1：贪心 batch 近似保证

**命题。** 若 \(a_{il}\ge0\)、\(\alpha_l,\mu\ge0\)、\(k(i,j)\ge0\)，则 \(F\) 归一化、单调且子模。逐步选择最大边际增益的贪心解 \(A_g\) 满足：
\[
F(A_g)\ge(1-1/e)F(A^\star).
\]

**证明。**

1. \(x\mapsto1-e^{-x}\) 是非减凹函数；非负 modular 函数 \(\sum_{i\in A}a_{il}\) 与非减凹函数复合后为单调子模。
2. facility-location \(\sum_j\max_{i\in A}k(i,j)\) 是经典单调子模函数。
3. 非负线性组合保持单调子模性。
4. 应用 Nemhauser–Wolsey–Fisher 基数约束贪心定理。证毕。

该保证只说明给定估计分数时 batch 优化接近该 surrogate 的最优值，不保证真实测试风险的 \(1-1/e\) 近似。

---

## 9. 估计误差与 selection regret

定义理想但未知的归一化 acquisition：
\[
a_{il}=w_l b_{il}(1+\lambda s_{il}),
\quad b_{il},s_{il}\in[0,1].
\]

估计值：
\[
\hat a_{il}=w_l\hat b_{il}(1+\lambda\hat s_{il}).
\]

### 引理 1：误差传播

若
\[
|\hat b_{il}-b_{il}|\le\epsilon_b,\qquad
|\hat s_{il}-s_{il}|\le\epsilon_s,
\]
且 \(0\le\hat b_{il},b_{il},\hat s_{il},s_{il}\le1\)，则
\[
|\hat a_{il}-a_{il}|
\le
w_l\big[(1+\lambda)\epsilon_b+\lambda\epsilon_s\big].
\]

**证明。**
\[
\begin{aligned}
|\hat b(1+\lambda\hat s)-b(1+\lambda s)|
&\le(1+\lambda\hat s)|\hat b-b|
+\lambda b|\hat s-s|\\
&\le(1+\lambda)\epsilon_b+\lambda\epsilon_s.
\end{aligned}
\]
乘以 \(w_l\) 即得。证毕。

因此样本总分误差满足：
\[
|\hat a_i-a_i|
\le
\epsilon_i
=
\sum_lw_l[(1+\lambda)\epsilon_b+\lambda\epsilon_s].
\]

### 定理 2：top-\(B\) modular acquisition regret

令 \(A^\star\) 是按真实 \(a_i\) 最大化 \(\sum_{i\in A}a_i\) 的大小 \(B\) 集合，\(\hat A\) 是按 \(\hat a_i\) 选择的集合。若对所有 \(i\)，
\[
|\hat a_i-a_i|\le\epsilon,
\]
则：
\[
\sum_{i\in A^\star}a_i-\sum_{i\in\hat A}a_i
\le2B\epsilon.
\]

**证明。**
\[
\sum_{A^\star}a_i
\le\sum_{A^\star}\hat a_i+B\epsilon
\le\sum_{\hat A}\hat a_i+B\epsilon
\le\sum_{\hat A}a_i+2B\epsilon.
\]
证毕。

该结果是 estimation-to-selection 的确定性 regret 界；它不需要声称 synergy 估计器在所有分布上一致。

### 推论：高概率界

若 BALD 与 LSMI 校准在概率至少 \(1-\delta\) 下分别满足一致误差上界 \(\epsilon_b(n,\delta)\)、\(\epsilon_s(n,\delta)\)，则以同样概率：
\[
\operatorname{Regret}_B
\le
2B
\sum_lw_l
\left[
(1+\lambda)\epsilon_b(n,\delta)
+\lambda\epsilon_s(n,\delta)
\right].
\]

具体收敛率取决于所用后验近似、熵估计器、表示维数和分布光滑性，本文不虚构统一的 \(O(n^{-1/2})\) 结论。

---

## 10. 算法

### Algorithm 1：CoMAL-SMI

输入：已标注集 \(\mathcal D\)、未标注池 \(\mathcal U\)、预算 \(B\)、轮数 \(C\)、后验样本数 \(K\)。

每轮执行：

1. 训练多模态任务模型与 label-wise prototype autoencoder。
2. 校准联合和单模态预测头。
3. 用 \(K\) 个后验样本计算 \(\hat b_{il}\)。
4. 用 LSMI 估计 \(\widehat{\mathbf m}_{il}\)，并通过合成校准所得分位数得到 \(\bar s_{il}\)。
5. 计算 \(a_{il}=w_l\bar b_{il}(1+\lambda\bar s_{il})\)。
6. 先取 top-\(M\) 候选以控制复杂度。
7. 在候选上构造 \(F(A)\)，用 lazy greedy 选择 \(B\) 个样本。
8. 查询 oracle，更新 `pair_wise_sampled` 与 `all_labeled_mask`。

### 10.1 与现有源码的接口

建议新增而不是直接破坏现有类：

- `MultimodalClfDataset`：返回图像、文本、标签及原有 mask；
- `MultimodalLabelWiseBackbone`：输出 \(o_i,z^v,z^t,z^{vt}\)；
- `InteractionEstimator`：封装 LSMI/CVX；
- `BayesianAcquisitionEstimator`：封装 ensemble/MC-dropout；
- `SubmodularBatchSelector`：实现 lazy greedy。

原 `ClfDataset.update_data`、cycle 管理、指标与基线采集函数可复用。

---

## 11. 复杂度与工程约束

设池大小 \(N\)，标签数 \(L\)，slot 维数 \(p\)，后验样本数 \(K\)，候选数 \(M\ll N\)。

- 冻结编码器特征：一次性 \(O(NC_{\mathrm{enc}})\)，磁盘约 \(O(N(d_v+d_t))\)；
- label-wise slots：约 \(O(NLp)\) 输出存储，标签很多时按标签分块；
- BALD：\(O(KNC_{\mathrm{head}})\)，若主干也采样则成本显著增加；
- sample-wise interaction：取决于 LSMI 实现与 entropy estimator，必须报告实测时间/显存，不做未经验证的线性复杂度承诺；
- 全池 facility-location 是 \(O(N^2)\) 存储/计算，不可接受；
- top-\(M\)+稀疏 \(k\)-NN 图后，lazy greedy 约为 \(O(BMk)\) 的相似度更新。

---

## 12. 实验方案

### 12.1 数据集

1. **合成数据（理论验证必须项）**
   - XOR：纯 synergy；
   - duplicate bit：纯 redundancy；
   - \(Y=V\)、\(T\perp Y\)：视觉 uniqueness；
   - AND/OR：混合 redundancy 与 synergy；
   - 加入可控 label noise，区分 aleatoric 与 epistemic。
2. **NUS-WIDE**
   - 图像 + 用户标签/文本；
   - 81 个概念；
   - 注意用户标签本身可能直接泄露标签，需要去除同名词。
3. **MM-IMDb**
   - 海报 + plot；
   - 多标签 genre，天然适合图文多标签分类。
4. **UPMC Food-101**
   - 图像 + recipe text；
   - 可用于单标签 sanity check，不作为主多标签结果。

MS-COCO caption 与 object label 的关系高度冗余且标签可由 caption 直接泄露，应仅作为补充，不作为唯一主基准。

### 12.2 基线

- Random；
- entropy、BALD；
- Core-Set、BADGE；
- CoMAL 原采集函数；
- CoMAL + multimodal fused encoder（换壳基线）；
- BALD + facility-location；
- CoMAL-SMI without synergy（\(\lambda=0\)）；
- CoMAL-SMI with global PID（无 sample-wise）；
- 完整 CoMAL-SMI；
- 小池 EER oracle。

### 12.3 主要检验

1. **交互估计有效性**：合成数据上估计值与真值/参考 PID 的误差；
2. **估计稳定性**：不同 seed、bootstrap、batch size 下 rank correlation；
3. **主动学习效果**：性能—累计标注成本曲线；
4. **统计比较**：至少 5 seeds，报告均值、95% CI 与 paired test；
5. **理论对应**：测量 \(\epsilon_b,\epsilon_s\) 的经验上界与 top-\(B\) selection regret；
6. **batch 目标**：贪心、随机 batch、纯 top-\(B\) 的覆盖差异；
7. **标注粒度**：完整样本查询与 `(sample,label)` 查询分别评估。

### 12.4 评价指标

- P@1/3/5、nDCG@3/5；
- Macro/Micro-F1；
- mean average precision；
- area under learning curve；
- 达到全监督 90%/95% 性能所需标注数；
- 每标签风险下降与交互类型的相关性。

---

## 13. 可证伪预测

若方法成立，应观察到：

1. XOR 上 \(\hat s\) 显著高于 duplicate/unique-only；
2. \(\lambda>0\) 只在存在可约 synergy 的数据上优于 \(\lambda=0\)；
3. 把 XOR 标签加入强 aleatoric noise 后，高 synergy 不再自动带来高 acquisition，因为 BALD 项下降；
4. sample-wise interaction 比 global \(S_l\) 更能预测单次查询后的局部风险下降；
5. 在纯 redundancy 数据上，完整方法应自动退化到 BALD+coverage，而非强行追逐 synergy。

若这些预测不成立，应否定“synergy 调制主动采集”的核心假设。

---

## 14. 失败条件与边界

1. PID 不唯一；不同 redundancy 定义可能改变数值与排序。
2. LSMI 的 sample-wise 值可能对表示、熵估计与校准敏感。
3. 交互结构会随模型表示变化，active cycle 间不能假定固定。
4. 高 synergy 不等于高价值；必须与 epistemic 项联合。
5. MC-dropout/ensemble 只是参数后验近似，BALD 可能失准。
6. label-wise slot 数随 \(L\) 线性增长，极端多标签场景需低秩共享。
7. 数据中的文字可能直接包含标签名称，造成虚假的 uniqueness/redundancy。
8. 子模保证只针对设计的 surrogate，不直接保证泛化风险。

---

## 15. 研究贡献的准确表述

可以主张：

1. 首次将 sample-wise multimodal interaction 与 label-wise Bayesian active learning 联合；
2. 把 CoMAL 的 label-wise slots 用作样本—标签交互估计接口；
3. 构造交互调制的 epistemic acquisition；
4. 给出 acquisition estimation regret 界；
5. 给出 label coverage + facility-location batch 目标及 \(1-1/e\) 保证。

不能主张：

- CCA 识别 PID；
- 性能差是一致 synergy 估计；
- synergy 样本无条件比其他样本更有标注价值；
- CoMAL-SMI 无条件支配 CoMAL；
- LSMI 在所有高维连续分布上无偏或一致。

---

## 16. 参考文献

1. Peng, C., Wang, H., Chen, K., Shou, L., Yao, C., Wu, R., Chen, G. “CoMAL: Contrastive Active Learning for Multi-Label Text Classification.” *KDD 2024*, 2364–2375. https://doi.org/10.1145/3637528.3671754
2. Williams, P. L., Beer, R. D. “Nonnegative Decomposition of Multivariate Information.” arXiv:1004.2515, 2010. https://arxiv.org/abs/1004.2515
3. Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., Ay, N. “Quantifying Unique Information.” *Entropy* 16(4), 2014. https://doi.org/10.3390/e16042161
4. Liang, P. P., Cheng, Y., Fan, X. et al. “Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework.” *NeurIPS 2023*. https://arxiv.org/abs/2302.12247
5. Yang, Z., Wang, H., Hu, D. “Efficient Quantification of Multimodal Interaction at Sample Level.” *ICML 2025*, PMLR 267:71302–71317. https://proceedings.mlr.press/v267/yang25aj.html
6. Venkatesh, P., Schamberg, G. “Partial Information Decomposition via Deficiency for Multivariate Gaussians.” *IEEE ISIT 2022*. https://doi.org/10.1109/ISIT50566.2022.9834649
7. Venkatesh, P. et al. “Gaussian Partial Information Decomposition: Bias Correction and Application to High-dimensional Data.” *NeurIPS 2023*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/ec0bff8bf4b11e36f874790046dfdb65-Abstract-Conference.html
8. Houlsby, N., Huszár, F., Ghahramani, Z., Lengyel, M. “Bayesian Active Learning for Classification and Preference Learning.” arXiv:1112.5745, 2011. https://arxiv.org/abs/1112.5745
9. Kirsch, A., van Amersfoort, J., Gal, Y. “BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning.” *NeurIPS 2019*. https://arxiv.org/abs/1906.08158
10. Pinsler, R., Gordon, J., Nalisnick, E., Hernández-Lobato, J. M. “Bayesian Batch Active Learning as Sparse Subset Approximation.” *NeurIPS 2019*. https://arxiv.org/abs/1908.02144
11. Wang, Z., Yue, Y., Zhou, J. et al. “Active Learning with Expected Error Reduction.” arXiv:2211.09283, 2022. https://arxiv.org/abs/2211.09283
12. Nemhauser, G. L., Wolsey, L. A., Fisher, M. L. “An Analysis of Approximations for Maximizing Submodular Set Functions—I.” *Mathematical Programming* 14, 1978. https://doi.org/10.1007/BF01588971
13. Sener, O., Savarese, S. “Active Learning for Convolutional Neural Networks: A Core-Set Approach.” *ICLR 2018*. https://arxiv.org/abs/1708.00489
14. Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., Agarwal, A. “Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds.” *ICLR 2020*. https://arxiv.org/abs/1906.03671
15. Khosla, P. et al. “Supervised Contrastive Learning.” *NeurIPS 2020*. https://arxiv.org/abs/2004.11362
16. Radford, A. et al. “Learning Transferable Visual Models From Natural Language Supervision.” *ICML 2021*. https://proceedings.mlr.press/v139/radford21a.html
17. Arevalo, J., Solorio, T., Montes-y-Gómez, M., González, F. A. “Gated Multimodal Units for Information Fusion.” arXiv:1702.01992, 2017. https://arxiv.org/abs/1702.01992
18. Chua, T.-S. et al. “NUS-WIDE: A Real-World Web Image Database.” *CIVR 2009*. https://doi.org/10.1145/1646396.1646452
19. Arevalo, J. et al. “Gated Multimodal Units for Information Fusion.” Workshop paper using MM-IMDb, 2017. https://arxiv.org/abs/1702.01992

---

## 17. 最终判断

CoMAL 最值得保留的不是其原始 cardinality heuristic，而是其 `(sample,label,latent)` 三维表示接口。该接口使样本级多模态 interaction 可以落到每个标签，而不是停留在数据集级总体统计。

真正可辩护的创新是：

\[
\boxed{
\text{label-wise epistemic information gain}
\times
\text{sample-wise multimodal synergy modulation}
\times
\text{submodular batch coverage}
}
\]

它有清晰的退化基线、可证的估计 regret、可证的 batch 近似保证，也允许实验直接否定核心假设。相比“换编码器”和不成立的 PID/CCA 论证，这是一条更严格、更有研究价值的路线。
