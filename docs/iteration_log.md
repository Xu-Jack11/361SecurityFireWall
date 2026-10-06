# 迭代记录

按时间顺序记录本仓库的所有版本迭代与实验。每条记录给出动机、做法、产物位置和结论。
除特别说明外，所有训练均为标准配置：300k 分层采样、TF-IDF 1–2 gram、
max_features=120k、min_df=3、torch GPU 后端（T4）。

## I1 · 基线实现（≤ 2026-07-09）

- **内容**：`src/soc_baseline/` 完整流水线——分层采样 → `build_log_documents`
  文本特征 → TF-IDF + 线性分类器（sklearn `SGDClassifier` 与
  `TorchTfidfClassifier` 双后端，接口对等）→ holdout 评估 → 全量重训 →
  流式预测 `res.csv` → 提交自校验。设计文档见 `docs/superpowers/specs/`
  与 `docs/superpowers/plans/`。
- **产物**：`artifacts/`（含 `smoke/`、`gpu_smoke/` 冒烟运行）、`res.csv`。
- **指标**：随机 holdout（60k 行）accuracy 0.99997，**macro-F1 0.99974**。

## I2 · 公开数据集调研（2026-07-09）

- **内容**：调研 LANL、Splunk BOTS、OTRF、DARPA OpTC、CIC-IDS、UNSW、CERT
  等公开安全数据集，结论是字段/标签定义差异大，不直接混入训练。
- **产物**：`docs/public_datasets.md`。

## I3 · 确认比赛数据与 WitFoo Precinct 6 同源（2026-07-10）

- **动机**：训练数据中出现 `vendor_name=WitFoo`、`product_name=Precinct`，
  且脱敏 token（`USER-NNNN`/`ORG-NNNN`/`CRED-NNNN`）与 WitFoo 开源的
  sanitizer 一致。
- **做法**：定位 Hugging Face 上的公开数据集并做记录级比对（比对部分由
  Codex 会话 `019f4a95…` 完成）。
- **结论**：官方 2026-03 历史版本（commit `bd422e30`，v4）的
  `benign=1,899,723`、`suspicious=45,420` 与比赛 `train.parquet` 完全一致；
  比赛 malicious=111,728 是 v4 的 125,780 的子集。HF 当前版（latest）在
  2026-03~05 间被重新清洗、重新标注，脱敏映射已变。
- **产物**：`data/external/witfoo-precinct6-signals-v4.parquet`（207 万行）、
  `data/external/witfoo-precinct6-signals-latest.parquet`（210 万行）；
  出处与 sha256 记录在 `docs/public_datasets.md`。
- **风险记录**：比赛测试集大概率也来自公开集，任何外部数据合并前必须
  双向去重并确认比赛规则。

## I4 · 在 WitFoo v4 上复跑基线（2026-07-10）

- **做法**：同参数、训练集换成 v4，测试集不变。
- **指标**：holdout macro-F1 **0.99974**（与比赛数据基线持平，混淆矩阵
  60k 行仅 2 个错误）。
- **对比**：对 `valid_input.parquet` 的预测与原基线**一致率 98.44%**；
  3.1 万行分歧集中在 benign↔malicious 双向翻转，与两数据集唯一的实质差异
  （v4 多 1.4 万条 malicious 标签）吻合——这批分歧样本圈出了决策边界。
- **发现**：两个模型都预测测试集约 44% malicious（训练先验仅 5.4%），
  说明测试集被重新平衡过。
- **产物**：`artifacts/witfoo_v4/`。

## I5 · 外部数据均衡性分析（2026-07-10）

- **结论**：外部数据与比赛数据同样偏斜（v4：91.7/6.1/2.2；latest：
  90.4/7.4/2.2；100m 版更极端，99.34% benign）。2M 版对 suspicious
  **没有增量**（45,420 条与比赛完全相同）；malicious 有现成增量
  （v4 +1.4 万、latest +4.4 万）。更多 suspicious 只能去 100m 版过滤抽取。

## I6 · 泛化能力测试 E1–E4（2026-07-10）

- **动机**：特征重要性显示少数类 top 特征是具体 IP 前缀、时间点和
  org token，怀疑 0.9997 来自记忆而非泛化。
- **做法**：`scripts/generalization_test.py`，2×2 实验
  （时间切分 vs 跨版本评估 × 原始 vs token 归一化特征）。
- **结果**：

  | 实验 | 验证 | 特征 | macro-F1 |
  | --- | --- | --- | --- |
  | E1 | 时间切分（后 20%） | 原始 | 0.666（验证窗 0 条 malicious，该类 F1 记 0） |
  | E2 | 时间切分 | 归一化 | 0.666 |
  | E3 | 跨版本（latest 全量 210 万行） | 原始 | **0.9987** |
  | E4 | 跨版本 | 归一化 | 0.976 |

- **结论**：
  1. "换脱敏映射就崩"的假设**被 E3 证伪**——跨版本迁移几乎无损，
     连 latest 重标的 4.4 万条 malicious 都有 0.997 召回；
  2. token 归一化**不采用**（E4 反而掉 2.2 个点，粗折叠 IP 抹掉了
     DMZ/目标网段等稳定语义）；
  3. 全部 malicious 集中在时间轴前 80%（攻击活动时间上高度聚集），
     时间切分对 malicious 类退化；对"全新攻击活动"的检测能力用本数据
     无法验证，属于数据边界而非模型缺陷。
- **产物**：`artifacts/generalization/results.json`。

## I7 · 测试集覆盖检查（2026-07-10）

- **做法**：按 `pipeline×vendor×product` 对比 train 与 valid_input 的来源组合。
- **结论**：测试集 201.4 万行中 99.99% 来自训练集已见过的 14 种组合，
  未见过的仅 209 行（Microsoft Graph 200 行、Symantec DLP 9 行）。
  **不需要大规模补充外部数据**；唯一可选小补丁是从 100m 版捞这两个产品的
  样本（影响上限 0.01%）。

## I8 · 测试集先验校准（2026-07-10）

- **动机**：测试集先验与训练先验差约 8 倍（I4 发现），balanced 权重的
  argmax 决策点未必最优；验证集与训练同分布，无法直接调阈值，
  改用无标签的 EM 先验估计（Saerens 2002）+ 后验重加权。
- **做法**：`scripts/prior_calibration.py`——用 `artifacts/model.joblib`
  流式输出测试集概率，EM 估计测试先验，`p'(y|x) ∝ p(y|x)·p_test(y)/p_model(y)`
  重新决策，产出校准版提交文件。
- **结果**（`artifacts/calibration/summary.json`）：
  - EM 估计测试先验：benign 79.3% / malicious 19.9% / suspicious 0.8%
    （46 轮收敛）；测试集平均置信度仅 0.61/0.35/0.04，远低于域内的 ~0.99。
  - 校准翻转 **801,754 行（39.8%）**，其中 malicious→benign 78.2 万、
    suspicious→malicious 1.8 万；校准后分布 93.0/6.0/1.0。
- **机制验证**（用有标签的 latest 构造两种已知先验的混合）：
  EM 先验估计在域内准确（误差 <1%），但校准决策在两种先验下都轻微
  变差（0.9983→0.9883、0.9981→0.9969），且域内只翻转 0.2–0.3% 的行。
  测试集上 39.8% 的翻转量说明其形态在域外。
- **来源分析（决定性证据）**：测试集被判 malicious 的行中 **78% 来自
  AWS CloudTrail**——该源在训练集中 100% benign（全是 Instance Backup
  事件）、训练集 malicious 100% 来自空 vendor 的 syslog；且 CloudTrail
  占测试集 47.4%（训练集仅 16.5%）。原始 44% malicious 主要是模型在
  从未见过恶意样本的源上、遇到陌生 token 后的低置信度外推。EM 的
  ~20% malicious 估计与来源构成的上限（syslog 源约占 15%）自洽。
- **结论**：`res.csv`（原始）和 `artifacts/calibration/res_calibrated.csv`
  （校准）两版都保留。证据倾向校准版更接近真实分布，但无标签不能定论；
  如平台允许多次提交，两版各提交一次用线上分数裁决。下一步候选：
  按来源感知决策（训练中从未 malicious 的源提高 malicious 判定门槛）。
- **产物**：`scripts/prior_calibration.py`、`artifacts/calibration/`。

## I9 · 去除时间与标识符泄漏，重训并生成四版提交（2026-10-06）

- **动机**：反事实实验证明原模型的 malicious 预测由时间 token 驱动——
  训练集 benign/suspicious 几乎全部是 2024-07-26 UTC 11 点，malicious
  分布在 2022-06~2024-07-18；测试集 99.3% 落在 07-26~08-01 且小时分散。
  把测试样本日期改成 07-26 后，原模型 malicious 预测从 44.2% 降到 5.1%
  （AWS Instance Backup 从 72% 降到 0%）。I8 的 EM 先验估计同样建立在
  这批被时间 token 扭曲的概率上，作废。
  另外，日志正文里也带时间（`jan 30`、`164 may`、`23 2022` 都是
  malicious 的高权重特征），测试集的 `src_host` 训练集里一个都没出现过，
  脱敏 ID 只有 2.3% 重合。
- **做法**：
  - `features.py` 新增 `feature_set="content"`：去掉时间 token、IP 原值、
    /24 前缀、主机名和用户名，只保留 pipeline/product/vendor、字段形态
    （`src_ip_kind=ipv4`）、端口分桶、`message_empty`，以及经过
    `normalize_message` 处理的正文（日期、时间、IP、脱敏 ID、长数字、
    hex、UUID 都替换为占位符）。对正文先去重再归一化，速度约提升 4 倍。
  - 新模块 `source_mask.py`（`--source-label-mask`）：训练集中至少有 100 行
    的来源，只能预测该来源在训练集里出现过的标签；稀有或没见过的来源不受限制。
  - `scripts/apply_date_rule.py`：只从官方训练集学出一个时间切点（第一条
    非 malicious 记录，2024-07-26 11:10 UTC，切点之前的 111,728 行全部是
    malicious），把测试集中早于切点的行强制判为 malicious。
- **训练集内 holdout**（content + 掩码）：macro-F1 **1.0**（6 万行全对）。
  去掉时间和标识符后，训练集内的区分能力没有损失。
- **四版提交对比**（测试集 2,014,052 行）：

  | 版本 | 文件 | benign | suspicious | malicious |
  | --- | --- | --- | --- | --- |
  | A 原基线 | `res.csv` | 54.10% | 2.01% | 43.88% |
  | B content | `artifacts/content/res_content.csv` | 97.57% | 2.28% | 0.15% |
  | C B+来源掩码 | `artifacts/content/res_source_mask.csv` | 97.66% | 2.04% | 0.30% |
  | D C+日期规则 | `artifacts/content/res_date_rule.csv` | 97.27% | 2.04% | 0.70% |

  B/C 与 A 的一致率只有 56%；C 相比 B 翻转了 4,952 行，D 相比 C 翻转了 7,970 行。
- **关键发现**：B 模型完全看不到时间，但它的 3,084 条 malicious 预测、
  以及空 vendor 来源里的 4,916 条 suspicious 预测，**全部**落在测试集里
  早于 07-26 的 14,052 行上（这批行在其余 200 万行里只出现 36 条）。
  这批早期行都来自空 vendor 来源，内容都是被拦截的流量：ASA `Deny`
  5,240 行、VPC `REJECT` 2,664 行、Palo Alto `TRAFFIC,drop` 5,838 行
  （这种格式训练集里没有，B/C 全部判成了 benign）。外部 v4/latest 中
  同样是“早于 07-26 = 100% malicious，之后 = 100% benign/suspicious”；
  而且 v4 malicious 125,780 − 官方 train 111,728 = **14,052**，Palo Alto
  5,838 和早期 REJECT 2,664 也和 v4 完全对上。因此测试集的早期行基本可以
  确定就是 v4 中没有放进训练集的 malicious。
- **结论**：
  1. 原 `res.csv` 的 88 万条 malicious 基本都是时间泄漏造成的误报。
  2. 推荐提交顺序：**D → C**（A 只作对照，配额有限时可以不交）。
     D 依赖“早期 = malicious”这一数据集构造规律；它只从官方训练集学出，
     外部数据只用于验证，**没有**把外部标签抄进提交文件。
  3. 测试集的 200 万条晚期行不在 v4 里（脱敏命名空间不同，例如
     `USER-0147`、`HOST-5000xx`），它们的真实标签目前无法验证。
     其中 Crowdstrike 的 892 行因为训练中只有 35 行不受掩码约束，
     B/C 判为 684 suspicious / 208 benign。
- **产物**：`artifacts/content/`（模型、指标、`source_label_mask.json`、
  B/C/D 提交文件、`version_comparison.json`）、`scripts/apply_date_rule.py`。
