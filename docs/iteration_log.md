# 迭代记录

按时间顺序记录本仓库的所有版本迭代与实验。每条记录给出动机、做法、产物位置和结论。
除特别说明外，所有训练均为标准配置：300k 分层采样、TF-IDF 1–2 gram、
max_features=120k、min_df=3、torch GPU 后端（T4）。

> **术语说明（2026-10-08）**：`data/valid_input.parquet` 是**验证集**，不是比赛测试集
> （答案在 `valid_answer_private.parquet`，见 I11）。I11 之前的记录把它叫作“测试集”，
> 原文保留未改。各版本在 valid 上的预测文件沿用了 `res*.csv` 的命名，
> 但它们都不是提交文件。之后在 valid 上的运行写入 `predictions.csv` / `valid_pred.csv`，
> `res.csv` 只留给真正测试集的提交。

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

## I10 · 在 Precinct 6 v2.1.0 上抽样评测（2026-10-06）

- **动机**：WitFoo 于 2026-09 发布 v2（当前为 HF tag `v2.1.0`），修复了 v1
  中“benign 和攻击在时间上从不重叠”的问题：`signals` 是 07-26 11:10 到
  08-01 的实时采集，其中 7,728 条事件关联命中的记录被**原位**标为 malicious；
  历史攻击单独放进 `incident_signals`（23.9 万行，全部 malicious）；token
  注册表也重建了。这是第一个不能靠时间区分标签的测试集。
- **做法**：`scripts/eval_precinct6_v2.py`。从 `signals` 分层抽 30 万行，
  另外单独统计其中 07-27 及之后的部分（共 20.9 万行，避开与训练集的 07-26
  重叠）；从 `incident_signals` 随机抽 5 万行。对 A/B/C/D 四个版本分别评测。
- **结果**（macro-F1；括号内为 malicious 的 P / R）：

  | 版本 | live 30 万 | live 07-27 之后 | incident（malicious 召回） |
  | --- | --- | --- | --- |
  | A 原基线 | 0.644（0.005 / 0.077） | 0.637（0.004 / 0.132） | 0.809 |
  | B content | **0.749**（0.41 / 0.22） | **0.698**（0.30 / 0.09） | 0.756 |
  | C B+来源掩码 | 0.655（0 / 0） | 0.654（0 / 0） | 0.030 |
  | D C+日期规则 | 0.655（0 / 0） | 0.654（0 / 0） | 0.9997 |

  benign 和 suspicious 方面，B/C/D 的 benign 召回都是 0.9999，suspicious F1 约 0.96。
  A 在 live 上把 7% 的 benign 判成了 malicious（约 2 万行），这在独立数据上
  再次证实了时间泄漏。
- **结论**：
  1. **v2 的 malicious 基本不可学**：它们和同一来源的 suspicious 文本相似
     （ASA：malicious 3,525 / suspicious 91,537；VPC：4,156 / 15,063），
     区别在于是否被 Precinct 的事件关联命中，单条日志看不出来。
     B 的 ASA malicious 召回是 50%，VPC 是 0%。
  2. **来源掩码和日期规则都只针对 v1 的打标方式**：v2 中 ASA/VPC 会出现
     malicious，被掩码直接压没（C 在 live 上 malicious 召回为 0，在
     incident 上只有 3%）。D 在 incident 上的 99.97% 完全来自日期，在 live
     上不起作用。
  3. 在 v2 上**泛化最好的是 B**（去掉泄漏特征、不加规则）。比赛提交仍建议
     按 I9 的 D → C 顺序，因为比赛测试集的构造方式和 v1 一致（早期行就是
     v4 中留出的 malicious）。但如果目标是一个可迁移的检测模型，应该
     以 B 为基础，C 和 D 不要带入。
  4. 想提升真实场景下的 malicious 检测，需要事件上下文（同一主机或会话的
     邻近事件、关联规则命中），单条日志分类做不到。
- **产物**：`data/external/precinct6-v2.1.0/`（数据，不进 git）、
  `scripts/eval_precinct6_v2.py`、`artifacts/precinct6_v2/`（`results.json`、
  按来源统计的 malicious 召回）。

## I11 · 用比赛验证集答案打分（2026-10-06）

- **数据**：`data/valid_answer_private.parquet`（用户导入，含 `event_id`、
  `label_binary`，覆盖 `valid_input` 全部 2,014,052 行；在 `data/` 下，不进 git）。
  答案分布：benign 1,959,573 / suspicious 40,427 / malicious **14,052**。
  **答案只用于打分，不能用于训练或调参。**
- **做法**：`scripts/score_submissions.py`，结果写入 `artifacts/answer_scores.json`。
- **结果**：

  | 版本 | macro-F1 | accuracy | 错误行数 | F1（benign / suspicious / malicious） |
  | --- | --- | --- | --- | --- |
  | A 原基线 `res.csv` | 0.5789 | 0.5662 | 873,756 | 0.714 / 0.996 / 0.027 |
  | A + I8 EM 校准 | 0.5900 | 0.9406 | 119,730 | 0.974 / 0.679 / 0.118 |
  | A·v4 训练（I4） | 0.5738 | 0.5677 | 870,625 | 0.715 / 0.975 / 0.031 |
  | B content | 0.7644 | 0.9942 | 11,606 | 0.998 / 0.936 / 0.359 |
  | C B+来源掩码 | 0.8649 | 0.9957 | 8,572 | 0.998 / 0.993 / 0.604 |
  | **D C+日期规则** | **0.9975** | **0.9997** | **602** | 1.000 / 0.993 / **1.000** |

- **结论**：
  1. I9 的推断全部得到答案证实：原基线有 87.2 万行 benign 被时间泄漏误判成
     malicious；malicious 恰好就是早于切点的 14,052 行（D 的 malicious
     P = R = 1.0）。训练集内 holdout 的 0.9997 和真实成绩 0.579 差得很远，
     随机切分的 holdout 不能用来评估这份数据。
  2. I8 的 EM 校准只提升到 0.590，证实它建立在被时间泄漏扭曲的概率上。
  3. D 剩下的 602 个错误：Crowdstrike Falcon 有 593 行 benign 被判成
     suspicious（训练集里 Crowdstrike 只有 35 行，而且全是 suspicious，
     从训练数据学不出它的 benign；答案里的 benign 是主机资产类 Artifact
     记录，suspicious 是 `Crowdstrike Detection`）；Symantec DLP 有 9 行
     suspicious 被判成 benign（训练集里没有这个来源）。这两处只有看了答案才能
     修，按答案改规则属于在验证集上过拟合，**没有做**。

## I12 · benign 优先的代价敏感决策（2026-10-06）

- **动机**：业务要求漏判 benign（真实是 benign 却被判成告警）的代价更高，
  拿不准时应该判 benign。D 剩下的 593 个 benign 漏判全部是 Crowdstrike，
  模型对它们给出的 suspicious 概率只有 0.41–0.86；而真正的 suspicious 大多
  在 0.99 以上。
- **做法**：新模块 `decision.py` 中的 `weighted_argmax`，在允许的标签里取
  `p(y|x) × weight[y]` 最大的一类（Bayes 代价决策）；CLI 参数
  `--benign-weight w` 表示只有当告警类的概率超过 benign 的 w 倍时才判告警。
  模型和来源掩码都不变，日期规则照常在最后套用（它在训练集里是 100% 纯的）。
- **权重扫描**（在答案上评估，D 的流程 + 不同的 w）：

  | w | macro-F1 | benign 漏判 | suspicious 漏判 |
  | --- | --- | --- | --- |
  | 1（即 D） | 0.99748 | 593 | 9 |
  | 2 | 0.99947 | 115 | 10 |
  | 5 | 0.99979 | 41 | 10 |
  | **10** | **0.99996** | **0** | 10 |
  | 20 | 0.99996 | 0 | 10 |
  | 50 | 0.99979 | 0 | 50 |
  | 100 | 0.99957 | 0 | 102 |

  取 **w = 10**：告警需要约 91% 以上的把握，是一个好解释的整数阈值，
  而且落在 5–20 这一段效果很好的区间里。**注意：这次扫描用到了答案**，
  所以 E 在这份验证集上的成绩偏乐观；如果还有另一份独立的测试集，
  预期成绩会略低。
- **结果**：E（`artifacts/content/res_date_rule_bw10.csv`）macro-F1
  **0.99996**，201 万行只错 10 行，benign 漏判为 **0**。剩下 10 个都是
  suspicious 被判成 benign：Symantec DLP 9 行（训练集里没有这个来源），
  Cisco Duo 1 行（训练集里 Duo 只有 12 行 suspicious）。
- **产物**：`src/soc_baseline/decision.py`、`tests/test_decision.py`、
  `--benign-weight`、`artifacts/content/res_source_mask_bw10.csv` 和
  `res_date_rule_bw10.csv`。

## I13 · 去掉时间依赖：timefree 特征 + 防火墙判决规则（2026-10-07）

- **动机**：D/E 的成绩依赖日期规则。检查还发现 content 特征也没把时间去干净：
  脱敏器只改写了 epoch 的一部分，留下年份前缀（Meraki `165tok_cred` = 2022，
  VPC `167tok_cred` 与 `17CRED-tok_cred` 分属不同年份），Palo Alto 的日期剩下
  `tok_user/05/08`；C 模型词表里有 1,113 个这类 token。在同一格式的恶意记录上，
  content 文档能以 0.79（ASA）/ 0.92（Meraki）的准确率猜出年份，多数类基线
  只有 0.66 / 0.72。
- **做法**：
  1. `--feature-set timefree`（`features.py`）：脱敏 token 连同粘在一起的数字、
     以及所有数字串一律替换成 `0`，月份、星期、AM/PM、时区词替换成 `tok_cal`。
     文档里不再有任何数值，也不读时间戳列。
  2. `fw_action` / `fw_action_vendor`：与厂商无关的防火墙判决
     （deny/drop/reject/blocked → block；accept/allow/permit → allow），并与
     “厂商字段是否为空”组合。只认独立出现的词，或 action/act/decision/
     disposition/verdict 这几个键的值；键名被脱敏的（`HOST-0121=BLOCKED`）
     视为未知键，照样计入。其他键的值不算：train 里无厂商 benign 中出现的
     block 词全部来自 HTTP 头 `X-XSS-Protection: 1; mode=block` 和
     `X-Frame-Options: DENY`。
  3. `scripts/apply_verdict_rule.py`（逻辑在 `verdict_rule.py`）：只从 train
     学出“无厂商 + block 判决 → malicious”，并检查支持数（≥ 1,000 行）和纯度
     （≥ 0.999），然后把测试集中这一格的行改判为 malicious。不用任何时间字段，
     用来替代日期规则。
- **为什么可行**（train，厂商状态 × 判决 × 标签）：

  | 厂商 / 判决 | benign | malicious | suspicious |
  | --- | --- | --- | --- |
  | 缺失 / block | 3 | **79,968** | 0 |
  | 缺失 / none | 238,919 | 31,760 | 0 |
  | 缺失 / allow | 8,377 | 0 | 0 |
  | 有 / block | 411 | 0 | 44,681 |
  | 有 / none 或 allow | 1,652,013 | 0 | 739 |

  同一条 ASA `Deny` 日志，厂商为空时是 malicious，带 `Cisco/ASA Firewall` 时
  是 suspicious。剩下 3 行 benign 是 Windows 报错正文里的 “is denied”。
  测试集中“缺失 / block”有 13,998 行，全部早于 07-26；晚期的无厂商行里
  没有一行带 block 判决。
- **结果**（valid 2,014,052 行，`artifacts/answer_scores.json`）：

  | 版本 | 是否用时间 | macro-F1 | 错误行数 | malicious F1 |
  | --- | --- | --- | --- | --- |
  | C content + 掩码 | 有残留（见上） | 0.8649 | 8,572 | 0.604 |
  | F0 timefree，无判决特征 | 否 | 0.9001 | 7,096 | 0.712 |
  | F timefree + 掩码 | 否 | 0.9017 | 7,014 | 0.717 |
  | **G F + 判决规则** | **否** | **0.99598** | **864** | 0.998 |
  | G + benign 权重 10 | 否 | 0.99895 | 150 | 0.998 |
  | D C + 日期规则（对照） | 是 | 0.99748 | 602 | 1.000 |
  | E D + benign 权重 10（对照） | 是 | 0.99996 | 10 | 1.000 |

- **时间无关性验证**（`scripts/check_time_independence.py`，结果在
  `artifacts/timefree/time_independence.json`）：
  - 反事实：14,052 条早期行加 20 万条随机晚期行，时间戳全部改到 2024-07-27，
    消息里每个数字随机替换，月份、星期名改成 Jul/Fri，再用 G（模型 + 规则）
    重新预测：**0 行**改变。
  - 年份探针：content 0.792 / 0.915 → timefree 0.689 / 0.737（多数类
    0.657 / 0.718）。残余的 2–3 个点来自不同年份出现的接口名、协议组合不同
    （timefree 下 ASA 恶意只剩 47 种不同文档），不是时间值；反事实测试已证明
    改动任何时间值都不改变预测。
- **G 剩下的 864 个错误**：
  - 54 条早期 malicious 是 2022 年的 winlogbeat / Duo 记录，没有判决词，
    内容与正常日志无异，不靠时间分不出来；
  - 801 条 Crowdstrike benign 被判成 suspicious（D 为 593；训练集里 Crowdstrike
    只有 35 行且全是 suspicious，不受掩码约束）；
  - 9 条 Symantec DLP suspicious 被判成 benign（训练集里没有这个来源）。
- **结论**：
  1. 不依赖时间的推荐提交是 **G**（`artifacts/timefree/res_verdict_rule.csv`）。
     如果业务上 benign 漏判代价更高，可交 G + w10
     （`artifacts/timefree_bw10/res_verdict_rule_bw10.csv`），但 w = 10 是 I12
     对着答案扫出来的，分数偏乐观。
  2. 模型里的判决特征单独贡献很小（F0 → F 只多 82 行）：线性词袋模型遇到训练
     没见过的 Palo Alto 格式时，`traffic`、`inbound`、`0/0/0` 等词把它拉向 benign，
     判决 token 压不过，所以需要显式规则。Palo Alto 的 5,826 行全部靠规则判对。
  3. 判决规则和日期规则一样利用了数据集的构造方式（历史攻击记录导出时没带
     厂商元数据）。它不依赖时间，但依赖“厂商字段为空”这一来源特征，只在比赛
     数据上成立，不是能迁移到真实 SOC 的安全信号。
  4. **这不是盲测**：规则的思路来自 I9/I11 对 valid 早期行的分析（Palo Alto drop、
     VPC REJECT 被漏判）。规则本身只从 train 学出，阈值（≥ 1,000 行、纯度 ≥ 0.999）
     在评分前就已固定，但设计时已经知道 valid 的样子。
- **产物**：`src/soc_baseline/features.py`（`timefree`、`firewall_action`）、
  `src/soc_baseline/verdict_rule.py`、`scripts/apply_verdict_rule.py`、
  `scripts/check_time_independence.py`、`tests/test_verdict_rule.py`、
  `artifacts/timefree/`、`artifacts/timefree_bw10/`、`artifacts/timefree_ablation/`。

## I14 · 在外部数据集上抽样评测时间无关版本（2026-10-07）

- **做法**：`scripts/eval_external.py`。v4、latest、v2 live 各分层抽 30 万行，
  v2 incident 随机抽 5 万行（与 I10 同一随机种子）；另外单独报告原始消息
  从未出现在 train 里的“novel”行。比较 B/C/D（content 系列）和
  F_nomask/F/G/G+w10（timefree 系列）。结果在 `artifacts/external_eval/`。
- **数据差异**（全量统计，无厂商 + block 判决 一格）：v4 为 93,966 malicious 对
  3 benign，latest 为 118,586 对 3，两者的 malicious 全部早于 07-26，与比赛数据
  构造相同。v2 中这一格在 live 里是 0 malicious 对 641 benign，在 incident 里
  只有 3 行，因为 v2 给历史攻击记录补上了厂商（ASA、VPC、PAN、Meraki）。
- **结果**（macro-F1；括号内为 malicious 召回）：

  | 版本 | v4 | v4 novel（2,042 行，全 M） | latest | v2 live | v2 incident |
  | --- | --- | --- | --- | --- | --- |
  | B content | 0.967 (0.913) | (0.223) | 0.961 (0.920) | 0.749 (0.222) | (0.756) |
  | C B + 掩码 | 0.988 (0.936) | (0.432) | 0.985 (0.921) | 0.655 (0) | (0.030) |
  | D C + 日期规则 | 1.000 (1.000) | (1.000) | 0.9996 (1.000) | 0.655 (0) | (1.000) |
  | F_nomask timefree | 0.977 (0.928) | (0.358) | 0.973 (0.923) | 0.656 (0.003) | (0.303) |
  | F timefree + 掩码 | 0.991 (0.952) | (0.568) | 0.990 (0.946) | 0.655 (0) | (0.030) |
  | **G F + 判决规则** | **0.9999 (1.000)** | **(0.998)** | **0.9995 (1.000)** | 0.655 (0) | (0.030) |
  | G + w10 | 0.9999 (1.000) | (0.998) | 0.9999 (1.000) | 0.655 (0) | (0.030) |

  latest 的 malicious 和 suspicious 消息都不在 train 里（脱敏命名空间不同），
  latest/novel 的结果与全量一致。v4 novel 就是比赛 valid 的那 14,052 条早期行
  中的一部分，不是独立数据。
- **结论**：
  1. 在按比赛方式构造的数据上（v4、latest），G 不用时间也和日期规则 D 持平；
     换了脱敏命名空间（latest）也不受影响。
  2. 在 v2 上所有时间无关版本的 malicious 都基本为 0（live 0，incident 3%，只有
     Cisco/Meraki 判对）。v2 去掉了比赛数据的两个构造痕迹：live 里攻击与正常
     不再按时间分开，incident 记录带上了厂商；判决规则不会触发，来源掩码还
     禁止 ASA/VPC 判 malicious。D 在 incident 上的 100% 只是因为历史攻击记录
     全在 07-26 之前，在 live 上同样是 0。
  3. **I10 中 B 在 v2 上“泛化最好”主要来自时间残留**：v2 的每一条 ASA 记录
     （incident 和 live、所有标签）syslog 头都带字面年份，content 文档里变成
     `tok_year`；而 B 的 malicious 对 suspicious 最强权重中就有 `tok_date tok_year`、
     `tok_year tok_time`、`tok_year`（train 里只有历史 malicious 记录头部带字面年份，
     07-26 采集的 suspicious 年份被脱敏成 `USER-9546`）。所以 B 把 v2 的 ASA 普遍
     推向 malicious：incident 召回 0.78，但 live 中 371 条 suspicious 也被判成
     malicious，live malicious 只对一半。timefree 把两种头部都变成
     `tok_cal 0 0 0:0:0`，这一信号就消失了。I10 的结论“v2 的 malicious 单条
     日志基本不可学”更加确定。
  4. 已知问题，未修：v2 live 中有 641 条 CloudTrail JSON 的
     `"logStreamName":"eni-…-reject"` 被判决提取器当成 block，在无厂商行上触发
     规则，抽样中多出约 100 个误报。修复方法是不把与 `-`、`_` 相连的词当作判决，
     但这是看了外部测试数据之后才发现的，修完之后 v2 live 就不再是独立测试。
- **产物**：`scripts/eval_external.py`、`artifacts/external_eval/results.json`、
  `artifacts/external_eval/malicious_recall_by_source.csv`。

## I15 · 本地 LLM 零样本推理与路由（2026-10-07）

- **动机**：TF-IDF 模型只认识 train 里出现过的日志格式。G 在 valid 上的 864 个错误中，
  801 个是 Crowdstrike 资产记录：train 里 Crowdstrike 只有 35 行，而且全是 suspicious。
  这次试验想知道：让开源 LLM 先读懂日志、推理一遍再给标签，能不能处理格式不同的日志。
- **环境**：单张 T4（15 GB，不支持 bf16 和 FlashAttention2）。模型用 Qwen3.5-4B
  （2026-02 发布，fp16 权重约 8 GB）。更新的 Qwen3.8-27B、GLM-5.3 等单卡放不下。
  - 权重从 ModelScope 直连下载，约 80 MB/s；默认代理不到 1.4 MB/s。
  - vLLM 0.19 在 T4 上要设 `max_num_seqs=32`、`max_num_batched_tokens=1024`、
    `gpu_memory_utilization=0.80`。原因是 Qwen3.5 线性注意力层的每序列状态，以及
    prefill 时的临时显存，vLLM 的 profiling 估不到，用默认参数会 OOM。
  - 每次启动要做 kernel autotune，第一次约 6 分钟。
- **做法**：`scripts/llm_reasoning_probe.py`，零样本。
  - prompt 里只有三个标签的书面定义，没有任何训练样例；输入是结构化字段加消息正文。
    正文里的日期、时间、epoch、年份全部掩码，IP 和端口保留。
  - 要求模型先简短推理，再输出一行 JSON，字段为 label、verdict、event_type、confidence。
  - 探针样本共 864 行：
    - train：每个（来源, 标签）格子先按 timefree 模板去重，再补充其他不同的消息。
      每格 benign 取 12 行、suspicious 40 行、malicious 120 行，无厂商 benign 取 40 行。
    - valid：选 TF-IDF 没见过或很少见的格式，包括 Palo Alto TRAFFIC、Crowdstrike、
      Graph、DLP、Duo，再加上无厂商的行。
    - valid 答案只用来抽样和打分。
  - 两种模式都跑了一遍：
    - thinking：用官方推荐的 thinking 采样参数，生成上限 6,144 token。
    - no-thinking：不开 thinking，模型只在回答里写几句推理。
- **探针结果**（valid 各组的准确率；括号内是“是否判为告警”的准确率）：

  | 分组 | 行数 | no-thinking | thinking | G |
  | --- | --- | --- | --- | --- |
  | Crowdstrike benign（资产记录） | 60 | **1.00** | 0.98 | 0.00 |
  | Crowdstrike suspicious（检测） | 60 | 0.60（1.00） | 0.80（0.93） | 1.00 |
  | Palo Alto drop（malicious） | 60 | 0.00（1.00） | 0.00（1.00） | 1.00 |
  | 无厂商的其他 malicious | 60 | 0.02（0.92） | 0.00（0.93） | 0.87 |
  | Graph / Duo / 无厂商 benign | 90 | 1.00 | 1.00 | 1.00 |
  | Symantec DLP（suspicious） | 5 | 0.00 | 0.00 | 0.00 |
  | **valid 合计，是否判为告警** | 336 | **0.970** | 0.961 | 0.783（F 0.569） |

  - train（零样本）：
    - 15 个 benign 来源中有 11 个准确率不低于 0.92。
    - ASA、VPC、Precinct、Duo 的 suspicious 全部判对。
    - malicious 一条也没判出来：带拦截判决的 64 条全判成 suspicious，不带判决的 56 条
      Meraki flow 记录全判成 benign。
  - LLM 自己抽取的 verdict 和 `firewall_action` 正则基本一致：正则判为 block 的
    268 行中，LLM 也判为 block 的有 265 行。
- **thinking 与 no-thinking 对比**：

  | | no-thinking | thinking |
  | --- | --- | --- |
  | 每条平均生成 token | 171 | 2,916 |
  | 864 行耗时 | 28 分钟 | 约 3 小时 |
  | 因超长被截断、没给出答案 | 1 行 | 14 行 |
  | train 三分类准确率 / 告警准确率 | 0.695 / 0.843 | 0.706 / 0.837 |
  | valid 三分类准确率 / 告警准确率 | 0.560 / 0.970 | 0.589 / 0.961 |

  两种模式的预测 92.7% 一致。thinking 改善了 AD suspicious（0.80 → 0.98），但更容易
  把 benign 判成告警（Barracuda WAF 1.00 → 0.83，Windows Logs 1.00 → 0.92）。
  它对 malicious 和 DLP 都没有帮助。
- **路由 R**（`route` 子命令）：
  - 规则只依据 train：训练行数少于 100 的来源（也就是 G 的来源掩码不做限制的那些）
    交给 no-thinking 的 LLM，其余行仍用 G。
  - LLM 判出的 malicious 改成 suspicious。依据是 train 探针中，LLM 判为 malicious
    的全是 Crowdstrike 检测记录，而这些记录的标签是 suspicious。
  - valid 中被路由的有 1,303 行：Crowdstrike 892、Graph 200、Barracuda ESS 168、
    Apache 34、DLP 9。LLM 推理耗时约 1 小时。结果在 `artifacts/llm_probe/route_scores.json`：

  | 版本 | 是否用时间 | macro-F1 | 错误行数 |
  | --- | --- | --- | --- |
  | G | 否 | 0.99598 | 864 |
  | G + benign 权重 10（w=10 是对着答案扫出来的） | 否 | 0.99895 | 150 |
  | **R = G + LLM 路由** | 否 | **0.99916** | **99** |

  - 改善来自 Crowdstrike：892 行全部判对，包括 G 判错的 801 行 benign 和 91 行 suspicious。
  - R 剩下的 99 个错误：
    - 54 条早期 malicious：没有判决词，和 G 相同。
    - 9 条 DLP：正文其实是 winlogbeat 的进程创建日志。
    - 36 条 Barracuda ESS benign 被判成 suspicious：都是被拦截的垃圾邮件
      （`"blocked":true`，taxonomy=spam）。按 prompt 的定义，“安全控制拦截了东西”
      就是 suspicious，但答案标的是 benign。G 在这 36 条上是对的。
- **结论**：
  1. LLM 零样本就能跨格式判断“是不是告警”。它对新格式的识别正好补上 TF-IDF 的盲区，
     在 valid 的格式迁移组上，告警准确率 0.97，G 只有 0.78。
  2. malicious 仍然判不出来。比赛的 malicious 指无厂商的历史攻击记录，由数据集的构造
     方式决定，单条日志的内容里没有这个信息。DLP、ESS 垃圾邮件也属于标签定义和内容
     对不上的情况。这些都不是推理能力不够，thinking 也解决不了。
  3. 单条日志分类不值得开长推理：token 多 17 倍，准确率基本持平，还会有截断。
  4. 推荐的组合是 R：已知格式用 G，稀有或未见过的来源交给 LLM。R 不依赖时间，也不需要
     对着答案调权重，错误数比 G + w10 还少。但 LLM 不能处理全部行：在已知的大流量格式上，
     它有几个百分点的误判（例如 ESS），放到全量数据上会变成成千上万行错误。
  5. **这不是盲测**：路由的思路来自 I11/I13 对 Crowdstrike 的分析，以及这次探针在 valid
     上的结果。不过路由阈值沿用了来源掩码原有的 100 行，malicious → suspicious 的映射
     只依据 train，prompt 也没有因为 valid 的结果改过。
  6. **下一步候选**：
     - 从 train 里归纳标签定义，或者从 train 取 few-shot 示例。例如 train 里 74 行 ESS
       全部是 benign，其中 12 行 taxonomy=spam，可以把“垃圾邮件过滤属于例行操作”写进定义。
     - 把同一主机或会话的邻近事件拼进 prompt，在 v2 live 上评估跨事件上下文。
     - 在格式差异更大的外部数据上评估路由。
- **产物**：
  - `scripts/llm_reasoning_probe.py`
  - `artifacts/llm_probe/`：
    - `sample.parquet`
    - `Qwen3.5-4B/`：thinking 模式
    - `Qwen3.5-4B-nothink/`：含 `res_route.csv`
    - `route_scores.json`
    - `run_*.log`

## I16 · 分类器 + LLM 混合方案：只用 train、防泄漏、benign 优先（2026-10-08）

- **动机**：重新设计一套“分类器 + LLM”的方案，不借助 I1–I15 的任何模型、规则、掩码或预测：只用 train
  训练，在 valid 上验证，避免数据泄漏，优先保证 benign 不被漏判。完整的调研报告（含消融与解释）：
  https://claude.ai/code/artifact/d297228a-b907-42ea-8b9e-6af8df21e987
- **防泄漏**（只用 train 审计，`artifacts/hybrid/audit.json`）：时间戳、event_id、被脱敏器写进伪名的年份
  （单独出现的 `USER-9564` 在 51,756 行 malicious、385 行 benign 上）都与标签完全对齐。分类器视图把伪名
  换成类型、数字串换成 0、屏蔽月份和星期词；LLM 视图只屏蔽时间和伪名。对 20,000 条消息随机改写数字和
  月份后，分类器文档 0 处变化。
- **train 内部验证**：文档 5 折（ud5）、来源留出（18 折）、(来源, 标签) 格子留出（10 折）；第二轮起加入
  LSA 聚类 5 折（cl5，300 簇，留出文档的新颖度分布接近 valid 输入）。LLM 在留出上用去掉被留出格子条目的
  编码手册评估。配置按“0.99 × 同分布代价 + 0.01 × 分布外代价”在每折同权和按文档合并两种口径下取 minimax。
- **最终方案（第四轮）**：TextCNN 与 TF-IDF（按行加权）的概率平均，benign 权重 w = 10；新来源优先、按
  新颖度（LSA 最近邻余弦）取 2,000 条交给 Qwen3.5-4B（零样本，train 编码手册，读标签 token 概率）；LLM 以
  10 倍阈值决定 benign 还是告警，告警类型和 malicious 判定来自分类器。
- **valid 结果**：

  | 轮次 | 要点 | macro-F1 | benign 漏判 | 错误行数 |
  | --- | --- | --- | --- | --- |
  | 1 | TF-IDF 文档加权，w = 300、wl = 300，train 网格（ud5） | 0.839 | 0 | 9,999 |
  | 2 | TextCNN，w = 100，train 网格（cl5） | 0.843 | 0 | 8,994 |
  | 3 | TextCNN，w = wl = 10，新颖度路由 | 0.873 | 0 | 7,751 |
  | 4（最终） | TextCNN + TF-IDF 平均，w = wl = 10 | 0.870 | 0 | 7,884 |

  只有第一轮是严格的留出结果；第二到四轮都受到前一轮 valid 结果的启发，参数仍然只来自 train 或事先定下的
  代价比。最终方案的 LLM 改了 1,108 行，全部正确。
- **结论**：
  1. train 内部的网格搜索会选出很高的 benign 权重（来源和格子留出把 benign 漏判按 10 倍计），而 valid 里
     漏掉的 malicious（无厂商的 Palo Alto TRAFFIC drop、VPC REJECT）在分类器上只有 0.33–0.43 的 malicious
     概率，高权重把它们压成了 benign。
  2. LLM 在被送对地方时很有效：按分类器置信度路由（事后）时它改对 7,366 行，macro-F1 0.979，代价是 34 行
     benign 漏判（全是 ASA 的 “denied by ACL” 记录）；按新颖度路由只覆盖分类器 13% 的错误；随机路由几乎没有
     收益（代价 9,016，只用分类器是 8,992）。两种路由信号对应两种偏移，组合路由需要新的测试集验证。去掉 train
     编码手册后，LLM 补报的告警从 1,108 降到 884（macro-F1 0.865）。
  3. benign 优先必须在两侧同时成立：LLM 侧不用 10 倍阈值会多 1,411 行 benign 误报；换成 Qwen3-4B-Instruct-2507
     即使阈值相同也多 1,986 行。Qwen3.5-4B 在 T4 上约 43 条/分钟，是路由预算的瓶颈：同样的配置把预算放到
     3,000、4,000 条（事后），macro-F1 升到 0.930、0.951，benign 漏判仍为 0。
- **产物**：
  - `src/soc_hybrid/`：`text.py`（两种视图）、`audit.py`、`splits.py`、`models.py`（TF-IDF LR、TextCNN）、
    `holdout.py`、`novelty.py`、`llm.py`、`llm_holdout.py`、`route_select.py`、`classifier_select.py`、
    `pipeline.py`、`evaluate.py`、`ablation.py`、`ablation_leak.py`、`report.py`、`llm_only.py`
  - `tests/test_hybrid.py`
  - `artifacts/hybrid/`：`audit.json`、`holdout/`、`route_select/`、`runs/final4/res.csv`（最终配置在 valid 上的预测）、
    各消融运行、`llm/*.jsonl`（LLM 回复缓存）

## I17 · 评分规则更正：威胁优先 + 部分分，重新选型（2026-10-08）

- **动机**：评分规则更正为同时考察威胁发现和三分类细分：威胁（suspicious、malicious）判成 benign 惩罚最重，
  suspicious 与 malicious 互判给部分分。I12 和 I16 的 benign 优先针对的是相反的错误。具体分值没有给出，代价矩阵
  写成 [[0, 1, 1], [m, 0, 0.5], [m, 0.5, 0]]（行为真实标签），m 取 2、5、10。报告按新规则改写（同一链接），
  旧规则的四轮移到附录。
- **决策与融合**（`decision.py`）：`min_cost` 按分类器概率取期望代价最小的标签；`cost_confidence` = 次小 /
  （最小 + 次小）期望代价，用于置信度路由。`fuse_codes` 有三种融合：`gate_km`（LLM 以自己的 m 判 benign 或告警）、
  `raise`（只能补报告警）、`mix_km`（LLM 与分类器的概率按比例平均后再判）；三者都保留分类器的 malicious 判定和告警类型。
- **选择**（`soc_hybrid.cost_select`，只用 train）：沿用 I16 的框架（0.99 × cl5 + 0.01 × LOSO/LOCO，两种口径），
  把 m 的三档也放进 minimax，共 6 个倍数。
  1. 只看分类器（7 个候选 × 8 个决策 m）：cl5 上约 0.1% 的漏报是高置信的（AD、Crowdstrike 的 suspicious，所在簇被
     整簇留出，威胁概率中位数 0.003），调阈值救不回来；提高决策 m 主要压低 LOSO/LOCO 的漏报。
  2. 路由 × 融合（3 个分类器 × 5 个决策 m × 触发条件 × 融合，共 9,180 种配置）：先为新触发条件在 cl5 上补了 245 条
     LLM 回答（`cost_select picks` → `llm_holdout --keep-previous --cl5-extra`）。选出 TextCNN + TF-IDF 按行加权平均、
     决策 m = 2、置信度 < 0.9 或新 (来源, 标签) 组合就送 LLM、`mix_km(0.75)`；最坏倍数 1.09，不用 LLM 时最好的是 1.83。
     cl5 路由上限取 1% 或 2% 选出同一配置。
- **valid 结果**（`runs/cost1/valid_pred.csv`，配置选定后只运行一次）：

  | 方案 | 威胁漏报 | 误报 | 互判 | 代价 m = 2 / 5 / 10 | macro-F1 |
  | --- | --- | --- | --- | --- | --- |
  | cost1（最终） | 239 | 34 | 24 | 524 / 1,241 / 2,436 | 0.9967 |
  | cost1 只用分类器 | 430 | 34 | 24 | 906 / 2,196 / 4,346 | 0.9943 |
  | I16 第四版（final4） | 7,884 | 0 | 0 | 15,768 / 39,420 / 78,840 | 0.8702 |

  LLM 改了 191 行，全部是把 benign 改成告警，全部正确（被路由的 8,733 行里分类器错 199 行，融合后剩 8 行）。剩下的
  230 行 malicious 漏报来自无厂商字段的 Palo Alto 格式流量日志、fqdn 拦截记录等，其中 220 行触发了置信度条件，但排在
  2,000 条预算之后。
- **消融**（`artifacts/hybrid/ablation_cost.json`，m = 10 的代价）：随机路由 4,424（比只用分类器的 4,346 还差）；告警类型
  交给 LLM 6,702（互判 8,536 行）；预算 1,000 条 3,076；Qwen3-4B-2507 2,376；通用 prompt 2,436（164 条 LLM 标签不同，
  最终结果不变）；分类器决策 m = 3 时 1,079、m = 10 时 9,539；只用 TF-IDF 6,507。事后更好的两个变体：按新颖度排序 436、
  只用 TextCNN 561，按协议都选不出来。全部 7,521 条触发文档都送 LLM（事后）：390（漏报 34、误报 36、互判 28；LLM 多跑约 100 分钟，改动的 398 行里 2 行改错），说明路由条件是对的、卡住的是预算。去掉防泄漏的 TF-IDF
  （决策 m = 2）在 valid 上漏报 1,564 行，防泄漏版 6,954 行：valid 的 malicious 同样集中在更早的日期；把 valid 的时间戳改成
  同一时刻，它有 5,080 行预测会变。
- **结论**：
  1. 决策规则要对准评分：同一个分类器从 benign 优先（w = 10）改为最小代价（m = 2），valid 漏报从 8,992 行降到 430 行，
     误报只多 34 行。
  2. 新规则下 LLM 的价值在于补抓分类器犹豫的威胁，但必须保留否决能力（train 上只能补报时倍数 3.66），也不能决定告警类型。
  3. 剩余误差主要由 LLM 预算和路由排序决定。更快的 Qwen3-4B-2507 在新规则下不比 Qwen3.5-4B 差，同样的时间能送约 8 倍的
     文档。组合排序（置信度 + 新颖度）和更大的预算要在新数据上验证后再采用。
- **产物**：`src/soc_hybrid/cost_select.py`；`decision.py`（`cost_matrix`、`min_cost`、`cost_confidence`、`fuse_codes`）；
  `metrics.py`（威胁漏报、误报、互判、三档代价）；`pipeline.py`（`--miss-weight`、`--fusion gate_km|raise|mix_km`、
  `--fusion-param`）；`ablation.py`（`derived_cost_variants`）；`ablation_leak.py`（最小代价决策）；
  `artifacts/hybrid/cost_select/`、`runs/cost1/`、`runs/cost_abl_*`、`run_ablations_cost.sh`。最终配置的命令：

  ```bash
  python -m soc_hybrid.pipeline --name cost1 --classifier textcnn --weights rows \
      --params '{"device": "cuda"}' --blend tfidf_word_lr:rows --miss-weight 2 \
      --doubt 0.9 --pair --unseen-source --fusion mix_km --fusion-param 0.75 \
      --budget 2000 --priority doubt
  ```

## I18 · 取消 LLM 的时间限制：不设路由上限（2026-10-09）

- **动机**：时间限制取消，不确定的文档全部交给 LLM，不设上限。2,000 条上限原本是 T4 的时间预算（Qwen3.5-4B 约 43 条/分钟），
  train 侧对应的“cl5 送 LLM 的文档不超过 2%”这条约束随之去掉。
- **train 复查**（`cost_select`）：在 cl5 的四个区间（置信度 0.9–0.95、0.95–0.99，新颖度 0.8–0.9、0.9–0.95）均匀抽样补了
  838 条 LLM 回答，cl5 合计 1,925 条；没有回答的路由文档改为按置信度分层折算；网格加入置信度阈值 0.95、0.99，共 15,420 种
  配置。原 minimax 规则选出 TextCNN + TF-IDF 文档加权、m = 2、置信度 < 0.99、`gate_km(1)`（cl5 送 7.5%，倍数 1.42），但它的
  优势来自 LOCO 里只有 12 条文档的 Duo suspicious 折（置信度 0.9 时都没路由，0.99 时全被 LLM 抓回），而且随候选集合变化。
  `cost_select robust` 在不让小折主导的汇总方式下（去掉 30 条以下的折、按文档数开方加权）选出的都是“置信度 < 0.9 或新组合”
  这套触发条件，原配置与最优只差不到 1%；只看按文档合并时才偏向 0.99。所以触发条件不变，只去掉上限（`--budget 0`）。
- **valid 结果**（`runs/cost2/valid_pred.csv`）：7,521 条文档、18,967 行送 LLM，覆盖分类器 488 个错误中的 478 个。威胁漏报 34、
  误报 36、互判 28，代价 m = 2 / 5 / 10 = 118 / 220 / 390，macro-F1 0.9991（2,000 条上限时漏报 239，代价 524 / 1,241 / 2,436）。
  LLM 补报 398 行告警，392 行完全正确，4 行从漏报变成互判，2 行误报（Barracuda WAF）。这个配置此前作为事后消融
  （`cost_abl_all_triggered`）已经跑过，结果相同。
- **消融**（以 cost2 为基准，`artifacts/hybrid/ablation_cost2.json`，m = 10 的代价；最终方案 390）：去掉 LLM 4,346；随机送同样多的
  7,521 条 4,875（漏报 414、误报 723，比不用 LLM 还差）；预算 2,000 条 2,436、1,000 条 3,076；Qwen3-4B-2507 1,091（漏报 12，但误报
  949，2,000 条上限下看不出的问题）；通用 prompt 524（误报 170）；告警类型交给 LLM 4,846（malicious 判成 suspicious 8,920 行）；
  LLM 概率占比 0.25 638；去掉否决 392（valid 上 LLM 从未否决）；分类器决策 m = 1、3、5 为 400、392、393，m = 10 为 416；只用 TF-IDF
  3,300（互判 2,692），只用 TextCNN 390。不设上限后，判别线和是否做两个分类器的平均都不再敏感，LLM 本身（型号、编码手册）的
  误报率成了主要差别。
- **产物**：`cost_select.py`（区间抽样、分层折算、`robust`）、`pipeline.py`（`--budget 0`）、`runs/cost2/`、
  `run_ablations_cost2.sh`、`cost_select/routing_grid_i17.parquet`（I17 的 2% 约束网格）。最终配置的命令：

  ```bash
  python -m soc_hybrid.pipeline --name cost2 --classifier textcnn --weights rows \
      --params '{"device": "cuda"}' --blend tfidf_word_lr:rows --miss-weight 2 \
      --doubt 0.9 --pair --unseen-source --fusion mix_km --fusion-param 0.75 --budget 0
  ```

## I19 · 外部数据集抽样测试（2026-10-09）

- **方法**（`python -m soc_hybrid.external --rows 10000`）：WitFoo 四组外部数据各随机抽 10,000 行（种子 0），完整走最终方案 cost2
  （train 上拟合的分类器、相同的触发条件、不设上限、同样的 prompt 和融合），标签只用于评分。每组另报两个新颖子集：原始正文
  在 train 里没出现过（I14 的口径），以及分类器文档在 train 里没出现过（更严）。
- **结果**：

  | 数据组 | 样本标签（benign / suspicious / malicious） | 威胁漏报 | 只用分类器时漏报 | 误报 | 互判 | macro-F1 | 送 LLM 的文档 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | v4 | 9,137 / 239 / 624 | 0 | 1 | 0 | 1 | 0.9990 | 41 |
  | latest | 9,031 / 234 / 735 | 0 | 2 | 0 | 0 | 1.0000 | 154 |
  | v2_live | 9,400 / 557 / 43 | 0 | 0 | 0 | 43 | 0.6543 | 129 |
  | v2_incident | 0 / 0 / 10,000 | 0 | 0 | 0 | 9,701 | 0.0581 | 450 |

- **结论**：
  1. 威胁发现能泛化：40,000 行里威胁零漏报、零误报；LLM 改了 3 行，正好补回分类器漏掉的 3 行威胁。
  2. 三分类细分不能泛化到 v2：v2_live 的 43 行和 v2_incident 的 9,701 行 malicious 都被判成 suspicious。v2 的 malicious 按事件关联
     原地标注、带着厂商字段，而按比赛数据的习惯，带厂商的拦截记录是 suspicious；I14 的时间无关基线在 v2 上同样没有 malicious 召回。
  3. v4 和 latest 主要是 train 的事件本身：v4 的新颖行只有 78 行（全是 malicious，都报了告警）；latest 重新脱敏后有 8,175 行的分类器
     文档是新的，结果仍然一行不错。
- **产物**：`src/soc_hybrid/external.py`、`artifacts/hybrid/external/`（`summary.json`、各组逐行结果）、`llm/external.jsonl`。

## I20 · v2 上 malicious 全判成 suspicious 的原因（2026-10-09）

- **问题**：I19 中四组外部数据都是威胁零漏报、零误报，但 v2_live 的 43 行、v2_incident 的 9,701 行 malicious 都判成了 suspicious。
- **方法**（`python -m soc_hybrid.v2_failure`，不读 valid）：读外部数据自带、比赛数据没有的打标字段（`lifecycle_stage`、
  `matched_rules`、`incident_ids`）；把 v4 的 malicious 和 v2 的 incident_signals 按（秒级时间戳, 源端口, 目的端口）配对，
  在配对事件上交换报文格式和厂商字段，交给最终方案的分类器；在 v2 自己的标签上训练 TF-IDF + 逻辑回归，检验单条记录能否
  区分 malicious 和 suspicious。
- **结果**：
  1. 所有版本的标签都由 Precinct 的关联结果决定：没有规则命中 = benign；命中检测规则（几乎都是 Blocked Action）= suspicious；
     被关联进事件 = malicious。v2 live 中 malicious 就是 `incident_ids` 非空的记录（7,728 行对 0 行）。
  2. 差别在导出格式。v4 和 latest 的 malicious 100% 没有厂商字段、没有规则命中记录，suspicious 100% 有厂商；train 同样
     （111,728 行对 0 行）。v2 的 malicious 带厂商、命中 Blocked Action，格式和 suspicious 相同。
  3. 同一批事件：v4 的 125,780 行 malicious，时间戳全部出现在 v2 incident_signals 里；配出 20,365 对一一对应的事件
     （ASA 12,705、Meraki 5,831、PAN 1,507、VPC 281），时间、端口和报文模板相同，v4 去掉了厂商并多换了几处伪名。
  4. 交换实验（判成 malicious 的比例）：

     | 报文 | 厂商字段 | 全部 | ASA | Meraki | PAN | VPC |
     | --- | --- | --- | --- | --- | --- | --- |
     | v4 | 没有（v4 原样） | 99.9% | 100% | 100% | 98.3% | 100% |
     | v4 | 换成 v2 的 | 28.7% | 0% | 99.9% | 0% | 0% |
     | v2 | 去掉 | 100% | 100% | 100% | 100% | 100% |
     | v2 | 有（v2 原样） | 28.7% | 0% | 99.9% | 0% | 0% |

     报文格式几乎不起作用，厂商字段决定类型。Meraki 的 flows 记录在 train 里只以无厂商的 malicious 出现（32,596 行），模板本身
     带着标签，所以 I19 中 v2_incident 判对的 299 行里有 286 行是 Meraki。全量上，incident_signals 原样 96.9% 判 suspicious、
     去掉厂商 99.96% 判 malicious；v2 live 的威胁记录去掉厂商后，99.8% 的 suspicious 也判成 malicious。原样输入时 p(malicious)
     区分两类的 AUC 是 0.518（ASA 0.446、VPC 0.538）。
  5. 单条记录分不开。分类器文档在 v2 live 的 114,521 条威胁记录里只有 127 种，99.6% 的 malicious 与 suspicious 文档相同，
     按每种文档的多数标签最多判对 5.5%；LLM 的输入有 63,788 种，最多判对 36.5%。用 v2 的标签训练（07-26 训练、之后测试，
     malicious 占 7.3%）：AUC 0.857 / 0.883 / 0.888（分类器文档 / LLM 输入 / 原始字段含 IP），平均精度 0.24–0.29，精度 ≥ 50%
     时召回为 0，按 0.5 判 malicious 的互判都比全判 suspicious 多（分类器文档 4,892 对 4,468）。AUC 主要来自 ASA 的方向：
     outside 进来的拦截 malicious 占 34.9%，dmz-2 出去的占 0.4%。随机 80/20 划分的平均精度也只有 0.37–0.50。
  6. 只用此前记录计算的 IP 对 / 源 / 目的出现次数，在 ASA 入站拦截内 AUC 0.56–0.65，VPC 内 0.44–0.48。live 的 1,584 个事件
     中位数只有 1 行（90% 不超过 4 行），只有 10 个出现在 incident_signals 里；malicious 中 5,779 行处置为 Disrupted。
  7. 威胁发现能泛化，因为 benign / 威胁的分界是单条记录的规则命中：v2 live 中 action 为 block 的 114,310 行全是威胁；train 里
     ASA Deny 和 VPC REJECT 记录也全是威胁（78,748 malicious、44,679 suspicious）。
  8. 路由没触发：v2_live 的 43 行 malicious 置信度都 ≥ 0.983，(来源, suspicious) 在 train 里见过。告警类型本来由分类器决定，
     编码手册里 malicious 的描述就是“没有厂商和产品元数据”的拦截记录。
- **结论**：v2 上失效的是 malicious 的定义，不是模型的缺陷。suspicious / malicious 的分界是“是否被关联进事件”，单条日志里
  没有；比赛数据通过“没有厂商字段”把它写进了记录，分类器和编码手册学到的都是这条构造痕迹。泄漏审计挡住了时间和伪名，
  却把厂商字段当成了正常内容；去掉厂商字段也救不回来，只会把 suspicious 推成 malicious。valid 按 v4 的方式构造，这条
  捷径在那里成立；要在 v2 这类数据上细分，需要事件关联的结果。
- **产物**：`src/soc_hybrid/v2_failure.py`、`artifacts/hybrid/v2_failure/summary.json`。

## I21 · 训练时去掉厂商字段（2026-10-09）

- **问题**：I20 显示分类器靠“没有厂商字段”判 malicious。如果训练时就去掉厂商和产品字段，能不能学到可迁移的区分方法？
- **方法**（`python -m soc_hybrid.vendor_blind`，valid 由 `python -m soc_hybrid.evaluate --runs vendor_blind,cost2` 评分）：
  分类器文档去掉 `fvendor_*`、`fproduct_*` 两个标记，其余不变（TextCNN + TF-IDF 平均，按行加权，每格最多 20,000 个文档，
  m = 2 最小代价），重新训练后与原分类器比较，都不接 LLM（告警类型本来就由分类器决定）。外部数据用 I19 的抽样（每组 10,000 行，
  种子 0）和 v2 的全部威胁记录。这是事后实验，没有据此改动最终方案。
- **结果**：

  | 数据 | 版本 | 威胁漏报 | 误报 | 互判 | malicious 召回 | macro-F1 | 代价 m2 / m5 / m10 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | valid | 原分类器 | 430 | 34 | 24 | 96.8% | 0.9943 | 906 / 2,196 / 4,346 |
  | valid | 去掉厂商 | 84 | 91 | 8,892 | 36.2% | 0.8105 | 4,705 / 4,957 / 5,377 |
  | v4 抽样 | 原 / 去掉 | 1 / 1 | 0 / 0 | 1 / 46 | 99.7% / 93.6% | 0.9988 / 0.9570 | |
  | latest 抽样 | 原 / 去掉 | 2 / 1 | 0 / 0 | 0 / 65 | 99.7% / 94.0% | 0.9995 / 0.9405 | |
  | v2_live 抽样 | 原 / 去掉 | 0 / 0 | 0 / 74 | 43 / 55 | 0% / 23.3% | 0.6543 / 0.6938 | |
  | v2_incident 抽样 | 原 / 去掉 | 0 / 0 | 0 / 0 | 9,701 / 455 | 3.0% / 95.5% | 0.0581 / 0.9767 | |

  1. 改学的线索：train 的威胁里，去掉厂商后只有 5.9% 的 malicious 与 suspicious 文档相同。TF-IDF 最强的 malicious 特征是
     ASA 的方向和接口（`src outside`、`dst inside`、`group dmz`），train 里从 outside 进来的拦截 93.5% 是 malicious
     （77,740 对 5,418）；suspicious 一侧是出站方向（`src dmz`、`dst outside`）、`reject ok`（VPC）、认证失败和脱敏伪名的组合（`zcred zcred`、
     `group zuser`）。
  2. valid 变差：malicious 中 PAN 6,038 行、VPC 2,664 行在 train 里从没以 malicious 出现（PAN 完全没有），原分类器靠无厂商
     判对，去掉厂商后全部判成 suspicious；ASA 的 5,112 行判对 5,086 行。
  3. v2 没有变好：v2_incident 多是入站拦截，召回升到 95%；v2 live 中入站拦截只有 34.9% 是 malicious，模型把 5,805 行入站
     suspicious 中的 3,819 行判成 malicious，全部 114,521 条 live 威胁记录的互判从 7,735 升到 8,831，p(malicious) 的 AUC 0.47
     （ASA 0.914、VPC 0.571）；抽样中多出 74 条误报（benign 来源失去了厂商标记）。
  4. 威胁漏报反而下降（valid 430 → 84）：原分类器把 334 行无厂商的 PAN 记录判成 benign（train 里也有无厂商的 benign）。
     这是在 valid 上看到的，没有据此改方案。
- **结论**：去掉厂商字段，分类器换成学“入站拦截 = malicious”这类数据构成上的差异，比赛数据上细分大幅变差，v2 live 也更差，
  证实了 I20 的两难：单条记录里既没有比赛数据之外也成立的 malicious 线索，也没有 v2 的事件关联信息。
- **产物**：`src/soc_hybrid/vendor_blind.py`、`artifacts/hybrid/vendor_blind/summary.json`、`artifacts/hybrid/runs/vendor_blind/`
  （`docs.parquet`、`scores.json`）、`models/*_novendor.joblib`。
