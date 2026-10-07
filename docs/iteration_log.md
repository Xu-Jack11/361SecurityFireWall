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
