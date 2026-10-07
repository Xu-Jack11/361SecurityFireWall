# 权威公开安全数据集调研

检索日期：2026-07-09（2026-07-10 增补 WitFoo Precinct 6）

## 同源数据集：WitFoo Precinct 6（最高优先级）

比赛数据与公开的 **WitFoo Precinct 6** 数据集同源：字段名逐字一致
（`timestamp/pipeline/src_ip/dst_ip/src_port/src_host/dst_host/username/message_sanitized/product_name/vendor_name`）、
`label_binary` 三分类取值一致、脱敏 token（`USER-NNNN`/`ORG-NNNN`/`CRED-NNNN`）一致，
且其 2026 年 3 月历史版本（v4）的 `benign=1,899,723`、`suspicious=45,420`
与仓库 `data/train.parquet` 完全相同（仓库 malicious=111,728，为 v4 的
125,780 的子集）。标签定义见官方生成代码：suspicious = 命中 261 条检测规则
之一，malicious = 属于已确认安全事件，benign = 其余。

- 数据卡：<https://huggingface.co/datasets/witfoo/precinct6-cybersecurity>（2.1M signals，另有 graph_edges/graph_nodes/incidents 子集）
- 大版本：<https://huggingface.co/datasets/witfoo/precinct6-cybersecurity-100m>（84M signals）
- 生成/脱敏代码：<https://github.com/witfoo/dataset-from-precinct6>
- 许可：Apache 2.0

已下载到本仓库 `data/external/`（`data/` 被 git-ignore，文件不进 git 历史）：

| 本地文件 | 行数 | 列数 | 来源 | sha256 |
| --- | --- | --- | --- | --- |
| `witfoo-precinct6-signals-latest.parquet` | 2,100,363 | 33 | HF `main`（refs/convert/parquet，2026-07 下载） | `7937dfda…f9ae209` |
| `witfoo-precinct6-signals-v4.parquet` | 2,070,923 | 27 | HF 历史 commit `bd422e30`（2026-03 v4，与比赛数据同版本） | `d9b8c032…2c222c` |
| `precinct6-v2.1.0/signals.parquet` | 2,011,674 | 38 | HF tag `v2.1.0`（commit `96311c2e`，2026-09-22，2026-10 下载） | `20d01fd2…62c6a746c` |
| `precinct6-v2.1.0/incident_signals.parquet` | 238,511 | 38 | 同上 | `378f0a3f…8de404c6` |

标签分布对照：

| 版本 | benign | malicious | suspicious |
| --- | --- | --- | --- |
| 仓库 `train.parquet` | 1,899,723 | 111,728 | 45,420 |
| v4（2026-03） | 1,899,723 | 125,780 | 45,420 |
| latest（2026-07） | 1,899,587 | 155,520 | 45,256 |
| v2.1.0 `signals`（实时采集，原位标注） | 1,897,153 | 7,728 | 106,793 |
| v2.1.0 `incident_signals`（历史攻击） | 0 | 238,511 | 0 |

**使用注意：**

1. **泄漏风险**：比赛 `valid_input.parquet` 很可能与公开集重叠。任何合并训练
   前必须用 `message_sanitized` 哈希等方式对 train 和 test 两侧去重；利用公开
   标签"查答案"属于合规问题，先确认比赛规则是否允许外部数据。
2. latest 版在 2026 年 3–5 月间经过多次重新清洗/重标注，脱敏映射已变化，
   与仓库数据只有少量行字节级一致；做记录级对齐应使用 v4 版。
3. v2（2026-09 起）**重新定义了标签**：malicious 是实时采集中被事件关联命中的
   原位记录（主要是 Cisco ASA 和 AWS VPC Security），历史攻击单独放进
   `incident_signals`；token 注册表也重建了。它的标签体系和比赛数据（v1 系）
   不同，只适合做泛化测试（见 `docs/iteration_log.md` I10）。
4. 补少数类样本时按 vendor/product 过滤到与比赛数据一致的组合
   （suspicious 主要来自 Cisco ASA Deny 与 AWS VPC REJECT 日志）。

本仓库默认只使用比赛方提供的 `data/train.parquet` 训练 baseline。下面的数据集适合用于后续增强实验，例如预训练日志词表、构造攻击链特征、做跨域鲁棒性测试、扩展可视化案例或验证检测规则。不要直接把外部标签简单映射进比赛训练集；不同数据集的采集环境、字段、标签粒度和脱敏方式差异很大，直接混合可能降低线上效果。

## 优先推荐

| 数据集 | 权威来源 | 内容与规模 | 与本赛题的关系 | 使用建议 |
| --- | --- | --- | --- | --- |
| LANL Comprehensive, Multi-Source Cyber-Security Events | [Los Alamos National Laboratory](https://csr.lanl.gov/data/cyber1/) | 58 天企业内网脱敏事件，包含 Windows 认证、进程启停、DNS、网络流和 red-team 事件；官方页面说明总计约 16.48 亿事件。 | 非常接近 SOC 多源日志场景，覆盖身份、主机、DNS、网络流和红队真值。 | 用于攻击链分析、时序聚合、实体图特征、少数恶意样本检测的鲁棒性评估。 |
| LANL Unified Host and Network Data Set | [Los Alamos National Laboratory](https://csr.lanl.gov/data/2017/) | 约 90 天企业网络 host event 与 netflow；主机和网络实体脱敏后一致，可联合分析。 | 与本赛题的 `src_host/dst_host/src_ip/dst_ip/timestamp` 字段高度相关。 | 用于构建主机-网络关系图、登录/进程/流量联合特征和异常检测对照实验。 |
| Splunk Boss of the SOC v3 | [Splunk 官方博客](https://www.splunk.com/en_us/blog/security/botsv3-dataset-released.html) / [GitHub](https://github.com/splunk/botsv3) | Splunk 发布的 SOC 竞赛数据集，包含云安全和 APT 调查场景，官方说明以开源许可发布。 | 更贴近 SOC 研判和告警调查流程，但不是逐事件三分类训练集。 | 用于可视化展示、分析叙事、攻击链样例和 SIEM 查询验证。 |
| OTRF Security Datasets | [OTRF GitHub](https://github.com/OTRF/Security-Datasets) | 开源安全事件数据，目标包括提供恶意/良性数据、支持 Sigma、Atomic Red Team、Threat Hunter Playbook 和 MITRE ATT&CK 映射。 | 适合补充具体 ATT&CK 技术的日志样例。 | 用于提取攻击技术关键词、生成规则解释样例、验证可疑行为发现模块。 |
| DARPA OpTC | [Five Directions / DARPA OpTC GitHub](https://github.com/FiveDirections/OpTC-data) | DARPA 公开发布，包含 Windows 10 端点传感器、eCAR、Bro/Zeek 相关数据和红队 ground truth；官方说明数据约 TB 级压缩 JSON。 | 适合高级 APT/端点遥测检测研究，但体量和复杂度较高。 | 用于后续图模型、端点行为序列、APT 攻击链实验；首版 baseline 不建议直接引入。 |

## 网络入侵与 IoT/IIoT 数据集

| 数据集 | 权威来源 | 内容与规模 | 与本赛题的关系 | 使用建议 |
| --- | --- | --- | --- | --- |
| CSE-CIC-IDS2018 | [University of New Brunswick CIC](https://www.unb.ca/cic/datasets/ids-2018.html) / [AWS Open Data](https://registry.opendata.aws/cse-cic-ids2018/) | CIC 与 CSE 合作，包含 Brute-force、Heartbleed、Botnet、DoS、DDoS、Web attacks、Infiltration 等 7 类攻击，含网络流量、主机日志和 80 个 CICFlowMeter 特征。 | 对 `network_flows/syslog` 类字段有参考价值，类别不均衡和攻击类型丰富。 | 用于网络流特征工程、IDS 传统 ML 对照、恶意类别召回率实验。 |
| UNSW-NB15 | [UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | Cyber Range 生成的现代正常与攻击流量，官方说明有 2,540,044 条记录、49 个特征和 9 类攻击。 | 适合网络入侵检测，但字段偏流量统计，与本赛题日志文本不同。 | 用于传统流量特征参考、攻击类别映射、模型跨数据集泛化测试。 |
| TON_IoT | [UNSW Research](https://research.unsw.edu.au/projects/toniot-datasets) | IoT/IIoT、Windows/Linux OS 日志、网络流量和遥测多源数据；官方说明可用于 IDS、威胁情报、恶意软件、取证和威胁狩猎。 | 覆盖云/边缘/主机/网络等异构来源，适合扩展多源 SOC 思路。 | 用于验证特征构造是否能适配多源日志、研究 IoT 场景异常检测。 |
| BoT-IoT | [UNSW Research](https://research.unsw.edu.au/projects/bot-iot-dataset) | UNSW Cyber Range 构建的 IoT botnet 环境，含 pcap、Argus、CSV；官方说明包含 DDoS、DoS、扫描、Keylogging、数据外传等攻击。 | 偏 IoT botnet 与网络取证，适合补充网络攻击样本。 | 用于少数攻击类别增强实验、DDoS/扫描/外传模式解释。 |

## 身份与内部威胁

| 数据集 | 权威来源 | 内容与规模 | 与本赛题的关系 | 使用建议 |
| --- | --- | --- | --- | --- |
| CERT Insider Threat Test Dataset | [CMU Software Engineering Institute](https://www.sei.cmu.edu/library/insider-threat-test-dataset/) / [DOI](https://doi.org/10.1184/R1/12841247.v1) | CERT Division 与 ExactData 在 DARPA I2O 资助下生成的合成内部威胁数据，包含背景行为、恶意 actor 和答案键。 | 对 `username/src_host/timestamp` 相关的身份行为建模有价值。 | 用于用户画像、异常登录、内部威胁案例解释；不要直接映射为比赛三分类标签。 |

## UNB CIC 数据集门户

[Canadian Institute for Cybersecurity datasets](https://www.unb.ca/cic/datasets/) 是一个持续维护的数据集入口，包含 IDS、IoT、DNS、Darknet、malware、phishing、graph learning 等类别。官方 FAQ 说明下载通常通过数据集页面表单完成，使用或再分发需要引用对应数据集和论文。

## 与本 baseline 的结合方式

1. **首版提交**：只用比赛 `train.parquet` 训练，避免外部域偏移污染线上预测。
2. **特征启发**：从 LANL、OpTC、BOTS、OTRF 中提炼实体关系、攻击技术关键词和时间窗口聚合思路。
3. **预训练或词表增强**：可以把外部日志只用于无监督文本向量、token 词表或 embedding 预训练，再在比赛数据上监督微调。
4. **鲁棒性验证**：用 CIC/UNSW/LANL 数据构造“跨源日志”测试，观察模型是否过度依赖某个厂商、产品或固定主机名。
5. **解释与可视化**：用 BOTS、OTRF、OpTC 的攻击链材料补充模型解释和 SOC 分析展示，而不是作为直接三分类训练标签。

