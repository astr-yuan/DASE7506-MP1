# 当前交付结果：V2已完成

- 完整test BPB：**1.6602157052495896**，达到1.5–1.7目标。
- CPU时间中位数比：1.980×；最保守比：2.032×；RAM：1.580 GiB；推理资产：4.172 MiB。
- 最终checkpoint：`code/runs/v2-selected/checkpoint.pt`。
- 训练链累计：58,982,400 targets；所有独立训练尝试合计：68,895,744 targets。
- 12项测试及真实checkpoint额外检查通过；保护文件、V1/V2冻结哈希一致。
- 下文按时间顺序保留计划、实现、失败尝试和结果，历史“待运行”描述代表当时状态。

---

# 开发与实验记录

## 规则与证据
仅修改 student.py 并添加配套文件；保留官方基线、训练器、评测器、测试、数据和 tokenizer。protected_hashes.json 记录原始 SHA256。只用 validation 选择方案，最终冻结后测试。已有 baseline test 是用户之前生成的记录，不用于候选筛选。

## 第 0 轮：核对原始环境与基线
- Python 3.12.10，PyTorch 2.7.1+cpu，4 线程，seed=17。
- 原有 baseline：1,200 步，batch=32，9,830,400 targets；validation BPB=2.0710836385；test BPB=2.1012604381；训练 296.89 秒。
- 已阅读 GUIDE、README 与此前 Mini Project 1 对话。对话仅作背景，约束以项目文档及用户要求为准。
- 不进行网站提交或发布，最终准备可供用户提交的本地材料。

## 第 1 轮：RoPE 等预算实验
- 修改：student.py 实现 RoPE；删除学习式位置 embedding；Q/K 按位置旋转，V 不变；仍使用严格因果 SDPA。
- 依据：绝对位置参数对每个位置分别学习；RoPE 让 attention 显式表达相对距离，有望改善有限训练下的位置泛化。
- 不变：4 层、128 宽、4 头、GELU、LayerNorm、共享词嵌入、AdamW、学习率计划、seed=17、32×256×1,200 targets。
- 消融：config 中 rope=false 返回官方 GPT；已有 baseline 是关闭机制的对照。两者参数量略不同，初始化随机数消耗也不同，不能声称排除了所有混杂因素。
- 所有新训练从随机初始化开始，parent checkpoint=null。
- 结果：待运行。

- 附加验证：官方及新增测试共 8 项通过；baseline checkpoint/source 哈希核验通过，12 个受保护文件保持不变。

- 新增 configs/rope.json 和 configs/ablation.json、environment.json、SUBMISSION_README.md。实际首轮使用 baseline config，其中 rope 缺省值为 true，和显式 rope=true 行为完全一致。
- 训练器保持原样，没有实现或使用断点续训；本次训练 ancestry 为 null。
- 第 300 步 validation=2.226245；第 600 步=2.032009，优于 baseline 完整训练的 2.071084。尚未运行候选 test。
- 辅助操作问题：首次 unittest 工作目录错误已更正；Python 从 code/ 追加父目录日志两次被拒绝，从项目根目录操作成功，无需扩大权限。
- RoPE 来源：Su et al., https://arxiv.org/abs/2104.09864；仅阅读论文说明，无外部训练数据或权重。

- 第 900 步 validation=1.955055；未扩大搜索，继续完成预定 1,200 步。唯一候选训练不使用任何父 checkpoint。

## 第 2 轮：冻结和资源验证（不修改模型）
- 1,200 步 validation=1.9223254395，训练 379.735 秒，参数 1,055,488；比 baseline validation 低 0.148758。达到充足优势后不再增加训练或搜索。
- FREEZE.json 在第一次候选 test 前生成，记录时间、选择依据、全部关键哈希与 ancestry。
- 首次官方完整 test BPB=1.9527401976，428,405 targets、1,292,013 bytes，满足 BPB<=2.10126。
- 发现初版内存脚本只采样 Windows venv 启动器（约 5MB），属于无效内存证据。将该 JSON 标为 invalid，并修正为枚举后代进程、累计各进程历史峰值 working set，作保守内存估计。模型/训练器/官方评测器未变。

## 第 3 轮：复现、报告与交付
- 三对交替 CPU 测量完成：中位数比 1.194944，最保守比 1.646098；RAM=1.579098 GiB；资产=4.164885 MiB，全部达标。
- 新目录仅复制推理代码、固定数据/评测器、checkpoint，再次得 BPB=1.952740197620131。使用同一依赖环境，不冒称全新环境。
- 新增 RESULTS.json、RUN_LOG.csv、README.md、SUBMISSION_README.md、REPORT.md 和三页 REPORT.pdf；报告包含方法、等预算/消融、成本、局限、AI 披露与引用。原官方 README 不改。
- PDF 构建脚本首次有字符串引号错误，修复后生成；首版四页末页只有引用，缩小字号/行距为三页并逐页渲染检查。只涉及文档，没有修改冻结模型。
- 没有新增训练、seed 搜索、测试集调参或父 checkpoint。

- 最终对真实训练 checkpoint 额外检验：256-token 因果性、batch 独立、重复调用状态独立、归一化、有限输出全部通过，证据见 measurements/checkpoint_contract.json。
- 精确依赖版本补记：numpy=2.5.3、tokenizers=0.21.4。
- 交付完整 ZIP 包：包含代码、原数据、baseline/最终 checkpoint、日志、评测原始记录与报告；排除 .venv、__pycache__、.DS_Store。另生成逐文件 SHA256 清单，方便核验不可变内容。无需重训即可评测。
- 学号、不可变 GitHub 链接、checkpoint 下载链接和课程网站发布尚需用户完成；本次开发未进行账号提交。


# 第二阶段：用户要求进一步接近 BPB 1.5–1.7

## V2 第 0 轮：预先记录方案
- 用户在 V1 交付后提出继续开发；V1 test 已知，不能声称后续 test 是完全未接触的独立留出证据。新阶段不查看测试文本，只按 validation 选择设置，在新冻结前不评测候选 test。
- V1 报告与日志保存于 history/v1；V1 源码、checkpoint、FREEZE.json 保持原样。
- 初始计划：从 V1 checkpoint 权重热启动，新增 6,000 步、batch=32、context=256、seed=23、peak lr=0.0006、weight_decay=0.1。重新创建 AdamW，不冒称恢复了原 optimizer。父模型 9,830,400 targets 全额计入。
- 增加定期保存、optimizer/RNG 恢复和验证选优；后半程等间隔权重平均。不是不同独立模型的集成，平均权重不增加推理大小。
- 在同一 checkpoint 上验证窗口内因果重复 n-gram 混合，开启/关闭可构成严格等训练成本的消融。
- 全部方案保存在 code/V2_PLAN.json；目标范围是实测目标，不伪造保证。

## V2 第 1 轮：训练器与 GPU 环境
- 新增 train_extended.py，不改官方 train.py。2-step/batch=2 功能检查完成，实际新增 1,024 targets（父成本仍保留）；live/average 验证、best/last/resume 保存、完整运行后的 resume 路径均通过。检查用 checkpoint 不参与选优或后续训练。
- 发现本机 NVIDIA RTX 3050 Laptop 4GiB，驱动 546.30。申请网络权限后建立独立 .venv-gpu，安装 torch==2.7.1 的官方 CUDA 11.8 wheel，以兼容现有驱动；不动 .venv CPU 环境。
- 新增训练器 device/precision 参数；计划 GPU BF16 训练、FP32 验证。最终 CPU FP32 正式评分不变。原因是目标需要显著增加训练量，GPU 可缩短运行时间。
- GPU 来源：官方 PyTorch previous versions / cu118 wheel index；依赖下载不是外部训练数据。

- GPU 环境尝试未完成：官方 2.8GB wheel 整包下载无有效进展；1MB Range 诊断成功，但分段下载脚本两次被自动审批拒绝（sandbox_approval 不允许），未进一步绕过。停止 pip 安装，采用已验证 CPU FP32 环境启动正式 v2-extended-cpu；.venv-gpu 不参与实验及交付。
- copy pilot 全部 10 个 full-validation 候选完成：最佳 alpha=0.4/order=2/smoothing=2，validation=1.84703516897；关闭时1.92232543951。没有新增训练或 test 调用。
- 本轮正式训练参数与 V2_PLAN 初始CPU方案一致，6,000 追加步；GPU/BF16 仅为未采用的加速尝试。

- 新增 freeze_v2.py：独立冻结清单记录 V2 配置、选择依据、累计训练量、祖先、源文件及推理资产字节；禁止覆盖已有 V2 冻结清单，不改 V1 FREEZE.json。
- 保持 CPU 训练不中断。另外尝试标准 pip 的官方主下载地址（带官方 SHA256），此窄范围命令获执行许可；没有运行被拒绝的分段脚本。即使 GPU 依赖安装成功，也不改变当前训练的计数或精度。

## V2 第 2 轮：正式扩展训练（CPU FP32）
- run=v2-extended-cpu，从 V1 最终权重开始，重建 AdamW，seed=23。完整计划6,000步，每步8,192 targets，全部继承父9,830,400 targets。
- 第600步神经模型 validation=1.8696979033，累计14,745,600 targets；模型checkpoint、optimizer/RNG恢复状态及事件日志已落盘。
- 同时保留 live/best 两份推理权重。resume.pt 是训练恢复资产，不应计为推理所需文件，最终包会分开说明。

- 第1,200追加步 validation=1.8162224102；累计19,660,800 targets。当前训练不改变原计划。
- copy pilot 10次完整验证共计135.658秒。所有候选指标保存在 validation_grid.json，不缓存或保存逐token答案。
- 标准 pip 官方主地址尝试同样未完成，已停止。GPU安装只是未采用的环境尝试，正式训练全部CPU FP32，不计为GPU训练。

- 第1,800追加步 validation=1.7783028187，累计24,576,000 targets。
- 新增 verify_v2.py：同时核对原始保护文件、V1冻结文件和（生成后）V2冻结文件，避免新开发破坏上轮可复现性。

- 第2,400追加步 validation=1.7553280276；第3,000步=1.7316384333。第3,000步首次建立平均权重，只有一个快照，所以与 live 分数完全相同，这是预期行为；后续节点才构成有意义的平均效果对比。
- 新增 summarize_v2.py：汇总最终三对资源测量、核验冻结哈希、报告目标范围是否实测达到；资源不合格时明确失败，不通过修改评测器隐瞒。

- 开发文件清单补记：student_v2.py 实现窗口内最长已观察后缀续词分布与神经概率混合；select_copy.py 在完整validation上比较10种预定配置；tests/test_v2.py新增4项针对性验证（总计12项测试通过）。依据是V1训练预算有限以及窗口内可重复利用的已观察语言片段，pilot结果支持该机制。
- V2_DEVELOPMENT.md提供中文解释和恢复命令；check_v2_checkpoint.py用于对最终真实权重做完整上下文因果性、归一化、独立性及精确关闭消融检查。
- 第3,600步 live=1.7174375488，average=1.7052748102；第4,200步 live=1.7074717144，average=1.6993242054。平均权重暂时更优，尚未冻结或测试V2。

- 第4,800步 live=1.6993187523、average=1.6957301265；第5,400步 live=1.6947695978、average=1.6932588130。改善放缓但仍继续完成预定预算，不据test调参。
- V2报告拟更新为4页并包含验证曲线。报告运行环境没有matplotlib，改用已提供的ReportLab标准LinePlot；未改训练依赖或安装额外绘图库。

- 正式6,000追加步完成：live validation=1.6924256015，average=1.6915309382；选中第6,000步平均权重，31个快照来自3,000–6,000步每100步一次。
- 扩展训练实测1,860.060秒；新处理49,152,000 targets，连同父9,830,400合计58,982,400。开始在此固定神经权重上重新执行10点validation混合网格，尚未进行V2 test。

## V2 第3轮：最终选择、冻结与测试
- 最终10点网格：选中alpha=0.25/order=2/smoothing=2，validation=1.6458924987；关闭copy时=1.6915309382。相同神经权重、相同58,982,400 targets，机制贡献为validation降低0.0456384395。
- V2已在首次新test前写入FREEZE_V2.json；真实权重的完整上下文因果性、前缀一致、batch独立、状态重置、归一化、有限输出和copy关闭精确等价全部通过。
- 首次官方完整CPU FP32 test：BPB=1.6602157052495896，覆盖428,405 targets和1,292,013 UTF-8 bytes，达到用户预期1.5–1.7范围。之后保持模型冻结，只做资源和复现检查。

## V2 第4轮：复现与交付
- 三次冻结模型BPB完全一致：1.6602157052496。CPU中位数比1.979648、最保守比2.031573、RAM 1.579708 GiB、推理资产4.172212 MiB，全部通过。
- 独立目录复制模型源码、原数据/评测器及checkpoint后，不重训得到同一BPB；使用既有Python环境，不冒称全新安装。
- 更新README.md、SUBMISSION_README.md、REPORT.md、4页REPORT.pdf及RUN_LOG.csv；保留history/v1及V1冻结代码/权重。
- 总唯一训练成本记录为68,895,744 targets：原baseline、原smoke、V1训练、1,024-target训练器检查和本轮49,152,000新增targets；父成本仅在ancestry中累计，搜索总额不重复相加。
- 新PDF使用ReportLab标准曲线绘图，全部页面渲染检查；交付包排除虚拟环境、安装缓存、临时目录和pycache。


## 2026-09-29 最终提交代码整理

- 依据用户要求，将神经网络和重复后缀机制合并到 student.py；将初始训练、继续训练、验证集选择统一到 train.py 的三个 --stage 入口。保留原始采样、优化器、学习率、随机种子和平均权重逻辑。
- 仅保留最终 checkpoint；将其 implementation 从 student_v2 改为 student，以匹配新模块名。全部参数张量和其余元数据逐项核对未变；新文件哈希随序列化改变，不再使用旧 checkpoint 哈希验证此包。原始来源 SHA-256：da42888086ed3c23c6fda90882d5439a1ece0f57dbab4398dd7b8d4f4b89e22e。
- common.py、evaluate.py、model.py、baseline 配置、原始契约测试及 data 文件逐字节哈希核对一致。移除历史实验产物、辅助开发脚本、缓存和虚拟环境；历史记录中的旧文件名只用于说明实验来源。
- 合并后 9 项契约/重复机制测试通过；完整测试集 CPU FP32 BPB=1.6602157052495896，与原冻结模型一致。新 checkpoint SHA-256=d30a6e33d5305ed8cba16f748698d29b167e81f441c1124c64af67870b4dd8a1。
- 初始训练和继续训练入口各执行 2 步、batch=2 的运行检查，分别新增 1,024 targets，祖先累计正确记录为 2,048；这些检查权重不用于最终提交。包含检查的历史唯一训练量更新为 68,897,792 targets。
- 本次只验证重构与打包，不重新调参或重新训练最终模型。
- 继续训练的恢复入口通过加载检查；验证集选择入口重跑全部10个既定配置，得到相同alpha=0.25/order=2/smoothing=2，validation BPB=1.6458924986975914。此为既定流程复核，不依据test重新选择。
- 代码包包含唯一最终权重、必要源码、未修改的官方数据/评测器、契约测试、README、简短最终报告与历史修改记录；不包含历史模型、实验中间产物、虚拟环境及缓存。


## 2026-09-29 重新训练：第一阶段1200步，加训12000步

- 用户预先指定 steps=12000、seed=23、lr=0.0006、batch=32、CPU FP32、4线程、每600步验证、从6000步每100步平均。第一阶段从零训练1200步、seed=17。不使用旧最终权重作为父模型。
- 复用原 code/.venv 的已验证依赖环境；代码执行目录为 code1。只新增结果、检查辅助脚本和文档，不修改训练算法及受保护文件。
- 官方baseline从先前同种子1200步实验复制，其checkpoint哈希与原metrics核验一致，不增加重复baseline训练成本；将在同一机器上重新评测资源。
- 新增measure_cpu.py（Windows进程树峰值内存测量）及check_checkpoint.py（真实权重契约检查）；后者引用统一模型的RotaryGPT完成精确关闭混合的消融核验。
- 所有候选模型与混合参数只用validation选择；先冻结，再首次测试新模型。旧模型test成绩已知，继续如实披露。

### 本轮训练完成与验收

- 追加第600步，live validation BPB=1.8700698677。
- 追加第1200步，live validation BPB=1.8169782906。
- 追加第1800步，live validation BPB=1.7793881073。
- 追加第2400步，live validation BPB=1.7576865031。
- 追加第3000步，live validation BPB=1.7365214819。
- 追加第3600步，live validation BPB=1.7225956252。
- 追加第4200步，live validation BPB=1.7140627788。
- 追加第4800步，live validation BPB=1.7069401414。
- 追加第5400步，live validation BPB=1.7011561298。
- 追加第6000步，live validation BPB=1.6956801417。
- 追加第6000步，average validation BPB=1.6956801417。
- 追加第6600步，live validation BPB=1.6910609567。
- 追加第6600步，average validation BPB=1.6701432507。
- 追加第7200步，live validation BPB=1.6848393998。
- 追加第7200步，average validation BPB=1.6665021079。
- 追加第7800步，live validation BPB=1.6812036031。
- 追加第7800步，average validation BPB=1.6644883725。
- 追加第8400步，live validation BPB=1.6793817057。
- 追加第8400步，average validation BPB=1.6634705251。
- 追加第9000步，live validation BPB=1.6766010907。
- 追加第9000步，average validation BPB=1.6626486573。
- 追加第9600步，live validation BPB=1.6741907238。
- 追加第9600步，average validation BPB=1.6621702149。
- 追加第10200步，live validation BPB=1.6724949089。
- 追加第10200步，average validation BPB=1.6618450380。
- 追加第10800步，live validation BPB=1.6716311471。
- 追加第10800步，average validation BPB=1.6617299972。
- 追加第11400步，live validation BPB=1.6699289503。
- 追加第11400步，average validation BPB=1.6616445925。
- 追加第12000步，live validation BPB=1.6708548535。
- 追加第12000步，average validation BPB=1.6615916362。
- 完成12000追加步；最优候选：第12000步 average。十点验证网格选中{'vocab': 2048, 'width': 128, 'heads': 4, 'depth': 4, 'context': 256, 'copy_alpha': 0.25, 'copy_order': 2, 'copy_smoothing': 2.0}，validation=1.6186115943175。
- 在新test前冻结源文件与权重哈希；完整CPU FP32 test BPB=1.6328937779090。三次重复结果与资源测量记录见checks/RESULTS.json。
- 本次新训练108,134,400 targets，历史累计177,032,192；不按本轮test继续修改设置。
- 补充measure_cpu.py及check_checkpoint.py是为落实资源和真实权重检查；训练实现、受保护文件不变。更新README、REPORT与PDF，原文档保存在checks/previous-documentation。

独立目录复现补充：将新代码包解压到独立目录，使用相同依赖环境运行包内官方评测器，得到相同test BPB=1.632893777909；没有依赖原训练目录的模型源码。此为复用环境的独立文件目录检查，不是全新环境安装检查。4页PDF已逐页渲染并检查。
