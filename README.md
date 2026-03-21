# MitogeneorderPath
基于图卷积网络（GCN）与对比学习的线粒体基因组基因排列顺序物种鉴定工具
专为多毛类动物（Polychaeta）设计，结合**两步判断策略**，利用线粒体基因组基因排列顺序信息实现高精度物种鉴定与进化关系分析，解决核心基因保守导致的近缘物种区分难题。

## 📌 项目背景
多毛类动物线粒体基因组中**15个核心基因（13个蛋白编码基因+2个rRNA）** 排列相对保守。

本工具摒弃传统分类范式，采用**GCN嵌入+对比学习+最近邻检索**，结合**完整基因顺序（含tRNA）精细匹配**，实现“高效筛选+精准区分”的物种鉴定，适配多毛类动物线粒体基因的进化特征。

| 参考指标 | 数值 |
|----------|------|
| 总参考物种数 | 101 |
| 核心基因排列类型 | 27种 |
| 共享相同核心排列的物种数 | 67种 |
| 核心基因数量 | 15个（13 PCGs + 2 rRNAs） |

## 🔬 核心方法
### 两步判断策略
针对多毛类基因保守特性设计，兼顾筛选效率与鉴定精度：
1. **第一步（主判断）**：15核心基因GCN嵌入+对比学习，快速筛选候选物种集
2. **第二步（精细判断）**：完整基因顺序（含tRNA）加权相似度匹配，区分核心排列相同的近缘物种

### 技术路线
```
线粒体基因组FASTA / 已有基因顺序文件
        │
        ▼
MITOS2注释（自动检测/安装环境）→ result.geneorder（全基因顺序）
        │
        ▼
基因名称标准化 → 过滤15个核心基因 → 以cox1为起点环状旋转
        │
        ▼
GCN编码（3层GCNConv + 全局平均池化）→ L2归一化32维嵌入向量
        │
        ▼
余弦距离最近邻检索 → 筛选候选物种集
        │
        ▼
完整基因顺序相似度计算（核心基因95%权重 + tRNA 5%权重）→ 综合排序
        │
        ▼
输出Top-K物种 + 基因顺序可视化比对 + 结果文件保存
```

### 图结构建模（贴合线粒体环状特征）
- **节点**：每个基因为一个节点，特征为基因类型的one-hot编码（含未知基因`<UNK>`占位）
- **边**：基因间**双向相邻连接** + **环状首尾连接**（还原线粒体环状DNA的空间特征）
- **图级表示**：通过`global_mean_pool`聚合所有节点特征，生成代表整个基因顺序的全局嵌入向量

## 🚀 快速开始
### 环境要求
- Python ≥ 3.8
- PyTorch ≥ 1.10
- PyTorch Geometric ≥ 2.0
- CUDA（推荐，自动检测，CPU亦可运行）
- Conda（用于MITOS2环境管理）

### 安装依赖
```bash
# 1. 创建并激活项目主环境
conda create -n mito_gcn python=3.10 -y
conda activate mito_gcn

# 2. 安装PyTorch（根据CUDA版本调整，示例为cu118）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装PyTorch Geometric
pip install torch_geometric

# 4. 安装其他核心依赖
pip install numpy pandas scikit-learn matplotlib

# 5. MITOS2无需手动安装（工具将自动检测/引导安装）
```

### 使用方法
直接运行主程序，工具将以**交互式引导**完成全流程，无需手动配置复杂参数：
```bash
python MitogeneorderPath.py
```

### 交互式流程
工具将依次引导完成以下步骤，支持**跳过训练/注释**等冗余步骤：
```
阶段1: 输入参考数据库路径 → 构建全基因参考库 + 15核心基因过滤索引
阶段2: 模型选择 → 加载已有训练模型 / 重新训练GCN模型
阶段3: 基因顺序获取 → 运行MITOS2注释FASTA / 上传已有基因顺序文件
阶段4: 核心基因过滤 → 提取15核心基因 + cox1环状旋转
阶段5: 物种预测 → 两步判断策略（主判断+精细判断）输出Top-K结果
阶段6: 可视化比对 → 生成核心基因顺序比对图
阶段7: 结果保存 → 输出JSON格式预测报告+可视化图片
```

## 📁 项目结构
```
MitogeneorderPath/
├── MitogeneorderPath.py    # 主程序（含所有可独立调用模块）
├── geneorder.csv           # 参考数据库（101个物种，物种名+基因顺序）
└── mito_analysis_results/  # 自动创建的输出目录
    ├── best_model.pth      # 训练好的GCN模型权重（含模型参数+物种列表）
    ├── gene_order_comparison.png  # 基因顺序可视化比对图
    ├── prediction_result.json     # 详细预测结果（JSON格式）
    └── mitos2_output/      # MITOS2注释输出文件（自动生成）
```

## 📊 输入文件格式
### 参考数据库 `geneorder.csv`
CSV格式，无表头，第一列为**物种名**，第二列为**逗号分隔的基因顺序**：
```csv
物种A,cox1,trnN,cox2,trnD,atp8,cox3,nad6,cob,atp6,nad5,nad4l,nad4,rrnS,rrnL,nad1,nad3,nad2
物种B,cox1,trnS,cox2,atp8,atp6,cox3,nad6,cob,nad5,nad4l,nad4,rrnS,rrnL,nad1,nad3,nad2
```

### 查询基因顺序文件
支持**MITOS2输出的`result.geneorder`** 或自定义文件，**空格/逗号分隔**均可，支持`>`注释行：
```
>Sample_Query
cox1 trnN cox2 trnD atp8 trnY cox3 trnQ nad6 cob trnW atp6 nad5 nad4l nad4 rrnS rrnL nad1 nad3 nad2
# 或逗号分隔
>Sample_Query
cox1,trnN,cox2,trnD,atp8,trnY,cox3,trnQ,nad6,cob,trnW,atp6,nad5,nad4l,nad4,rrnS,rrnL,nad1,nad3,nad2
```

### 线粒体基因组FASTA文件
标准FASTA格式，用于MITOS2自动注释：
```
>Query_Mito_Genome
ATCGGCTAATCGGCTAATCGGCTAATCGGCTAATCGGCTAATCGGCTAATCGGCTAATCGGCTA
ATCGGCTAATCGGCTAATCGGCTAATCGGCTAATCGGCTAATCGGCTAATCGGCTAATCGGCTA
```

## 📋 输出结果说明
### 控制台输出（关键信息）
```
[第一步] 15核心基因主判断...
✅ 找到 3 个核心基因完全匹配的物种:
    - 物种A
    - 物种B
    - 物种C

[第二步] 完整基因顺序精细判断（候选数：3）...

📊 最终Top-5匹配结果:
 1. 物种A
    - 核心基因匹配度: 100.0%
    - 完整基因相似度: 98.5%
    - GCN余弦距离: 0.0215
 2. 物种B
    - 核心基因匹配度: 100.0%
    - 完整基因相似度: 95.3%
    - GCN余弦距离: 0.0589
 3. 物种C
    - 核心基因匹配度: 100.0%
    - 完整基因相似度: 92.1%
    - GCN余弦距离: 0.0963

[完成] 全部分析结果已保存至: ./mito_analysis_results/
```

### 预测结果JSON（`prediction_result.json`）
包含**全量分析信息**，便于后续二次分析：
```json
{
  "query_gene_order_full": ["cox1", "trnN", "cox2", ...],
  "query_gene_order_filtered": ["cox1", "cox2", "atp8", ...],
  "core_genes_used": ["cox1", "cox2", "atp8", "cox3", "nad6", "cob", "atp6", "nad5", "nad4l", "nad4", "rrnS", "rrnL", "nad1", "nad3", "nad2"],
  "predicted_species": "物种A",
  "nearest_neighbors": [
    {
      "species": "物种A",
      "score": 0.985,
      "core_similarity": 1.0,
      "trna_similarity": 0.85,
      "core_order_sim": 1.0,
      "gcn_distance": 0.0215
    },
    ...
  ],
  "exact_match_species": ["物种A", "物种B", "物种C"],
  "similarity_info": {...}
}
```

### 可视化比对图（`gene_order_comparison.png`）
展示查询样本与Top-K参考物种的**核心基因顺序差异**，红色竖线标记基因排列不同的位置，直观体现物种间的基因顺序差异。

## 🧩 15个核心基因（鉴定主特征）
| 基因类型 | 基因名 | 编码产物 | 功能说明 |
|----------|--------|----------|----------|
| 蛋白编码基因 | cox1, cox2, cox3 | 细胞色素c氧化酶亚基 | 呼吸链电子传递核心组件 |
| 蛋白编码基因 | cob | 细胞色素b | 呼吸链复合物Ⅲ亚基 |
| 蛋白编码基因 | nad1, nad2, nad3, nad4, nad4l, nad5, nad6 | NADH脱氢酶亚基 | 呼吸链复合物Ⅰ亚基 |
| 蛋白编码基因 | atp6, atp8 | ATP合酶亚基 | 线粒体ATP合成核心组件 |
| rRNA基因 | rrnS（12S） | 小亚基核糖体RNA | 核糖体翻译核心组件 |
| rRNA基因 | rrnL（16S） | 大亚基核糖体RNA | 核糖体翻译核心组件 |

## ⚙️ 全局超参数配置
在`MitogeneorderPath.py`头部可直接修改，适配不同数据集/硬件环境：
```python
# 训练参数
BATCH_SIZE = 32          # 训练批次大小
HIDDEN_DIM = 64          # GCN隐藏层维度
EMBED_DIM = 32           # 最终嵌入向量维度
EPOCHS = 500             # 对比学习训练轮数
LR = 0.005               # 优化器学习率
MARGIN = 2.0             # 对比学习负对最小距离（保证不同排列的区分度）

# 环境/路径参数
DEFAULT_OUTPUT_DIR = "./mito_analysis_results/"  # 默认输出目录
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动硬件检测
UNKNOWN_GENE = "<UNK>"  # 未知基因统一占位符
```

## 🧩 核心模块API（均可独立调用）
所有功能模块解耦设计，支持单独调用实现个性化分析：
```python
from MitogeneorderPath import (
    # 数据库构建
    build_reference_database, build_filtered_index,
    # MITOS2相关
    check_mitos2_installed, install_mitos2, run_mitos,
    # 基因顺序处理
    extract_gene_order, extract_gene_order_from_file, filter_gene_order,
    # 模型训练/加载
    train_gcn_model, load_trained_model,
    # 物种预测/相似度计算
    predict_species, full_gene_order_similarity,
    # 可视化
    plot_comparison
)

# 示例：基础物种预测流程
# 1. 构建参考库
ref_db = build_reference_database("geneorder.csv")
filtered_index = build_filtered_index(ref_db)

# 2. 加载模型
model, species_list, train_dataset = load_trained_model("./mito_analysis_results/best_model.pth", filtered_index)

# 3. 读取查询基因顺序
query_full = extract_gene_order_from_file("query_geneorder.txt")
query_filtered = filter_gene_order(query_full, filtered_index["core_gene_to_idx"])

# 4. 物种预测
result = predict_species(
    query_filtered, query_full,
    filtered_index, ref_db,
    model, species_list, train_dataset,
    top_k=5
)

# 5. 可视化比对
plot_comparison(
    query_filtered,
    [filtered_index["filtered_species_genes"][sp] for sp in result["exact_match_species"]],
    result["exact_match_species"],
    filtered_index["core_gene_to_idx"],
    filtered_index["core_idx_to_gene"],
    save_path="./comparison.png"
)
```

## 🧠 模型架构与训练
### GCN嵌入模型（`GNNClassifier`）
3层图卷积网络，专为基因顺序图结构设计，无分类层，仅做**特征提取**：
```
输入: 基因顺序图（N节点 × N基因类型one-hot特征）
         │
    ┌────┴────┐
    │ GCNConv │  N → 64 + ReLU + Dropout(0.3) （捕捉局部相邻特征）
    └────┬────┘
    ┌────┴────┐
    │ GCNConv │  64 → 64 + ReLU + Dropout(0.2)（融合深层结构特征）
    └────┬────┘
    ┌────┴────┐
    │ GCNConv │  64 → 32 （生成低维嵌入）
    └────┬────┘
         │
全局平均池化 → L2归一化 → 32维图级嵌入向量（单位超球面）
```

### 对比学习损失（`ContrastiveLoss`）
摒弃传统分类损失，采用对比损失让模型学习**基因顺序的相似性结构**：
- **正对**（相同基因排列的样本）：最小化嵌入向量余弦距离（损失=距离平方）
- **负对**（不同基因排列的样本）：将余弦距离推开至`MARGIN=2.0`以上（损失=max(0, 2.0-距离)²）
- **训练监控**：通过**基因顺序分离度**评估模型效果，保存分离度最高的模型

### 相似度计算规则
#### 1. 核心基因相似度（95%权重）
- 以cox1为起点环状对齐，逐位匹配核心基因
- 匹配度 = 匹配基因数 / 最大核心基因数
#### 2. tRNA相似度（5%权重）
- 基于核心基因骨架的tRNA插入位置（gap）计算Jaccard相似度
- 标准化tRNA名称（如trnS1→trnS），消除注释命名差异
#### 3. 综合得分
**综合相似度 = 核心基因相似度 × 0.95 + tRNA相似度 × 0.05**

## ⚠️ 重要注意事项
1. **MITOS2环境管理**：工具将自动检测MITOS2环境，未安装时引导自动安装，无需手动配置；若安装失败，可直接上传已有`result.geneorder`文件跳过注释。
2. **模型兼容性**：加载已有模型时，必须使用**与训练时完全相同的`geneorder.csv`**，否则会因基因索引不匹配报错。
3. **环状基因组对齐**：所有核心基因顺序均以**cox1为起点**旋转，解决线粒体环状DNA因注释起点不同导致的排列差异问题。
4. **未知基因处理**：注释中无法识别的基因将以`<UNK>`占位，不影响核心基因筛选与模型分析。
5. **结果解读**：若输出多个高相似度物种，说明这些物种共享相同的核心基因排列，需结合**形态学特征/其他分子标记（如COI序列）** 进一步确认。
6. **服务器适配**：Matplotlib采用`Agg`非交互式后端，适配无显示器的Linux服务器，无需配置图形界面。

## 📄 许可证
本项目仅供**学术研究与非商业使用**，禁止用于商业用途。

## 🙏 致谢
- [MITOS2](https://github.com/ChristianBernt/MITOS2) - 线粒体基因组高效注释工具，为本项目提供基因顺序注释支持
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 轻量级图神经网络库，为本项目提供GCN模型与图数据处理支持
- PyTorch 团队 - 提供高效的深度学习框架，支持模型训练与硬件加速
