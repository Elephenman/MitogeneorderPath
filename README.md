# MitogeneorderPath

> 基于图卷积网络（GCN）与对比学习的线粒体基因组基因排列顺序物种鉴定工具

专为多毛类动物（Polychaeta）设计，利用线粒体基因组中 **13个蛋白编码基因 + 2个rRNA** 的排列顺序信息，通过 GCN 嵌入和最近邻检索实现物种鉴定。

## 📌 背景

多毛类动物线粒体基因的核心基因排列高度保守。在本工具的101个参考物种中：

| 指标 | 数值 |
|------|------|
| 总参考物种数 | 101 |
| 不同基因排列数 | 27 |
| 共享最常见排列的物种数 | 67 (66%) |

由于大量物种共享相同的核心基因排列，传统分类方法（将每个物种作为独立类别）无法有效工作。本工具采用 **GCN 嵌入 + 对比学习 + 最近邻检索** 的范式来解决这个问题。

## 🔬 方法

### 技术路线

```
线粒体基因组 FASTA
        │
        ▼
    MITOS2 注释 ──→ result.geneorder（全基因顺序）
        │
        ▼
  过滤 15 个核心基因 ──→ [cox1, cox2, atp8, cox3, nad6, cob, atp6,
                         nad5, nad4l, nad4, rrnS, rrnL, nad1, nad3, nad2]
        │
        ▼
  GCN 编码（3层 GCNConv + Global Mean Pooling）
        │
        ▼
  L2 归一化嵌入向量（32维）
        │
        ▼
  与参考库嵌入计算距离 ──→ 综合排序 ──→ 最近邻物种列表
```

### 为什么用对比学习而非分类？

- **分类问题**：67个物种共享同一张图 → GCN 无论怎么训练，同一输入只能产生同一输出 → Val Acc = 0
- **嵌入问题**：对比学习只要求"相同排列拉近，不同排列推开" → 27种排列完全可分 → 预测时通过嵌入距离检索最近邻

### 图结构

- **节点**：每个核心基因为一个节点，特征为 15 维 one-hot 向量
- **边**：相邻基因双向连接 + 首尾环状连接（线粒体基因组为环状 DNA）
- **图级表示**：Global Mean Pooling → L2 归一化

## 🚀 快速开始

### 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.10
- PyTorch Geometric ≥ 2.0
- CUDA（推荐，CPU 也可运行）

### 安装依赖

```bash
# 创建 conda 环境
conda create -n mito_gcn python=3.10 -y
conda activate mito_gcn

# 安装 PyTorch（根据你的 CUDA 版本调整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 PyTorch Geometric
pip install torch_geometric

# 安装其他依赖
pip install numpy pandas scikit-learn matplotlib

# 安装 MITOS2（用于线粒体基因组注释，可选）
conda create --strict-channel-priority -c conda-forge -c bioconda -n mitos2 "mitos>=2" -y
```

### 使用方法

```bash
python MitogeneorderPath.py
```

工具会以交互式引导你完成以下流程：

```
阶段1: 构建参考数据库（从 geneorder.csv）
阶段2: 模型选择（加载已有模型 / 重新训练）
阶段3: 获取查询序列（MITOS2注释 / 上传已有基因顺序文件）
阶段4: 基因顺序过滤（提取15个核心基因）
阶段5: 最近邻检索（GCN嵌入 + 基因顺序匹配度综合排序）
阶段6: 可视化比对（生成基因顺序比对图）
阶段7: 保存结果（JSON 格式预测报告）
```

## 📁 项目结构

```
MitogeneorderPath/
├── machineslearn/
│   ├── MitogeneorderPath.py    # 主程序
│   └── reference/
│       └── geneorder.csv       # 参考数据库（101个物种的基因顺序）
├── mito_analysis_results/       # 输出目录（自动创建）
│   ├── best_model.pth          # 训练好的模型权重
│   ├── gene_order_comparison.png  # 基因顺序比对可视化图
│   └── prediction_result.json  # 预测结果报告
└── README.md
```

## 📊 输入文件格式

### 参考数据库 `geneorder.csv`

CSV 格式，第一行为表头：

```csv
sample_id,standardized_gene_sequence
U24570.1,"cox1,trnN,cox2,trnD,atp8,trnY,trnG,cox3,trnQ,nad6,cob,trnW,atp6,..."
AY961084.1,"cox1,trnN,cox2,trnD,atp8,trnY,cox3,trnQ,nad6,cob,trnW,atp6,..."
```

### 查询基因顺序文件

与 MITOS2 输出的 `result.geneorder` 格式一致，空格或逗号分隔均可：

```
>Sample1
cox1 trnN cox2 trnD atp8 trnY cox3 trnQ nad6 cob ...
```

## 📋 输出示例

### 控制台输出

```
[模块9] 预测结果（最近邻检索）
  最相似物种  : U24570.1
  GCN嵌入距离 : 0.1234
  基因顺序匹配度: 0.9333 (93%)
  ⚠ 有 67 个物种共享相同基因顺序:
    - U24570.1
    - AY961084.1
    - ...

  最近邻 5:
  排名 物种ID         GCN距离    顺序匹配    综合得分
  ----------------------------------------------------
     1 U24570.1        0.1234      0.9333      0.0534
     2 EF656365.1      0.5678      0.6000      0.3878
     3 KR534502.1      0.8901      0.4667      0.7501
```

### 预测结果 JSON

```json
{
  "query_gene_order_full": ["cox1", "trnN", "cox2", ...],
  "query_gene_order_filtered": ["cox1", "cox2", "atp8", "cox3", ...],
  "core_genes_used": ["atp6", "atp8", "cob", ...],
  "predicted_species": "U24570.1",
  "nearest_neighbors": [...],
  "similarity_info": {
    "best_gcn_distance": 0.1234,
    "best_order_similarity": 0.9333,
    "exact_order_matches": 67,
    "same_order_species": ["U24570.1", ...]
  }
}
```

## 🧩 15个核心基因

| 类型 | 基因名 | 编码产物 |
|------|--------|----------|
| 蛋白编码基因 | cox1, cox2, cox3 | 细胞色素c氧化酶亚基 |
| 蛋白编码基因 | cob | 细胞色素b |
| 蛋白编码基因 | nad1, nad2, nad3, nad4, nad4l, nad5, nad6 | NADH脱氢酶亚基 |
| 蛋白编码基因 | atp6, atp8 | ATP合酶亚基 |
| rRNA | rrnS (12S) | 小亚基核糖体RNA |
| rRNA | rrnL (16S) | 大亚基核糖体RNA |

## ⚙️ 超参数配置

在 `MitogeneorderPath.py` 中修改全局配置：

```python
BATCH_SIZE = 32          # 训练批大小
HIDDEN_DIM = 64          # GCN隐藏层维度
EMBED_DIM = 32           # 嵌入向量维度
EPOCHS = 500             # 训练轮数
LR = 0.005               # 学习率
MARGIN = 2.0             # 对比学习间隔（负对最小距离）
```

## 📖 模块 API

所有模块均可独立调用：

```python
from MitogeneorderPath import (
    build_reference_database,
    build_filtered_index,
    train_gcn_model,
    load_trained_model,
    predict_species,
    filter_gene_order,
    extract_gene_order_from_file,
)

# 1. 构建参考数据库 + 过滤索引
ref_db = build_reference_database("./reference/geneorder.csv")
filtered = build_filtered_index(ref_db)

# 2. 训练或加载模型
model, species_list, dataset = train_gcn_model(filtered)
# 或: model, species_list, dataset = load_trained_model("best_model.pth", filtered)

# 3. 预测
result = predict_species(
    query_gene_order=["cox1","cox2","atp8","cox3",...],
    filtered_index=filtered,
    model=model,
    species_list=species_list,
    train_dataset=dataset,
)
```

## 🧠 模型架构

```
Input: Gene Order Graph (N nodes × 15-dim one-hot)
         │
    ┌────┴────┐
    │ GCNConv │  15 → 64  + ReLU + Dropout(0.3)
    └────┬────┘
    ┌────┴────┐
    │ GCNConv │  64 → 64  + ReLU + Dropout(0.2)
    └────┬────┘
    ┌────┴────┐
    │ GCNConv │  64 → 32
    └────┬────┘
         │
  Global Mean Pool → L2 Normalize → 32-dim Embedding
```

**训练损失**：Contrastive Loss
- 正对（相同基因排列）：最小化嵌入距离
- 负对（不同基因排列）：推开至 margin 以上

## ⚠️ 注意事项

1. **基因顺序保守性**：本工具基于15个核心基因的排列顺序进行鉴定。由于多毛类中这些基因高度保守，鉴定结果可能给出多个候选物种（共享相同排列），需结合形态学或其他分子标记进一步确认。

2. **MITOS2 注释**：首次运行 MITOS2 时会自动下载参考数据库，请确保网络畅通。如果已有注释结果（`result.geneorder` 文件），可直接上传跳过注释步骤。

3. **模型兼容性**：加载已有模型时，必须使用与训练时相同的 `geneorder.csv`，否则基因索引不匹配会报错。

4. **GPU 加速**：自动检测 CUDA，有 GPU 时使用 GPU 训练，否则使用 CPU。

## 📄 许可证

本项目仅供学术研究使用。

## 🙏 致谢

- [MITOS2](https://github.com/ChristianBernt/MITOS2) - 线粒体基因组注释工具
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络库
