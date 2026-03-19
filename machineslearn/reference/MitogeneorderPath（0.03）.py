"""
MitogeneorderPath - 基于线粒体基因组基因排列顺序的物种鉴定工具

专为多毛类动物设计，结合图卷积网络(GCN)技术与对比学习，
基于101个物种的线粒体基因组基因排列顺序进行物种鉴定和进化关系分析。

核心特点:
  - 仅使用15个核心基因（13个蛋白编码基因 + 2个rRNA）构建模型
  - GCN嵌入 + 对比学习 + 最近邻检索（非分类范式）
  - 支持加载已有训练模型，跳过训练步骤
  - 支持直接上传已有基因顺序文件，跳过MITOS2注释步骤
  - 自动检测/引导安装MITOS2环境

方法说明:
  由于多毛类动物线粒体基因中，15个核心基因排列高度保守
  （101个物种中仅有27种不同排列，其中67个物种共享相同排列），
  传统分类方法无法有效区分。本工具采用 GCN 嵌入 + 对比学习：
  1. GCN 将基因顺序编码为低维嵌入向量
  2. 对比学习使相同排列的样本靠近、不同排列的远离
  3. 预测时通过嵌入距离 + 基因顺序匹配度综合排序
  4. 输出最近邻物种及基因顺序匹配百分比

模块说明（均可独立调用）:
  1. build_reference_database() - 参考数据库构建
  2. build_filtered_index()      - 构建15基因过滤索引
  3. check_mitos2()              - MITOS2环境检测
  4. install_mitos2()            - MITOS2安装引导
  5. run_mitos()                 - MITOS2注释调用
  6. extract_gene_order()        - 从MITOS2输出提取基因顺序
  7. extract_gene_order_from_file() - 从已有文件提取基因顺序
  8. filter_gene_order()         - 基因顺序过滤（仅保留15个核心基因）
  9. train_gcn_model()           - GCN对比学习训练
  10. load_trained_model()       - 加载已有训练模型
  11. predict_species()          - 物种最近邻检索
  12. plot_comparison()          - 基因顺序可视化比对

依赖:
  pip install numpy pandas torch torch_geometric scikit-learn matplotlib
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适配无显示器的Linux服务器
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split

# ============================================================
# 全局配置
# ============================================================

DEFAULT_CSV_PATH = input("请输入数据库的文件路径: ")     # 参考数据库CSV路径
DEFAULT_OUTPUT_DIR = "./mito_analysis_results/"       # 输出目录

BATCH_SIZE = 32
HIDDEN_DIM = 64
EMBED_DIM = 32
EPOCHS = 500
LR = 0.005
MARGIN = 2.0  # 对比学习间隔（不同基因顺序对的最小距离）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNKNOWN_GENE = "<UNK>"  # 未知基因统一占位符

# ============================================================
# 15个核心基因（13个蛋白编码基因 + 2个rRNA）
# ============================================================

CORE_GENES = [
    "cox1", "cox2", "atp8", "cox3", "nad6", "cob", "atp6",
    "nad5", "nad4l", "nad4", "rrnS", "rrnL", "nad1", "nad3", "nad2",
]


# ============================================================
# 基因名称标准化
# ============================================================

GENE_NAME_MAP = {
    # COX 基因
    "coxi": "cox1", "cox_i": "cox1", "co1": "cox1",
    "coxii": "cox2", "cox_ii": "cox2", "co2": "cox2",
    "coxiii": "cox3", "cox_iii": "cox3", "co3": "cox3",
    # CytB -> cob
    "cytb": "cob", "cytB": "cob", "CytB": "cob",
    # ATP 基因
    "atpase6": "atp6", "atpase8": "atp8",
    # NADH 基因
    "nd1": "nad1", "nd2": "nad2", "nd3": "nad3",
    "nd4": "nad4", "nd4l": "nad4l", "nd5": "nad5", "nd6": "nad6",
    # rRNA 别名
    "rrnl": "rrnL", "12s": "rrnS", "12S": "rrnS",
    "16s": "rrnL", "16S": "rrnL",
    "rrn12": "rrnS", "rrn16": "rrnL",
    "rnr1": "rrnS", "rnr2": "rrnL",
    # tRNA 单字母退化标记
    "l": "trnL", "s": "trnS",
    # 注释错误修正
    "trnA-asn": "trnN",
}


def normalize_gene_name(name):
    """标准化单个基因名称。无法识别返回 None。"""
    name = name.strip()
    if not name:
        return None

    cleaned = name.lstrip("-").lower()
    if cleaned.startswith("oh") or cleaned == "oh":
        return None

    if name.startswith("-"):
        name = name[1:].strip()
        if not name:
            return None

    key = name.lower()

    if key in GENE_NAME_MAP:
        return GENE_NAME_MAP[key]

    if key.startswith("trn"):
        return key

    known_bases = {
        "cox1", "cox2", "cox3", "cob", "atp6", "atp8",
        "nad1", "nad2", "nad3", "nad4", "nad4l", "nad5", "nad6",
        "rrns", "rrnl",
    }
    if key in known_bases:
        return key

    return None


def normalize_gene_order(raw_genes, keep_unknown=True):
    """标准化基因名称列表。"""
    result = []
    for g in raw_genes:
        g = g.strip()
        if not g:
            continue
        norm = normalize_gene_name(g)
        if norm is not None:
            result.append(norm)
        elif keep_unknown:
            result.append(UNKNOWN_GENE)
    return result


# ============================================================
# 模块1：参考数据库构建
# ============================================================

def build_reference_database(csv_path):
    """
    从CSV文件构建参考数据库（全基因版本，后续由 build_filtered_index 过滤）。
    """
    df = pd.read_csv(csv_path)
    species_col = df.columns[0]
    gene_col = df.columns[1]

    ref_db = {
        "species_genes": {},
        "gene_set": set(),
        "species_list": [],
    }

    for _, row in df.iterrows():
        species = str(row[species_col]).strip()
        gene_str = str(row[gene_col]).strip()
        genes = normalize_gene_order(gene_str.split(","), keep_unknown=True)
        ref_db["species_genes"][species] = genes
        ref_db["species_list"].append(species)
        for g in genes:
            if g != UNKNOWN_GENE:
                ref_db["gene_set"].add(g)

    all_genes = sorted(ref_db["gene_set"])
    if UNKNOWN_GENE not in all_genes:
        all_genes.append(UNKNOWN_GENE)

    ref_db["all_genes"] = all_genes
    ref_db["gene_to_idx"] = {g: i for i, g in enumerate(all_genes)}
    ref_db["idx_to_gene"] = {i: g for i, g in enumerate(all_genes)}
    ref_db["gene_num"] = len(all_genes)
    ref_db["num_classes"] = len(ref_db["species_list"])

    lengths = [len(v) for v in ref_db["species_genes"].values()]
    unk_total = sum(
        1 for genes in ref_db["species_genes"].values()
        for g in genes if g == UNKNOWN_GENE
    )

    print(f"[模块1] 参考数据库构建完成")
    print(f"  物种数量    : {ref_db['num_classes']}")
    print(f"  基因种类    : {ref_db['gene_num']} (含 {UNKNOWN_GENE})")
    print(f"  基因数量范围: {min(lengths)} ~ {max(lengths)}")
    print(f"  未知基因标记: {unk_total} 个")

    return ref_db


# ============================================================
# 模块2：构建15基因过滤索引
# ============================================================

def build_filtered_index(ref_db):
    """
    在参考数据库基础上，构建仅包含15个核心基因的过滤索引。
    对每个物种的基因顺序过滤，仅保留核心基因，并以cox1起始旋转。
    """
    core_genes = sorted(CORE_GENES)
    core_gene_to_idx = {g: i for i, g in enumerate(core_genes)}
    core_idx_to_gene = {i: g for i, g in enumerate(core_genes)}

    filtered_species_genes = {}
    for species, genes in ref_db["species_genes"].items():
        filtered = [g for g in genes if g in core_gene_to_idx]
        if "cox1" in filtered:
            idx = filtered.index("cox1")
            filtered = filtered[idx:] + filtered[:idx]
        filtered_species_genes[species] = filtered

    valid_species = [
        sp for sp, genes in filtered_species_genes.items() if len(genes) > 0
    ]

    lengths = [
        len(v) for v in filtered_species_genes.values() if len(v) > 0
    ]
    missing_species = [
        sp for sp, genes in filtered_species_genes.items() if len(genes) == 0
    ]

    print(f"[模块2] 15基因过滤索引构建完成")
    print(f"  核心基因列表: {core_genes}")
    print(f"  核心基因数量: {len(core_genes)}")
    if lengths:
        print(f"  过滤后基因数量范围: {min(lengths)} ~ {max(lengths)}")
    if missing_species:
        print(f"  ⚠ 警告: {len(missing_species)} 个物种不包含任何核心基因")
        for sp in missing_species[:5]:
            print(f"    - {sp}")
        if len(missing_species) > 5:
            print(f"    - ... 共{len(missing_species)}个")

    return {
        "core_gene_to_idx": core_gene_to_idx,
        "core_idx_to_gene": core_idx_to_gene,
        "core_genes": core_genes,
        "core_gene_num": len(core_genes),
        "filtered_species_genes": filtered_species_genes,
        "valid_species": valid_species,
    }


# ============================================================
# 模块3：MITOS2环境检测与安装
# ============================================================

def check_mitos2_installed():
    """检测本地是否已安装MITOS2。"""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, timeout=10
        )
        if "mitos2" in result.stdout:
            try:
                test_cmd = (
                    'bash -c \''
                    'source /root/miniconda3/etc/profile.d/conda.sh && '
                    'conda activate mitos2 && '
                    'runmitos.py --help'
                    '\''
                )
                ret = subprocess.run(
                    test_cmd, shell=True,
                    capture_output=True, text=True, timeout=15
                )
                if ret.returncode == 0:
                    print("[MITOS2] ✓ 已检测到MITOS2环境 (mitos2)")
                    return True
                else:
                    print("[MITOS2] ⚠ mitos2环境存在但runmitos.py不可用")
                    return False
            except (subprocess.TimeoutExpired, Exception):
                return False
        else:
            print("[MITOS2] ✗ 未检测到mitos2 conda环境")
            return False
    except FileNotFoundError:
        print("[MITOS2] ✗ 未检测到conda，无法检查MITOS2环境")
        return False
    except subprocess.TimeoutExpired:
        print("[MITOS2] ✗ conda环境检查超时")
        return False


def install_mitos2():
    """引导用户安装MITOS2。"""
    print("\n[MITOS2] 开始安装MITOS2...")
    print("=" * 50)

    create_cmd = [
        "conda", "create", "--strict-channel-priority",
        "-c", "conda-forge", "-c", "bioconda",
        "-n", "mitos2", "mitos>=2", "-y"
    ]

    print(f"[MITOS2] 执行: {' '.join(create_cmd)}")
    print("[MITOS2] 这可能需要几分钟时间，请耐心等待...\n")

    try:
        result = subprocess.run(
            create_cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"[MITOS2] 环境创建失败:")
            print(result.stderr)
            return False
        print("[MITOS2] ✓ conda环境创建成功")
    except subprocess.TimeoutExpired:
        print("[MITOS2] ✗ 环境创建超时，请手动执行:")
        print(f"  {' '.join(create_cmd)}")
        return False
    except Exception as e:
        print(f"[MITOS2] ✗ 环境创建异常: {e}")
        return False

    print("[MITOS2] 验证安装...")
    try:
        test_cmd = (
            'bash -c \''
            'source /root/miniconda3/etc/profile.d/conda.sh && '
            'conda activate mitos2 && '
            'runmitos --help'
            '\''
        )
        ret = subprocess.run(
            test_cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        if ret.returncode == 0:
            print("[MITOS2] ✓ 安装验证成功！runmitos.py可用")
            return True
        else:
            print("[MITOS2] ✗ 安装验证失败，请手动检查:")
            print("  conda activate mitos2 && runmitos.py --help")
            return False
    except Exception as e:
        print(f"[MITOS2] ✗ 验证过程异常: {e}")
        return False


# ============================================================
# 模块4：MITOS2注释调用
# ============================================================

def run_mitos(input_fasta, out_dir):
    """调用本地安装的MITOS2进行线粒体基因组注释。"""
    os.makedirs(out_dir, exist_ok=True)

    cmd = (
        'bash -c \''
        'source /root/miniconda3/etc/profile.d/conda.sh && '
        'conda activate mitos2 && '
        f'runmitos --input {input_fasta} --code 2 '
        f'--outdir {out_dir} --refdir /root --refseqver refseq89o'
        '\''
    )

    print(f"[模块4] MITOS2注释启动")
    print(f"  输入: {input_fasta}")
    print(f"  输出: {out_dir}")

    ret = os.system(cmd)
    if ret == 0:
        print(f"[模块4] MITOS2注释完成")
        return True
    else:
        print(f"[模块4] MITOS2运行异常 (返回码: {ret})，请检查环境配置")
        return False


# ============================================================
# 模块5：基因顺序提取
# ============================================================

def extract_gene_order(out_dir, gene_to_idx=None):
    """从MITOS2输出目录的 result.geneorder 中提取并标准化基因顺序。"""
    gene_order_file = os.path.join(out_dir, "result.geneorder")

    if not os.path.exists(gene_order_file):
        raise FileNotFoundError(f"未找到基因顺序文件: {gene_order_file}")

    with open(gene_order_file, "r") as f:
        lines = f.readlines()

    gene_line = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        gene_line = stripped
        break

    if not gene_line:
        raise ValueError("result.geneorder 中未找到基因序列数据")

    raw_genes = gene_line.split()
    gene_order = normalize_gene_order(raw_genes, keep_unknown=True)

    unk_count = sum(1 for g in gene_order if g == UNKNOWN_GENE)
    print(f"[模块5] 基因顺序提取完成")
    print(f"  基因数量: {len(gene_order)}")
    if unk_count > 0:
        print(f"  未知基因: {unk_count} 个")
    print(f"  顺序: {gene_order}")

    return gene_order


def extract_gene_order_from_file(file_path):
    """从用户提供的基因顺序文件中提取（空格或逗号分隔均可）。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r") as f:
        lines = f.readlines()

    gene_line = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        gene_line = stripped
        break

    if not gene_line:
        raise ValueError("文件中未找到基因序列数据")

    raw_genes = gene_line.split(",") if "," in gene_line else gene_line.split()
    gene_order = normalize_gene_order(raw_genes, keep_unknown=True)

    unk_count = sum(1 for g in gene_order if g == UNKNOWN_GENE)
    print(f"[模块5] 基因顺序文件读取完成")
    print(f"  文件: {file_path}")
    print(f"  基因数量: {len(gene_order)}")
    if unk_count > 0:
        print(f"  未知基因: {unk_count} 个")
    print(f"  顺序: {gene_order}")

    return gene_order


# ============================================================
# 模块6：基因顺序过滤
# ============================================================

def filter_gene_order(gene_order, core_gene_to_idx):
    """过滤出15个核心基因，以cox1为起始旋转。"""
    filtered = [g for g in gene_order if g in core_gene_to_idx]

    if "cox1" in filtered:
        idx = filtered.index("cox1")
        filtered = filtered[idx:] + filtered[:idx]
        print(f"[模块6] 已将cox1旋转至起始位置 (原位置: {idx})")
    else:
        print(f"[模块6] ⚠ 警告: 未找到cox1基因，保持原始顺序")

    print(f"[模块6] 基因顺序过滤完成")
    print(f"  原始基因数量: {len(gene_order)}")
    print(f"  过滤后基因数量: {len(filtered)}")
    print(f"  过滤后顺序: {filtered}")

    return filtered


# ============================================================
# 模块7：GCN模型
# ============================================================

def genes_to_graph(gene_order, gene_to_idx, label=None):
    """将基因顺序转换为 PyG 图数据对象（含环状首尾连接）。"""
    num_gene_types = len(gene_to_idx)

    node_features = []
    for gene in gene_order:
        idx = gene_to_idx.get(gene, gene_to_idx.get(UNKNOWN_GENE, 0))
        feat = [0.0] * num_gene_types
        feat[idx] = 1.0
        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float)

    edges = []
    for i in range(len(gene_order) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    # 环状连接
    if len(gene_order) > 2:
        edges.append([0, len(gene_order) - 1])
        edges.append([len(gene_order) - 1, 0])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    y = torch.tensor([label], dtype=torch.long) if label is not None else None
    return Data(x=x, edge_index=edge_index, y=y)


class GeneOrderDataset(torch.utils.data.Dataset):
    """封装多条基因顺序为 PyG 数据集。"""

    def __init__(self, gene_orders, labels, gene_to_idx):
        self.data_list = []
        for i, go in enumerate(gene_orders):
            data = genes_to_graph(go, gene_to_idx, label=labels[i])
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class GNNClassifier(nn.Module):
    """
    两层 GCN 嵌入模型。
    前向传播返回图级嵌入向量（用于最近邻比较）。
    """

    def __init__(self, num_node_features, hidden_dim, embed_dim):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embed_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        graph_embedding = global_mean_pool(x, batch)
        # L2归一化，使嵌入在单位超球面上
        return F.normalize(graph_embedding, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数。
    相同基因顺序的样本对被拉近，不同的被推开。
    """

    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, order_keys):
        """
        参数:
            embeddings : (batch_size, embed_dim) L2归一化后的嵌入
            order_keys : list[str] 每个样本的基因顺序key（用于判断正/负对）
        返回:
            scalar loss
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        total_loss = 0.0
        num_pairs = 0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # 余弦距离（因为已经L2归一化）
                dist = 1.0 - (embeddings[i] * embeddings[j]).sum()
                if order_keys[i] == order_keys[j]:
                    # 正对：相同基因顺序 → 距离应接近0
                    total_loss = total_loss + dist ** 2
                else:
                    # 负对：不同基因顺序 → 距离应超过margin
                    total_loss = total_loss + F.relu(self.margin - dist) ** 2
                num_pairs += 1

        return total_loss / max(num_pairs, 1)


def gene_order_to_key(gene_order):
    """将基因顺序列表转为可哈希的元组key，用于对比学习。"""
    return tuple(gene_order)


def train_gcn_model(filtered_index, output_dir=None,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM, lr=LR):
    """
    使用对比学习训练 GCN 嵌入模型（仅15核心基因）。
    不做分类，而是学习基因顺序的向量表示。

    返回 (model, species_list, full_dataset)
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.pth")
    core_gene_to_idx = filtered_index["core_gene_to_idx"]
    num_features = filtered_index["core_gene_num"]
    filtered_species_genes = filtered_index["filtered_species_genes"]
    valid_species = filtered_index["valid_species"]
    num_classes = len(valid_species)

    if num_classes == 0:
        raise ValueError("没有可用的物种数据（过滤后所有物种的核心基因数量为0）")

    gene_orders = [filtered_species_genes[sp] for sp in valid_species]
    order_keys = [gene_order_to_key(go) for go in gene_orders]

    # 全量训练（嵌入学习不需要预留验证集，但为了监控过拟合保留少量）
    all_indices = list(range(len(gene_orders)))
    _, monitor_idx = train_test_split(all_indices, test_size=0.1, random_state=42)

    # 训练数据集（无label，纯嵌入学习）
    train_data_list = []
    for go in gene_orders:
        train_data_list.append(genes_to_graph(go, core_gene_to_idx))

    monitor_orders = [gene_orders[i] for i in monitor_idx]
    monitor_keys = [order_keys[i] for i in monitor_idx]

    # 手动mini-batch采样（避免PyG DataLoader对tuple键的处理问题）
    from torch_geometric.data import Batch

    # 训练数据：全量列表，手动采样mini-batch（避免PyG DataLoader对tuple的处理问题）
    all_train_indices = list(range(len(train_data_list)))

    # 初始化模型
    model = GNNClassifier(num_features, hidden_dim, embed_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    loss_fn = ContrastiveLoss(margin=MARGIN)

    # 统计不同基因顺序
    unique_orders = len(set(order_keys))
    print(f"[模块7] 开始训练GCN嵌入模型（对比学习）")
    print(f"  总样本数    : {len(gene_orders)}")
    print(f"  物种类别    : {num_classes}")
    print(f"  不同基因顺序: {unique_orders}")
    print(f"  节点特征    : {num_features}维 (one-hot)")
    print(f"  隐藏层      : {hidden_dim} → 嵌入维度: {embed_dim}")
    print(f"  设备        : {DEVICE}")
    if num_classes > unique_orders:
        print(f"  ⚠ 注意: 物种数({num_classes}) > 不同基因顺序数({unique_orders})")
        print(f"    模型将基于基因顺序相似度进行最近邻检索，而非严格分类")

    # 预计算所有嵌入用于监控
    def compute_all_embeddings(model_ref):
        model_ref.eval()
        all_embeds = []
        with torch.no_grad():
            for data in train_data_list:
                data = data.to(DEVICE)
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
                embed = model_ref(data)
                all_embeds.append(embed.cpu().numpy())
        return np.vstack(all_embeds)

    def evaluate_separability(embeddings, keys, _monitor_keys):
        """评估嵌入空间中不同基因顺序的可分离度。"""
        correct = 0
        total = 0
        for i in monitor_idx:
            dists = np.linalg.norm(embeddings - embeddings[i], axis=1)
            dists[i] = float('inf')
            nearest = np.argmin(dists)
            if keys[nearest] == keys[i]:
                correct += 1
            total += 1
        return correct / max(total, 1)

    def manual_batch_loss(model_ref, indices, loss_fn_ref):
        """手动构建mini-batch并计算对比损失。"""
        model_ref.train()
        batch_data_list = [train_data_list[i].clone() for i in indices]
        batch_keys = [order_keys[i] for i in indices]
        batch = Batch.from_data_list(batch_data_list).to(DEVICE)
        embeddings = model_ref(batch)
        return loss_fn_ref(embeddings, batch_keys)

    best_separability = 0.0
    np.random.seed(42)

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(all_train_indices)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(all_train_indices), batch_size):
            batch_indices = all_train_indices[start:start + batch_size]
            if len(batch_indices) < 2:
                continue
            optimizer.zero_grad()
            loss = manual_batch_loss(model, batch_indices, loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step()

        # 每20轮评估一次
        if epoch % 20 == 0 or epoch == 1:
            all_embeds = compute_all_embeddings(model)
            sep = evaluate_separability(all_embeds, order_keys, monitor_keys)
            marker = " *" if sep > best_separability else ""
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"基因顺序分离度: {sep:.4f}{marker}")

            if sep > best_separability:
                best_separability = sep
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "species_list": valid_species,
                    "order_keys": [list(k) for k in set(order_keys)],
                    "hidden_dim": hidden_dim,
                    "embed_dim": embed_dim,
                    "num_node_features": num_features,
                    "num_classes": num_classes,
                    "unique_orders": unique_orders,
                }, model_path)

    # 最终评估
    all_embeds = compute_all_embeddings(model)
    sep = evaluate_separability(all_embeds, order_keys, monitor_keys)
    print(f"[模块7] 训练完成 | 最佳基因顺序分离度: {best_separability:.4f}")
    print(f"  模型已保存: {model_path}")

    # 返回全量数据集用于最近邻检索
    full_dataset = GeneOrderDataset(
        gene_orders, list(range(len(gene_orders))), core_gene_to_idx
    )

    return model, valid_species, full_dataset


# ============================================================
# 模块8：加载已有训练模型
# ============================================================

def load_trained_model(model_path, filtered_index):
    """
    加载已训练好的GCN嵌入模型，重建数据集用于最近邻检索。
    返回 (model, species_list, full_dataset)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    saved_num_features = checkpoint["num_node_features"]
    core_gene_num = filtered_index["core_gene_num"]

    if saved_num_features != core_gene_num:
        raise ValueError(
            f"模型不兼容: 模型使用 {saved_num_features} 个基因特征, "
            f"当前过滤索引使用 {core_gene_num} 个核心基因。"
            f"请确保使用相同的geneorder.csv。"
        )

    hidden_dim = checkpoint["hidden_dim"]
    embed_dim = checkpoint["embed_dim"]
    saved_species = checkpoint["species_list"]

    model = GNNClassifier(saved_num_features, hidden_dim, embed_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 重建数据集
    core_gene_to_idx = filtered_index["core_gene_to_idx"]
    filtered_species_genes = filtered_index["filtered_species_genes"]

    gene_orders = [
        filtered_species_genes[sp] for sp in saved_species
        if sp in filtered_species_genes and len(filtered_species_genes[sp]) > 0
    ]

    full_dataset = GeneOrderDataset(
        gene_orders, list(range(len(gene_orders))), core_gene_to_idx
    )

    print(f"[模块8] 模型加载完成")
    print(f"  模型路径    : {model_path}")
    print(f"  节点特征    : {saved_num_features}维")
    print(f"  隐藏层      : {hidden_dim} → 嵌入维度: {embed_dim}")
    print(f"  参考物种数  : {len(saved_species)}")
    print(f"  训练样本数  : {len(full_dataset)}")

    return model, saved_species, full_dataset


# ============================================================
# 模块9：物种预测与最近邻检索
# ============================================================

def predict_species(query_gene_order, filtered_index, model, species_list,
                    train_dataset, top_k=5):
    """
    对查询核心基因顺序进行最近邻物种检索。
    使用GCN嵌入距离 + 基因顺序编辑距离综合排序。

    返回 dict(predicted_species, nearest_neighbors, similarity_info)
    """
    core_gene_to_idx = filtered_index["core_gene_to_idx"]

    # 1. GCN嵌入距离
    query_graph = genes_to_graph(query_gene_order, core_gene_to_idx)
    query_graph = query_graph.to(DEVICE)
    query_graph.batch = torch.zeros(
        query_graph.x.size(0), dtype=torch.long, device=DEVICE
    )

    model.eval()
    with torch.no_grad():
        query_embedding = model(query_graph).cpu().numpy()

    train_embeddings = []
    with torch.no_grad():
        for data in train_dataset:
            data = data.to(DEVICE)
            data.batch = torch.zeros(
                data.x.size(0), dtype=torch.long, device=DEVICE
            )
            embed = model(data).cpu().numpy()
            train_embeddings.append(embed)
    train_embeddings = np.vstack(train_embeddings)

    gcn_distances = np.linalg.norm(train_embeddings - query_embedding, axis=1)

    # 2. 基因顺序相似度（直接比对）
    # 计算查询与每个参考的基因顺序匹配度
    order_similarities = []
    for i in range(len(species_list)):
        ref_order = train_dataset[i].x.argmax(dim=1).tolist()
        # 转回基因名
        core_genes = filtered_index["core_genes"]
        ref_genes = [core_genes[idx] if idx < len(core_genes) else "?" for idx in ref_order]
        # 计算位置匹配率（以较长的为基准）
        max_len = max(len(query_gene_order), len(ref_genes))
        if max_len == 0:
            order_similarities.append(1.0)
            continue
        matches = sum(
            1 for a, b in zip(query_gene_order, ref_genes) if a == b
        )
        order_similarities.append(matches / max_len)

    # 3. 综合排序（GCN距离为主，基因顺序相似度为辅）
    # 归一化
    gcn_dist_norm = gcn_distances / (gcn_distances.max() + 1e-8)
    order_sim_arr = np.array(order_similarities)

    # 综合得分：GCN距离越小越好，顺序相似度越高越好
    combined_score = gcn_dist_norm - order_sim_arr * 0.3

    nearest_indices = np.argsort(combined_score)[:top_k]

    nearest_neighbors = []
    for idx in nearest_indices:
        species = species_list[idx]
        nearest_neighbors.append({
            "species": species,
            "gcn_distance": round(float(gcn_distances[idx]), 4),
            "order_similarity": round(float(order_similarities[idx]), 4),
            "combined_score": round(float(combined_score[idx]), 4),
        })

    predicted_species = nearest_neighbors[0]["species"]
    best_sim = nearest_neighbors[0]["order_similarity"]
    best_dist = nearest_neighbors[0]["gcn_distance"]

    # 检查是否有完全匹配的基因顺序
    exact_matches = sum(1 for s in order_similarities if s == 1.0)
    same_order_species = [
        species_list[i] for i in range(len(species_list))
        if order_similarities[i] == 1.0
    ]

    result = {
        "predicted_species": predicted_species,
        "nearest_neighbors": nearest_neighbors,
        "similarity_info": {
            "best_gcn_distance": round(float(best_dist), 4),
            "best_order_similarity": round(float(best_sim), 4),
            "exact_order_matches": exact_matches,
            "same_order_species": same_order_species[:10],
        },
    }

    print(f"\n[模块9] 预测结果（最近邻检索）")
    print(f"  最相似物种  : {predicted_species}")
    print(f"  GCN嵌入距离 : {best_dist:.4f}")
    print(f"  基因顺序匹配度: {best_sim:.4f} ({int(best_sim*100)}%)")
    if exact_matches > 1:
        print(f"  ⚠ 有 {exact_matches} 个物种共享相同基因顺序:")
        for sp in same_order_species[:5]:
            print(f"    - {sp}")
        if len(same_order_species) > 5:
            print(f"    - ... 共{exact_matches}个")
    print(f"\n  最近邻 {top_k}:")
    print(f"  {'排名':>4} {'物种ID':<14} {'GCN距离':>10} {'顺序匹配':>10} {'综合得分':>10}")
    print(f"  {'-'*52}")
    for i, nn in enumerate(nearest_neighbors, 1):
        print(f"  {i:>4} {nn['species']:<14} {nn['gcn_distance']:>10.4f} "
              f"{nn['order_similarity']:>10.4f} {nn['combined_score']:>10.4f}")

    return result


# ============================================================
# 模块10：基因顺序可视化比对
# ============================================================

def plot_comparison(query_order, ref_orders, ref_names,
                    core_gene_to_idx, core_idx_to_gene,
                    save_path=None, show=False):
    """可视化比对查询样本与参考样本的核心基因顺序差异。"""
    n = min(len(ref_orders), 3)
    if n == 0:
        print("[模块10] 无参考序列可比对")
        return

    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]

        q_idx = [core_gene_to_idx.get(g, 0) for g in query_order]
        r_idx = [core_gene_to_idx.get(g, 0) for g in ref_orders[i]]

        ax.plot(range(len(q_idx)), q_idx, 'o-', label="Query",
                markersize=5, linewidth=1.2)
        ax.plot(range(len(r_idx)), r_idx, 's--',
                label=f"Ref: {ref_names[i]}",
                markersize=5, linewidth=1.2)

        # 标记差异位置
        min_len = min(len(query_order), len(ref_orders[i]))
        for j in range(min_len):
            if query_order[j] != ref_orders[i][j]:
                ax.axvline(x=j, color='red', alpha=0.15, linewidth=0.8)

        ax.set_xlabel("Gene Position")
        ax.set_ylabel("Gene Index")
        ax.set_title(f"Comparison #{i+1}: Query vs {ref_names[i]}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[模块10] 比对图已保存: {save_path}")
    if show:
        plt.show()

    plt.close()


# ============================================================
# 主流程 - 多重交互式
# ============================================================

def safe_input(prompt):
    """兼容不同环境的 input 封装。"""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""
    except KeyboardInterrupt:
        print("\n用户中断，退出程序。")
        sys.exit(0)


def main(csv_path=DEFAULT_CSV_PATH, output_dir=DEFAULT_OUTPUT_DIR):
    """
    完整分析流程（交互式）:
      1. 构建参考数据库 + 15基因过滤索引
      2. 交互选择: 加载已有模型 or 重新训练
      3. 交互选择: 运行MITOS2注释 or 上传已有基因顺序文件
      4. 预测物种 + 可视化比对 + 保存结果
    """
    os.makedirs(output_dir, exist_ok=True)

    # ==== 阶段1: 构建参考数据库 + 15基因过滤索引 ====
    print("=" * 60)
    print("  MitogeneorderPath - 线粒体基因顺序物种鉴定工具")
    print("  (仅使用15个核心基因: 13 PCGs + 2 rRNAs)")
    print("=" * 60)

    print(f"\n{'='*60}")
    print("  阶段1: 构建参考数据库")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"错误: 参考数据库文件不存在 - {csv_path}")
        print("请确认路径后重新运行")
        return None

    ref_db = build_reference_database(csv_path)
    filtered_index = build_filtered_index(ref_db)

    # ==== 阶段2: 模型选择 ====
    print(f"\n{'='*60}")
    print("  阶段2: 模型选择")
    print(f"{'='*60}")

    has_model = safe_input(
        "\n是否已有本脚本训练好的模型？(yes/no): "
    ).lower()

    model = None
    species_list = None
    train_dataset = None

    if has_model in ("yes", "y", "是"):
        model_path = safe_input("请输入已有模型的文件路径: ").strip()
        if not model_path:
            print("错误: 路径不能为空")
            return None
        try:
            model, species_list, train_dataset = load_trained_model(model_path, filtered_index)
            print("  ✓ 模型加载成功，跳过训练步骤")
        except (FileNotFoundError, ValueError) as e:
            print(f"  ✗ 模型加载失败: {e}")
            print("  将重新训练模型...")
            model = None

    if model is None:
        print("\n  开始训练新模型...")
        try:
            model, species_list, train_dataset = train_gcn_model(
                filtered_index, output_dir
            )
        except Exception as e:
            print(f"  ✗ 模型训练失败: {e}")
            return None

    # ==== 阶段3: 获取基因顺序 ====
    print(f"\n{'='*60}")
    print("  阶段3: 获取查询样本基因顺序")
    print(f"{'='*60}")

    need_annotation = safe_input(
        "\n需要运行MITOS2注释序列？(yes/no)\n"
        "  输入 no 可直接上传已有的基因顺序文件(result.geneorder): "
    ).lower()

    gene_order_full = None  # 完整基因顺序（注释后）
    gene_order_filtered = None  # 过滤后的核心基因顺序

    if need_annotation in ("no", "n", "否"):
        # 直接上传已有基因顺序文件
        file_path = safe_input(
            "请输入已有基因顺序文件的路径(result.geneorder): "
        ).strip()
        if not file_path:
            print("错误: 路径不能为空")
            return None
        try:
            gene_order_full = extract_gene_order_from_file(file_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: {e}")
            return None
    else:
        # 需要运行MITOS2注释
        # 检查MITOS2是否已安装
        print("\n  检查MITOS2安装状态...")
        mitos_installed = check_mitos2_installed()

        if not mitos_installed:
            install_choice = safe_input(
                "\n  未检测到MITOS2，是否自动安装？(yes/no): "
            ).lower()
            if install_choice in ("yes", "y", "是"):
                mitos_installed = install_mitos2()
                if not mitos_installed:
                    print("  ✗ MITOS2安装失败，请手动安装后重试")
                    fallback = safe_input(
                        "  是否改为上传已有基因顺序文件？(yes/no): "
                    ).lower()
                    if fallback in ("yes", "y", "是"):
                        file_path = safe_input(
                            "  请输入基因顺序文件路径: "
                        ).strip()
                        if file_path:
                            try:
                                gene_order_full = extract_gene_order_from_file(file_path)
                            except (FileNotFoundError, ValueError) as e:
                                print(f"  错误: {e}")
                                return None
                        else:
                            print("  错误: 路径不能为空")
                            return None
                    else:
                        print("  程序退出")
                        return None
            else:
                print("  未安装MITOS2且用户选择不安装，程序退出")
                return None

        if mitos_installed and gene_order_full is None:
            # MITOS2可用，获取FASTA文件并运行注释
            input_fasta = safe_input(
                "\n请输入待分析的线粒体基因组FASTA文件路径: "
            ).strip()
            if not input_fasta:
                print("错误: 路径不能为空")
                return None
            if not os.path.exists(input_fasta):
                print(f"错误: 文件不存在 - {input_fasta}")
                return None

            mitos_ok = run_mitos(input_fasta, output_dir)
            if not mitos_ok:
                print("  MITOS2运行失败，尝试使用已有的注释结果...")
                gene_order_file = os.path.join(output_dir, "result.geneorder")
                if os.path.exists(gene_order_file):
                    print(f"  找到已有注释文件: {gene_order_file}")
                else:
                    fallback = safe_input(
                        "  也未找到已有注释文件，是否上传基因顺序文件？(yes/no): "
                    ).lower()
                    if fallback in ("yes", "y", "是"):
                        file_path = safe_input(
                            "  请输入基因顺序文件路径: "
                        ).strip()
                        if file_path:
                            try:
                                gene_order_full = extract_gene_order_from_file(file_path)
                            except (FileNotFoundError, ValueError) as e:
                                print(f"  错误: {e}")
                                return None
                        else:
                            return None
                    else:
                        return None

            if gene_order_full is None:
                try:
                    gene_order_full = extract_gene_order(output_dir)
                except (FileNotFoundError, ValueError) as e:
                    print(f"错误: {e}")
                    return None

    # ==== 阶段4: 过滤为核心基因 ====
    print(f"\n{'='*60}")
    print("  阶段4: 基因顺序过滤")
    print(f"{'='*60}")

    gene_order_filtered = filter_gene_order(
        gene_order_full, filtered_index["core_gene_to_idx"]
    )

    if len(gene_order_filtered) == 0:
        print("错误: 过滤后基因顺序为空，查询序列中不包含任何核心基因")
        return None

    # ==== 阶段5: 预测物种 ====
    print(f"\n{'='*60}")
    print("  阶段5: 物种预测")
    print(f"{'='*60}")

    result = predict_species(
        gene_order_filtered, filtered_index, model, species_list, train_dataset
    )

    # ==== 阶段6: 可视化比对 ====
    print(f"\n{'='*60}")
    print("  阶段6: 可视化比对")
    print(f"{'='*60}")

    nn_species = [nn["species"] for nn in result["nearest_neighbors"]]
    ref_orders = [
        filtered_index["filtered_species_genes"][sp]
        for sp in nn_species
        if sp in filtered_index["filtered_species_genes"]
    ]
    nn_valid = nn_species[:len(ref_orders)]

    plot_comparison(
        gene_order_filtered, ref_orders, nn_valid,
        filtered_index["core_gene_to_idx"],
        filtered_index["core_idx_to_gene"],
        save_path=os.path.join(output_dir, "gene_order_comparison.png"),
    )

    # ==== 阶段7: 保存结果 ====
    print(f"\n{'='*60}")
    print("  阶段7: 保存结果")
    print(f"{'='*60}")

    output_result = {
        "query_gene_order_full": gene_order_full,
        "query_gene_order_filtered": gene_order_filtered,
        "core_genes_used": filtered_index["core_genes"],
        **result,
    }
    result_path = os.path.join(output_dir, "prediction_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  [完成] 全部分析结果已保存至: {output_dir}")
    print(f"  - 比对图   : {os.path.join(output_dir, 'gene_order_comparison.png')}")
    print(f"  - 预测结果 : {result_path}")
    print(f"{'='*60}")

    return output_result


if __name__ == "__main__":
    main()
