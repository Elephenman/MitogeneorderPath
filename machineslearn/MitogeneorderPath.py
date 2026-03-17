"""
MitogeneorderPath - 基于线粒体基因组基因排列顺序的物种鉴定工具

专为多毛类动物设计，结合图卷积网络(GCN)技术，
基于101个物种的线粒体基因组基因排列顺序进行物种鉴定和进化关系分析。

模块说明（均可独立调用）:
  1. build_reference_database() - 参考数据库构建
  2. run_mitos()               - MITOS2注释调用
  3. extract_gene_order()      - 基因顺序提取与标准化
  4. train_gcn_model()         - GCN模型训练
  5. predict_species()         - 物种预测与最近邻检索
  6. plot_comparison()         - 基因顺序可视化比对

依赖:
  pip install numpy pandas torch torch_geometric scikit-learn matplotlib
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适配无显示器的Linux服务器
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ============================================================
# 全局配置
# ============================================================

DEFAULT_CSV_PATH = "./geneorder.csv"       # 参考数据库CSV路径
DEFAULT_OUTPUT_DIR = "./mito_analysis_results/"  # 输出目录

BATCH_SIZE = 32
HIDDEN_DIM = 64
EPOCHS = 100
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNKNOWN_GENE = "<UNK>"  # 未知基因统一占位符


# ============================================================
# 基因名称标准化
# ============================================================

# 已知的基因名别名映射（小写key → 标准名）
GENE_NAME_MAP = {
    # COX 基因
    "coxi": "cox1", "cox_i": "cox1", "co1": "cox1",
    "coxii": "cox2", "cox_ii": "cox2", "co2": "cox2",
    "coxiii": "cox3", "cox_iii": "cox3", "co3": "cox3",
    # CytB → cob
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
    """
    标准化单个基因名称。
    返回标准化名称，如果无法识别则返回 None（由调用方决定处理）。
    """
    name = name.strip()
    if not name:
        return None

    # ---------- 跳过MITOS2控制区标记 ----------
    # OH-a, -OH-b, OH-c, -oh 等均为控制区注释，不是基因
    cleaned = name.lstrip("-").lower()
    if cleaned.startswith("oh") or cleaned == "oh":
        return None

    # ---------- 去掉MITOS2部分注释的前缀"-" ----------
    # 如 -trnP 表示不完整的trnP注释，仍保留
    if name.startswith("-"):
        name = name[1:].strip()
        if not name:
            return None

    key = name.lower()

    # ---------- 查别名映射表 ----------
    if key in GENE_NAME_MAP:
        return GENE_NAME_MAP[key]

    # ---------- trnX 统一小写 ----------
    if key.startswith("trn"):
        return key

    # ---------- 蛋白编码基因和rRNA 统一小写 ----------
    known_bases = {
        "cox1", "cox2", "cox3", "cob", "atp6", "atp8",
        "nad1", "nad2", "nad3", "nad4", "nad4l", "nad5", "nad6",
        "rrns", "rrnl",
    }
    if key in known_bases:
        return key

    # ---------- 无法识别 ----------
    return None


def normalize_gene_order(raw_genes, keep_unknown=True):
    """
    标准化一个基因名称列表。
    
    参数:
        raw_genes: 原始基因名列表
        keep_unknown: True=将未知基因替换为UNKNOWN_GENE; False=直接丢弃
    
    返回:
        list[str]: 标准化后的基因顺序
    """
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
        # else: 丢弃
    return result


# ============================================================
# 模块1：参考数据库构建
# ============================================================

def build_reference_database(csv_path):
    """
    从CSV文件构建参考数据库。
    
    CSV格式要求:
        第一行为表头 (sample_id,standardized_gene_sequence)
        之后每行: 物种ID,"基因1,基因2,...,基因N"
    
    参数:
        csv_path: CSV文件路径
    
    返回:
        ref_db (dict):
            species_genes  : {物种ID: [基因列表]}
            species_list   : [物种ID] (CSV顺序)
            gene_to_idx    : {基因名: 索引}
            idx_to_gene    : {索引: 基因名}
            all_genes      : [所有基因名] (排序后)
            gene_num       : 基因种类总数
            num_classes    : 物种总数
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

    # 构建基因索引（排序保证可复现）
    all_genes = sorted(ref_db["gene_set"])
    # 确保UNKNOWN_GENE在索引中
    if UNKNOWN_GENE not in all_genes:
        all_genes.append(UNKNOWN_GENE)

    ref_db["all_genes"] = all_genes
    ref_db["gene_to_idx"] = {g: i for i, g in enumerate(all_genes)}
    ref_db["idx_to_gene"] = {i: g for i, g in enumerate(all_genes)}
    ref_db["gene_num"] = len(all_genes)
    ref_db["num_classes"] = len(ref_db["species_list"])

    # 打印统计
    lengths = [len(v) for v in ref_db["species_genes"].values()]
    unk_total = sum(
        1 for genes in ref_db["species_genes"].values() for g in genes if g == UNKNOWN_GENE
    )

    print(f"[模块1] 参考数据库构建完成")
    print(f"  物种数量    : {ref_db['num_classes']}")
    print(f"  基因种类    : {ref_db['gene_num']} (含 {UNKNOWN_GENE})")
    print(f"  基因数量范围: {min(lengths)} ~ {max(lengths)}")
    print(f"  未知基因标记: {unk_total} 个")

    return ref_db


# ============================================================
# 模块2：MITOS2注释调用
# ============================================================

def run_mitos(input_fasta, out_dir):
    """
    调用本地安装的MITOS2进行线粒体基因组注释。
    需要本地已配置好 mitos2 conda 环境。

    参数:
        input_fasta: 输入FASTA文件路径
        out_dir    : MITOS2输出目录

    返回:
        bool: 运行是否成功
    """
    os.makedirs(out_dir, exist_ok=True)

    cmd = (
        'bash -c \''
        'source /root/miniconda3/etc/profile.d/conda.sh && '
        'conda activate mitos2 && '
        f'runmitos --input {input_fasta} --code 2 '
        f'--outdir {out_dir} --refdir /root --refseqver refseq89o'
        '\''
    )

    print(f"[模块2] MITOS2注释启动")
    print(f"  输入: {input_fasta}")
    print(f"  输出: {out_dir}")

    ret = os.system(cmd)
    if ret == 0:
        print(f"[模块2] MITOS2注释完成")
        return True
    else:
        print(f"[模块2] MITOS2运行异常 (返回码: {ret})，请检查环境配置")
        return False


# ============================================================
# 模块3：基因顺序提取与标准化
# ============================================================

def extract_gene_order(out_dir, gene_to_idx=None):
    """
    从MITOS2输出目录中提取并标准化基因顺序。
    读取 out_dir/result.geneorder 文件。

    result.geneorder 格式示例:
        >Seq1
        trnL2 atp6 trnN ... -trnP OH-c ... -OH-b ... OH-a ... trnV

    参数:
        out_dir     : MITOS2输出目录
        gene_to_idx : 可选，参考数据库的基因索引（用于输出统计信息）

    返回:
        list[str]: 标准化后的基因名称列表
    """
    gene_order_file = os.path.join(out_dir, "result.geneorder")

    if not os.path.exists(gene_order_file):
        raise FileNotFoundError(f"未找到基因顺序文件: {gene_order_file}")

    with open(gene_order_file, "r") as f:
        lines = f.readlines()

    # 跳过header行 (以>开头)，取第一个非空非header行为基因序列
    gene_line = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        gene_line = stripped
        break

    if not gene_line:
        raise ValueError("result.geneorder 中未找到基因序列数据")

    # 空格分隔，逐个标准化
    raw_genes = gene_line.split()
    gene_order = normalize_gene_order(raw_genes, keep_unknown=True)

    # 以cox1为起始位置旋转（环状线粒体基因组惯例）
    if "cox1" in gene_order:
        idx = gene_order.index("cox1")
        gene_order = gene_order[idx:] + gene_order[:idx]
        print(f"[模块3] 已将cox1旋转至起始位置 (原位置: {idx})")
    else:
        print(f"[模块3] 警告: 未找到cox1基因，保持原始顺序")

    unk_count = sum(1 for g in gene_order if g == UNKNOWN_GENE)
    print(f"[模块3] 基因顺序提取完成")
    print(f"  基因数量: {len(gene_order)}")
    if unk_count > 0:
        print(f"  未知基因: {unk_count} 个")
    print(f"  顺序: {gene_order}")

    return gene_order


# ============================================================
# 模块4：GCN模型 - 数据构建与训练
# ============================================================

def genes_to_graph(gene_order, gene_to_idx, label=None):
    """
    将基因顺序列表转换为 PyTorch Geometric 图数据对象。

    节点特征: one-hot encoding（维度 = 基因种类数）
    边: 相邻基因之间的无向连接

    参数:
        gene_order : list[str], 基因名称列表
        gene_to_idx: dict, 基因名 → 索引
        label      : int or None, 物种标签（训练时传入）

    返回:
        torch_geometric.data.Data
    """
    num_gene_types = len(gene_to_idx)

    # one-hot 节点特征
    node_features = []
    for gene in gene_order:
        idx = gene_to_idx.get(gene, gene_to_idx.get(UNKNOWN_GENE, 0))
        feat = [0.0] * num_gene_types
        feat[idx] = 1.0
        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float)

    # 无向边: 相邻基因双向连接
    edges = []
    for i in range(len(gene_order) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    y = torch.tensor([label], dtype=torch.long) if label is not None else None

    return Data(x=x, edge_index=edge_index, y=y)


class GeneOrderDataset(torch.utils.data.Dataset):
    """将多条基因顺序封装为 PyG 数据集"""

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
    两层 GCN 分类器。
    forward(..., return_embedding=True) 返回图级嵌入向量，
    用于最近邻比较；默认返回分类 log_softmax 输出。
    """

    def __init__(self, num_node_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, return_embedding=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 图级表示
        graph_embedding = global_mean_pool(x, batch)

        if return_embedding:
            return graph_embedding

        logits = self.fc(graph_embedding)
        return F.log_softmax(logits, dim=1)


def train_gcn_model(ref_db, output_dir=None,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    hidden_dim=HIDDEN_DIM, lr=LR):
    """
    使用参考数据库训练 GCN 模型。

    参数:
        ref_db     : build_reference_database() 的返回值
        output_dir : 模型保存目录
        epochs     : 训练轮数
        batch_size : 批大小
        hidden_dim : GCN隐藏层维度
        lr         : 学习率

    返回:
        (model, label_encoder, train_dataset, val_dataset)
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.pth")
    gene_to_idx = ref_db["gene_to_idx"]
    num_features = ref_db["gene_num"]
    num_classes = ref_db["num_classes"]

    # ---- 准备训练数据（从参考数据库取所有物种的基因顺序） ----
    gene_orders = [ref_db["species_genes"][sp] for sp in ref_db["species_list"]]

    le = LabelEncoder()
    labels = le.fit_transform(ref_db["species_list"])

    # 按索引划分训练/验证集
    all_indices = list(range(len(gene_orders)))
    train_idx, val_idx = train_test_split(
        all_indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_orders = [gene_orders[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_orders = [gene_orders[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = GeneOrderDataset(train_orders, train_labels, gene_to_idx)
    val_dataset = GeneOrderDataset(val_orders, val_labels, gene_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ---- 初始化模型 ----
    model = GNNClassifier(num_features, hidden_dim, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    print(f"[模块4] 开始训练GCN模型")
    print(f"  训练集: {len(train_dataset)} 样本 | 验证集: {len(val_dataset)} 样本")
    print(f"  节点特征: {num_features}维 (one-hot) | 隐藏层: {hidden_dim} | 类别: {num_classes}")
    print(f"  设备: {DEVICE}")

    # ---- 训练循环 ----
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(train_dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
        val_acc = correct / len(val_dataset)

        if epoch % 10 == 0 or val_acc > best_val_acc:
            marker = " *" if val_acc > best_val_acc else ""
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_classes": list(le.classes_),
                "hidden_dim": hidden_dim,
                "num_node_features": num_features,
                "num_classes": num_classes,
            }, model_path)

    print(f"[模块4] 训练完成 | 最佳验证准确率: {best_val_acc:.4f}")
    print(f"  模型已保存: {model_path}")

    return model, le, train_dataset, val_dataset


# ============================================================
# 模块5：物种预测与最近邻检索
# ============================================================

def predict_species(query_gene_order, ref_db, model, le, train_dataset, top_k=3):
    """
    对查询基因顺序进行物种预测，并在训练集中检索最近邻。

    参数:
        query_gene_order : list[str], 查询样本的标准化基因顺序
        ref_db           : 参考数据库
        model            : 训练好的 GNN 模型
        le               : LabelEncoder (与训练时一致)
        train_dataset    : 训练数据集
        top_k            : 返回最近邻数量

    返回:
        dict: 包含 predicted_species, confidence, nearest_neighbors
    """
    gene_to_idx = ref_db["gene_to_idx"]

    # 构建查询图
    query_graph = genes_to_graph(query_gene_order, gene_to_idx)
    query_graph = query_graph.to(DEVICE)
    query_graph.batch = torch.zeros(query_graph.x.size(0), dtype=torch.long, device=DEVICE)

    model.eval()

    with torch.no_grad():
        query_embedding = model(query_graph, return_embedding=True).cpu().numpy()
        log_probs = model(query_graph).cpu().numpy()

    # 获取所有训练样本的嵌入
    train_embeddings = []
    with torch.no_grad():
        for data in train_dataset:
            data = data.to(DEVICE)
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
            embed = model(data, return_embedding=True).cpu().numpy()
            train_embeddings.append(embed)
    train_embeddings = np.vstack(train_embeddings)

    # 欧氏距离 → 最近邻
    distances = np.linalg.norm(train_embeddings - query_embedding, axis=1)
    nearest_indices = np.argsort(distances)[:top_k]

    nearest_neighbors = []
    for idx in nearest_indices:
        species = le.classes_[train_dataset[idx].y.item()]
        nearest_neighbors.append({
            "species": species,
            "distance": round(float(distances[idx]), 4),
        })

    # 分类预测
    predicted_idx = int(log_probs.argmax(axis=1)[0])
    predicted_species = le.classes_[predicted_idx]
    confidence = float(np.exp(log_probs[0, predicted_idx]))

    result = {
        "predicted_species": predicted_species,
        "confidence": round(confidence, 4),
        "nearest_neighbors": nearest_neighbors,
    }

    print(f"\n[模块5] 预测结果")
    print(f"  预测物种: {predicted_species} (置信度: {confidence:.4f})")
    print(f"  最近邻 {top_k}:")
    for i, nn in enumerate(nearest_neighbors, 1):
        print(f"    {i}. {nn['species']}  距离: {nn['distance']:.4f}")

    return result


# ============================================================
# 模块6：基因顺序可视化比对
# ============================================================

def plot_comparison(query_order, ref_orders, ref_names,
                    gene_to_idx, idx_to_gene,
                    save_path=None, show=False):
    """
    可视化比对查询样本与参考样本的基因顺序差异。

    参数:
        query_order : list[str], 查询基因顺序
        ref_orders  : list[list[str]], 参考基因顺序列表
        ref_names   : list[str], 参考物种名称
        gene_to_idx : dict
        idx_to_gene : dict
        save_path   : 图片保存路径 (None则不保存)
        show        : 是否弹窗显示 (需要非Agg后端)
    """
    n = min(len(ref_orders), 3)
    if n == 0:
        print("[模块6] 无参考序列可比对")
        return

    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]

        # 基因名 → 索引值用于Y轴绘图
        q_idx = [gene_to_idx.get(g, gene_to_idx.get(UNKNOWN_GENE, 0))
                 for g in query_order]
        r_idx = [gene_to_idx.get(g, gene_to_idx.get(UNKNOWN_GENE, 0))
                 for g in ref_orders[i]]

        ax.plot(range(len(q_idx)), q_idx, 'o-', label="Query",
                markersize=5, linewidth=1.2)
        ax.plot(range(len(r_idx)), r_idx, 's--',
                label=f"Ref: {ref_names[i]}",
                markersize=5, linewidth=1.2)

        # 标记差异位置（红色竖线）
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
        print(f"[模块6] 比对图已保存: {save_path}")
    if show:
        plt.show()

    plt.close()


# ============================================================
# 主流程
# ============================================================

def main(csv_path=DEFAULT_CSV_PATH, output_dir=DEFAULT_OUTPUT_DIR):
    """
    完整分析流程 (按顺序调用全部6个模块):
      1. 构建参考数据库
      2. 运行MITOS2注释
      3. 提取基因顺序
      4. 训练GCN模型
      5. 预测物种
      6. 可视化比对 + 保存结果
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- 步骤1: 构建参考数据库 ----
    print("=" * 60)
    ref_db = build_reference_database(csv_path)

    # ---- 步骤2: 获取输入 ----
    input_fasta = input("\n请输入待分析的线粒体基因组FASTA文件路径: ").strip()
    if not os.path.exists(input_fasta):
        print(f"错误: 文件不存在 - {input_fasta}")
        return None
    print("=" * 60)

    # ---- 步骤3: 运行MITOS2 ----
    mitos_ok = run_mitos(input_fasta, output_dir)
    if not mitos_ok:
        print("MITOS2运行失败，尝试使用已有的注释结果...")
        gene_order_file = os.path.join(output_dir, "result.geneorder")
        if not os.path.exists(gene_order_file):
            print(f"错误: 也未找到已有注释文件 {gene_order_file}")
            return None
    print("=" * 60)

    # ---- 步骤4: 提取基因顺序 ----
    try:
        gene_order = extract_gene_order(output_dir, ref_db["gene_to_idx"])
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return None
    print("=" * 60)

    # ---- 步骤5: 训练GCN ----
    model, le, train_dataset, val_dataset = train_gcn_model(ref_db, output_dir)
    print("=" * 60)

    # ---- 步骤6: 预测物种 ----
    result = predict_species(gene_order, ref_db, model, le, train_dataset)
    print("=" * 60)

    # ---- 步骤7: 可视化比对 ----
    nn_species = [nn["species"] for nn in result["nearest_neighbors"]]
    ref_orders = [ref_db["species_genes"][sp] for sp in nn_species
                  if sp in ref_db["species_genes"]]
    nn_valid = nn_species[:len(ref_orders)]

    plot_comparison(
        gene_order, ref_orders, nn_valid,
        ref_db["gene_to_idx"], ref_db["idx_to_gene"],
        save_path=os.path.join(output_dir, "gene_order_comparison.png"),
    )
    print("=" * 60)

    # ---- 步骤8: 保存结果JSON ----
    output_result = {
        "query_gene_order": gene_order,
        **result,
    }
    result_path = os.path.join(output_dir, "prediction_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 全部分析结果已保存至: {output_dir}")
    print(f"  - 模型权重 : {os.path.join(output_dir, 'best_model.pth')}")
    print(f"  - 比对图   : {os.path.join(output_dir, 'gene_order_comparison.png')}")
    print(f"  - 预测结果 : {result_path}")

    return output_result


if __name__ == "__main__":
    main()
