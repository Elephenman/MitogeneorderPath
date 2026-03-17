GB_FILES = "./数据集.gb"
GENE_ORDER_CSV_PATH = "./geneorder.csv"
input_fasta = input("请输入待分析的线粒体基因组FASTA文件路径：").strip()
OUTPUT_FOLDER = "./mito_analysis_results/"
 
import os
import numpy as np
import pandas as pd 
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
# 自动创建输出文件夹
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
RESULT_LOG_PATH = os.path.join(OUTPUT_FOLDER, "analysis_result.txt")

#  模块1：参考数据库自动构建
def build_reference_database(csv_path):
    df = pd.read_csv(csv_path, header=None)
    ref_db = {
        "species_genes": defaultdict(list),  # 物种: 基因列表
        "gene_to_idx": {},                   # 基因: 标准索引
        "idx_to_gene": {},                   # 索引: 基因
        "gene_num": 0,
        "species_list": []                    # 物种列表
    }
    
    # 处理每一行数据
    for _, row in df.iterrows():
        species = row[0]
        gene_str = str(row[1])
        
        # 分割基因序列
        genes = [g.strip() for g in gene_str.split(",")]
        
        # 添加到数据库
        ref_db["species_genes"][species] = genes
        ref_db["species_list"].append(species)
        labels=ref_db["species_list"]
        # 如果是第一个物种，设置为标准基因顺序
        if ref_db["gene_num"] == 0:
            ref_db["gene_num"] = len(genes)
            ref_db["gene_to_idx"] = {gene: idx for idx, gene in enumerate(genes)}
            ref_db["idx_to_gene"] = {idx: gene for idx, gene in enumerate(genes)}
    
    return ref_db,labels
ref_db, labels = build_reference_database(GENE_ORDER_CSV_PATH)
#模块2：用mitos2注释  得用linux命令行调用，python里调用系统命令
def run_mitos(input_fasta, out_dir):
    # 核心命令修正为 runmitos（无.py后缀）
    cmd = f"""
    bash -c '
        source /root/miniconda3/etc/profile.d/conda.sh && \
        conda activate mitos2 && \
        runmitos \
            --input {input_fasta} \
            --code 2 \
            --outdir {out_dir} \
            --refdir /root \
            --refseqver refseq89o
    '
    """
    os.makedirs(out_dir, exist_ok=True)
    os.system(cmd)
    print(f"MITOS已后台启动，输出目录：{out_dir}")

#模块3：注释文件基因顺序提取以及标准化
def gene_order(out_dir):
    gene_order_file = os.path.join(out_dir, "result.geneorder")
    gene_order = []
    with open(gene_order_file, "r") as f:
        line = f.readlines()  
        gene_line = line[1].strip()  
        gene_order = gene_line.split()  
    print(gene_order)
    if 'cox1' in gene_order:
        idx = gene_order.index('cox1')
        gene_order = gene_order[idx:] + gene_order[:idx]
    else:
        print(gene_order)
        print("警告：未找到cox1基因，保持原始顺序")
    return gene_order
gene_order = gene_order(OUTPUT_FOLDER)
print("基因顺序提取完成，结果如下：")
print(gene_order)

#模块4：训练图嵌入模型

# 参数配置
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_CLASSES = len(ref_db["species_list"])  # 假设是多分类任务
EPOCHS = 100
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将基因顺序转换为图数据集
class GeneOrderDataset(torch.utils.data.Dataset):
    def __init__(self, gene_orders, labels, ref_db):
        self.data_list = []
        
        for idx, gene_order in enumerate(gene_orders):
            node_features = []
            for gene in gene_order:
                node_features.append(ref_db["gene_to_idx"].get(gene, -1))  # 未找到的基因标记为-1
            num_nodes = len(node_features)
            edges = []
            for i in range(num_nodes - 1):
                edges.append([i, i+1])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            data = Data(
                x=torch.tensor(node_features, dtype=torch.long).unsqueeze(-1),  # [num_nodes, 1]
                edge_index=edge_index,
                y=torch.tensor(labels[idx], dtype=torch.long)
            )
            self.data_list.append(data)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]     

# 编码标签
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# 划分训练集和验证集（保持类别比例）
train_genes, val_genes, train_labels, val_labels = train_test_split(
    gene_order, labels_encoded, 
    test_size=0.2, stratify=labels_encoded, random_state=42
)

# 创建PyTorch Geometric数据集和数据加载器
train_dataset = GeneOrderDataset(train_genes, train_labels, ref_db)
val_dataset = GeneOrderDataset(val_genes, val_labels, ref_db)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


class GNNClassifier(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 图卷积层
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 全局平均池化
        x = global_mean_pool(x, data.batch)
        
        # 输出层
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    # 初始化模型、优化器和损失函数
model = GNNClassifier(
    num_node_features=1,  # 因为节点特征是单个整数（gene_to_idx的索引）
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.NLLLoss()  # 因为模型输出是log_softmax

# 训练循环
best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    
    avg_train_loss = total_loss / len(train_loader.dataset)
    
    # 验证阶段
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
    
    val_acc = correct / len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER, "best_model.pth"))

print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

model = GNNClassifier(
    num_node_features=1,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(OUTPUT_FOLDER, "best_model.pth")))
model.eval()
new_data = Data(
    x=torch.tensor([ref_db["gene_to_idx"].get(g, -1) for g in gene_order], dtype=torch.long).unsqueeze(-1),
    edge_index=torch.tensor([[i, i+1] for i in range(len(gene_order)-1)], dtype=torch.long).t().contiguous()
)
new_embedding = model(new_data).detach().numpy()  # [1, HIDDEN_DIM]

# 计算所有训练样本的嵌入向量（仅需运行一次）
train_embeddings = []
train_labels_all = []

with torch.no_grad():
    for batch in train_loader:
        batch = batch.to(DEVICE)
        embed = model(batch)
        train_embeddings.append(embed.cpu().numpy())
        train_labels_all.extend(batch.y.tolist())

train_embeddings = np.vstack(train_embeddings)
# 计算欧氏距离
distances = np.linalg.norm(train_embeddings - new_embedding, axis=1)
indices = np.argsort(distances)[:3]  # 最近的3个样本索引

# 获取对应标签和距离
nearest_labels = [ref_db["species_list"][i] for i in indices]
nearest_distances = distances[indices]

print("最接近的3个物种及距离：")
for label, dist in zip(nearest_labels, nearest_distances):
    print(f"{label}: {dist:.4f}")
# 绘制对比图
def plot_comparison(query_order, ref_order, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(query_order)), query_order, 'o-', label="Query")
    ax.plot(range(len(ref_order)), ref_order, 's-', label="Reference")
    
    # 标记不同基因
    for i in range(len(query_order)):
        if query_order[i] != ref_order[i]:
            ax.annotate("", xy=(i, query_order[i]), xytext=(i, ref_order[i]),
                        arrowprops=dict(arrowstyle="->", color="red"))
    
    ax.set_xticks([])
    ax.set_yticks(range(len(ref_db["gene_to_idx"])))
    ax.set_yticklabels(ref_db["idx_to_gene"].values())
    ax.set_title(title)
    ax.legend()
    plt.show()

plot_comparison(gene_order, train_genes[indices[0:2]], "Gene Order Comparison")

