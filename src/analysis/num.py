import pandas as pd
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from embedding.sentence_embedding import SentenceEmbeddingModel

# 读取数据（确保文件有 'query' 字段）
shadow_queries_path = './Datasets/nq/train_queries.jsonl'
target_queries_path = './Datasets/nq/test_queries.jsonl'

shadow_queries = pd.read_json(shadow_queries_path, lines=True)
target_queries = pd.read_json(target_queries_path, lines=True)

# 提取query文本列表
shadow_queries_text = shadow_queries['text'].tolist()
target_queries_text = target_queries['text'].tolist()

# 加载模型
model_path = '/data1/shaoyangguang/offline_model/contriever'
model = SentenceEmbeddingModel(model_path, device=2)

# 批量编码函数
def batch_encode(model, texts, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model(batch)  # 假设 model.encode 返回 [batch_size, emb_dim] 的张量
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)

# 对shadow与target进行编码
shadow_embeddings = batch_encode(model, shadow_queries_text, batch_size=16)
target_embeddings = batch_encode(model, target_queries_text, batch_size=16)

# 计算相似度
all_sims = []
with torch.no_grad():
    # 遍历每个target embedding
    for t_emb in target_embeddings:
        # t_emb: [emb_dim]
        # shadow_embeddings: [num_shadow, emb_dim]
        sim = F.cosine_similarity(t_emb.unsqueeze(0), shadow_embeddings, dim=-1)  # [num_shadow]
        all_sims.append(sim)

all_sims = torch.cat(all_sims, dim=0)

mean_sim = all_sims.mean().item()
std_sim = all_sims.std().item()

print("Mean similarity across all target-shadow pairs:", mean_sim)
print("Standard deviation of similarity:", std_sim)