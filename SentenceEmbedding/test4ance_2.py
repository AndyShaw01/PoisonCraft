from sentence_transformers import SentenceTransformer
import pdb
import torch
# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-ance-firstp')

# Encode some texts
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# save model

model.save("/data1/shaoyangguang/offline_model/ance")

# pdb.set_trace()
embeddings = torch.tensor(model.encode(sentences))
print(embeddings.shape)
# (3, 768)

# Get the similarity scores between all sentences
# similarities = model.similarity(embeddings, embeddings)
similarity = torch.mm(embeddings, embeddings.T)
print(similarity)
# print(embeddings[0])