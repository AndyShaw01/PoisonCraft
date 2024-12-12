from SentenceEmbedding import SentenceEmbeddingModel
import pdb
import torch

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# Load the model
model_path = "/data1/shaoyangguang/offline_model/ance"
model = SentenceEmbeddingModel(model_path).eval()

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# Compute embeddings
embeddings = model(sentences)
# Compute dot product of the embeddings
similarity = torch.mm(embeddings, embeddings.T)

print(similarity)
# print(embeddings[0])