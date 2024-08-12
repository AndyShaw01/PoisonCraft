from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Custom similarity model
class SentenceEmbeddingModel(nn.Module):
    def __init__(self, model_path):
        # model_path = "MPNetModel"
        super(SentenceEmbeddingModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, sentences=None, input_ids=None, inputs_embeds=None):
        if inputs_embeds is not None:  # Calculate embeddings from sentences directly
            # Tokenize sentences if input embeddings are not provided
            pdb.set_trace()
            model_output = self.model(inputs_embeds=inputs_embeds).last_hidden_state
            encoded_input = {'attention_mask': inputs_embeds != 0}
        elif input_ids is not None:  # Calculate embeddings from input ids
            model_output = self.model(input_ids=input_ids).last_hidden_state
            encoded_input = {'attention_mask': input_ids != 0}
        else:
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded_input)

        # Perform pooling
        pdb.set_trace()
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings

# Loss function based on cosine similarity
class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, embedding1, embedding2):
        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(embedding1, embedding2)
        # Loss is 1 - similarity
        loss = 1 - cosine_similarity
        pdb.set_trace()
        # Average the loss over the batch
        return loss.mean()


if __name__ == "__main__":
    # Example sentences A and B
    sentence_A = "I’m a student.!!!!!!!!!!!!"
    sentence_B = "Man? What can I say."

    # Instantiate model and loss function
    model_path = "./saved_model"
    similarity_model = SentenceEmbeddingModel(model_path)
    loss_function = SimilarityLoss()

    # Compute embeddings
    embeddings_A = similarity_model([sentence_A])
    embeddings_B = similarity_model([sentence_B])

    # Calculate loss
    loss = loss_function(embeddings_A, embeddings_B)

    print(f"Loss (1 - Cosine Similarity): {loss.item()}")