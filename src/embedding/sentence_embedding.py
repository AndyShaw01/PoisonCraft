import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

class SentenceEmbeddingModel(nn.Module):
    """
    Sentence Embedding Model

    Args:
        model_path (str): model path
        device (int): device to use

    Returns:
        SentenceEmbeddingModel: model instance
    """
    def __init__(self, model_path, device=0):
        super(SentenceEmbeddingModel, self).__init__()
        self.model_path = model_path
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._load_model(model_path)
        self.model.to(self.device)
        if "ance" in model_path.lower():
            self.pooling, self.dense, self.layer_norm = self._get_ance_layers(model_path)
            self.dense.to(self.device)
            self.layer_norm.to(self.device)
            self.pooling.to(self.device)

    def _load_model(self, model_path):
        """load model and tokenizer"""
        if "contriever" in model_path or "simcse" in model_path or "bge" in model_path:
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        else:
            raise ValueError("Model not supported")

    def forward(self, sentences=None, input_ids=None, inputs_embeds=None):
        """compute sentence embeddings"""
        encoded_input = self._prepare_input(sentences, input_ids, inputs_embeds)
        model_output = self.model(**encoded_input)

        if "simcse" in self.model_path:
            return model_output.pooler_output

        if "bge" in self.model_path:
            cls_embeddings = model_output.last_hidden_state[:, 0, :]
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            return cls_embeddings
        return self.mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])

    def _prepare_input(self, sentences, input_ids, inputs_embeds):
        """prepare input for model"""
        if inputs_embeds is not None:
            return {'attention_mask': (inputs_embeds != 0).any(dim=-1).long(), 'inputs_embeds': inputs_embeds}
        elif input_ids is not None:
            return {'input_ids': input_ids, 'attention_mask': input_ids != 0}
        else:
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            return encoded_input

    def mean_pooling(self, token_embeddings, attention_mask):
        """calculate mean pooling"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def zero_grad(self):
        """zero grad for model"""
        self.model.zero_grad()
