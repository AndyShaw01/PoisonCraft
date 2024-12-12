import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MPNetModel, T5EncoderModel, T5Tokenizer, MPNetTokenizer, AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel, DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from sentence_transformers.models import Pooling
from sentence_transformers import SentenceTransformer

import pdb


model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-question_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-question_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-ctx_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_pooling_results(token_embeddings, attention_mask, pooling_mode='mean'):
    pooling = Pooling(word_embedding_dimension=768 ,pooling_mode=pooling_mode)
    standard_out = {}
    standard_out['token_embeddings'] = token_embeddings
    standard_out['attention_mask'] = attention_mask
    pooling_out = pooling(standard_out)
    return pooling_out['sentence_embedding']

# Custom similarity model
class SentenceEmbeddingModel(nn.Module):
    def __init__(self, model_path, device=0):
        self.model_path = model_path
        super(SentenceEmbeddingModel, self).__init__()
        if "MPNetModel" in model_path:
            self.model = MPNetModel.from_pretrained(model_path)
            self.tokenizer = MPNetTokenizer.from_pretrained(model_path)
        elif "t5" in model_path:
            self.model = T5EncoderModel.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
        elif "contriever" in model_path:
            self.model = AutoModel.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif "ance" in model_path:
            # self.model = RobertaModel.from_pretrained(model_path)
            # self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            ance_model = SentenceTransformer(model_path)
            # 提取模型组件
            self.model = ance_model[0].auto_model  # RobertaModel
            self.pooling = ance_model[1]  # Pooling layer
            self.dense = ance_model[2]    # Dense layer
            self.layer_norm = ance_model[3]  # LayerNorm layer
            self.tokenizer = ance_model.tokenizer
            del ance_model
            # pdb.set_trace()
        elif "dpr-c" in model_path:
            self.model = DPRContextEncoder.from_pretrained(model_path)
            self.tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path)
        elif "dpr-q" in model_path:
            self.model = DPRQuestionEncoder.from_pretrained(model_path)
            self.tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path)
        else:
            ValueError("Model not supported")
        # pdb.set_trace()
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if "ance" in self.model_path.lower():
            self.dense.to(self.device)
            self.layer_norm.to(self.device)
            self.pooling.to(self.device)
    
    def forward_new(self, sentences=None, input_ids=None, inputs_embeds=None):
        if inputs_embeds is not None:
            # 处理 inputs_embeds 的情况
            encoded_input = {'attention_mask': (inputs_embeds != 0).any(dim=-1).long()}
            model_output = self.model(inputs_embeds=inputs_embeds)
        elif input_ids is not None:
            # 确保 input_ids 是二维的
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                print("Unsqueezed input_ids to add batch dimension.")
            
            # 确保 input_ids 是 LongTensor
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
                print("Converted input_ids to torch.long.")
            
            # 确保 input_ids 在正确的设备上
            input_ids = input_ids.to(self.device)
            
            # 打印 input_ids 的形状和类型以供调试
            print(f"input_ids dtype: {input_ids.dtype}, shape: {input_ids.shape}, device: {input_ids.device}")
            
            # 创建 attention_mask
            encoded_input = {'attention_mask': input_ids != 0}
            
            # 调用模型
            try:
                model_output = self.model(input_ids=input_ids, attention_mask=encoded_input['attention_mask'])
                print("Model output obtained successfully.")
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                raise e
        else:
            # 处理 sentences 的情况
            encoded_input = self.tokenizer(
                sentences, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            model_output = self.model(**encoded_input)
        
        # 处理模型输出
        if "dpr" in self.model_path:
            sentence_embeddings = model_output.pooler_output
        elif "ance" in self.model_path:
            standard_out = {
                'token_embeddings': model_output.last_hidden_state,
                'attention_mask': encoded_input['attention_mask']
            }
            features = self.pooling(standard_out)
            features = self.dense(features)
            features = self.layer_norm(features)
            sentence_embeddings = features['sentence_embedding']
        else:
            sentence_embeddings = get_pooling_results(
                model_output.last_hidden_state, 
                encoded_input['attention_mask'], 
                pooling_mode='mean'
            )
        
        return sentence_embeddings

    def forward(self, sentences=None, input_ids=None, inputs_embeds=None):
        if inputs_embeds is not None:  # Calculate embeddings from sentences directly
            # Tokenize sentences if input embeddings are not provided
            encoded_input = {'attention_mask': (inputs_embeds != 0).any(dim=-1).long()}
            model_output = self.model(inputs_embeds=inputs_embeds)
        elif input_ids is not None:  # Calculate embeddings from input ids
            encoded_input = {'attention_mask': input_ids != 0}
            model_output = self.model(input_ids=input_ids, attention_mask=encoded_input['attention_mask'])
            print(model_output)
        else:                
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded_input)
            
        if "dpr" in self.model_path:
            sentence_embeddings = model_output.pooler_output
        if "ance" in self.model_path:
            standard_out = {
                'token_embeddings': model_output.last_hidden_state,
                'attention_mask': encoded_input['attention_mask']
            }
            features = self.pooling(standard_out)
            features = self.dense(features)
            features = self.layer_norm(features)
            sentence_embeddings = features['sentence_embedding']
        else:
            sentence_embeddings = get_pooling_results(model_output.last_hidden_state, encoded_input['attention_mask'], pooling_mode='mean')
        
        return sentence_embeddings
    def zeio_grad(self):
        """
        Zero out the gradients of the model's parameters.
        """
        self.model.zero_grad()

if __name__ == "__main__":
    # test
    model_path = "/data1/shaoyangguang/offline_model/ance"