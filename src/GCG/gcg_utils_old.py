import torch

from transformers import (BertModel, RobertaModel, T5EncoderModel, MPNetModel)
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM, MptForCausalLM, Qwen2ForCausalLM, 
                          GemmaForCausalLM, MistralForCausalLM) # 
import time
from SentenceEmbedding.SentenceEmbedding import SentenceEmbeddingModel
def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()
    
    non_ascii_toks = []
    ascii_toks = []
    for i in range(tokenizer.vocab_size):
        try:
            token = tokenizer.decode([i])
        except:
            token = ''
        
        if not is_ascii(token):
            non_ascii_toks.append(i)
        else:
            ascii_toks.append(i)
    special_tokens = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id
    ]
    for tok in special_tokens:
        if tok is not None and tok not in non_ascii_toks:
            non_ascii_toks.append(tok)
    return torch.tensor(non_ascii_toks, device=device), torch.tensor(ascii_toks, device=device)


def get_embedding_weight(model):
    """
    Creates the batch of target texts with -1 placed at the end of the sequences for padding (for masking out the loss)
    """
    if isinstance(model, SentenceEmbeddingModel):
        model = model.model
    # encode items and get the max length
    if isinstance(model, MPNetModel) or isinstance(model, RobertaModel):
        return model.get_input_embeddings().weight
    elif isinstance(model, T5EncoderModel):
        return model.shared.weight
    elif isinstance(model, BertModel):
        if hasattr(model, 'name_or_path') and 'contriever' in model.name_or_path:
            return model.get_input_embeddings().weight
        else:# other bert models
            return model.get_input_embeddings().weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embeddings(model, input_ids):
    if isinstance(model, SentenceEmbeddingModel):
        model = model.model

    # encode items and get the max length
    if isinstance(model, MPNetModel) or isinstance(model, RobertaModel):
        return model.embeddings.word_embeddings(input_ids).half()
    elif isinstance(model, T5EncoderModel):
        return model.shared(input_ids).half()
    elif isinstance(model, BertModel):
        if hasattr(model, 'name_or_path') and 'contriever' in model.name_or_path:
            return model.embeddings.word_embeddings(input_ids).half()
        else:# other bert models
            return model.embeddings.word_embeddings(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_fixed_list(model_path):
    if 'MPNetModel' in model_path:
        return ['!']
    elif 't5' in model_path:
        return ['*']
    elif 'contriever' in model_path:
        return ['!']
    elif 'ance' in model_path:
        return ['!']
    else:
        raise ValueError(f"Unknown model type: {model_path}")
    