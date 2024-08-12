import torch

from transformers import (BertModel, RobertaModel)
from transformers.models.mpnet.modeling_mpnet import MPNetModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM, MptForCausalLM, Qwen2ForCausalLM, 
                          GemmaForCausalLM, MistralForCausalLM) # 
import time
from SentenceEmbedding.SentenceEmbedding import SentenceEmbeddingModel

def get_nonascii_toks(model_path, tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    non_ascii_toks = []
    ascii_toks = []
    if 'Llama-2' in model_path:
        # append 0 to 259
        non_ascii_toks = list(range(3, 259))
        for i in range(259, tokenizer.vocab_size):
            if not is_ascii(tokenizer.decode([i])):
                non_ascii_toks.append(i)
            else:
                ascii_toks.append(i)
    elif 'mpt' in model_path:
        for i in range(2, tokenizer.vocab_size):
            if not is_ascii(tokenizer.decode([i])):
                non_ascii_toks.append(i)
            else:
                ascii_toks.append(i)
    elif 'gemma' in model_path:
        # append 4 to 107
        non_ascii_toks = list(range(4, 107))
        for i in range(107, tokenizer.vocab_size):
            if not is_ascii(tokenizer.decode([i])):
                non_ascii_toks.append(i)
            else:
                ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        non_ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        non_ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        non_ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        non_ascii_toks.append(tokenizer.unk_token_id)
    
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
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embeddings(model, input_ids):
    if isinstance(model, SentenceEmbeddingModel):
        model = model.model

    # encode items and get the max length
    if isinstance(model, MPNetModel) or isinstance(model, RobertaModel):
        return model.embeddings.word_embeddings(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_fixed_list(model_path):
    if 'SBERT' in model_path:
        return ['!']
    else:
        raise ValueError(f"Unknown model type: {model_path}")
    