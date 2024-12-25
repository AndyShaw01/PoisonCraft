import torch

from transformers import (BertModel, RobertaModel, T5EncoderModel, MPNetModel)
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM, MptForCausalLM, Qwen2ForCausalLM, 
                          GemmaForCausalLM, MistralForCausalLM) # 
import time
from src.embedding.sentence_embedding import SentenceEmbeddingModel

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
    
import torch
import torch.nn.functional as F
import numpy as np
import logging

def initialize_logging():
    """Initialize the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def token_gradients(model, input_ids, question_embedding, control_slice):
    """
    Computes gradients of the loss with respect to control tokens.

    Args:
        model: The model to compute gradients.
        input_ids: The input token IDs.
        question_embedding: The question embedding for loss computation.
        control_slice: The slice of input IDs for control tokens.

    Returns:
        Gradients with respect to control tokens.
    """
    embed_weights = model.get_input_embeddings().weight
    one_hot = torch.zeros(
        input_ids[control_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[control_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # Combine with the rest of the embeddings
    embeds = model.get_input_embeddings()(input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, :control_slice.start, :],
            input_embeds,
            embeds[:, control_slice.stop:, :]
        ],
        dim=1
    )
    sentence_embedding = model(inputs_embeds=full_embeds).pooler_output
    cosine_similarity = F.cosine_similarity(sentence_embedding, question_embedding)
    loss = F.mse_loss(cosine_similarity, torch.tensor([1.0], device=model.device))
    loss.backward()

    return one_hot.grad.clone()


def get_loss(question_embedding, target_embeddings, mode='mse', reduction='none', dim=1):
    """
    Compute loss between question embedding and target embeddings.

    Args:
        question_embedding: Embedding of the query/question.
        target_embeddings: Embeddings of the target or candidates.
        mode: Type of loss function ('mse' by default).
        reduction: Reduction method ('none' or 'mean').
        dim: Dimension for cosine similarity.

    Returns:
        Computed loss tensor.
    """
    cosine_similarity = F.cosine_similarity(
        question_embedding, target_embeddings, dim=dim
    ) if dim == 1 else F.cosine_similarity(question_embedding, target_embeddings, dim=dim).mean(dim=0)
    label = torch.ones_like(cosine_similarity, device=question_embedding.device)

    if mode == 'mse':
        return F.mse_loss(cosine_similarity, label, reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss mode: {mode}")


def filter_candidates(control_cand, tokenizer, curr_control):
    """
    Filter invalid candidate control tokens.

    Args:
        control_cand: Candidate control token tensor.
        tokenizer: Tokenizer for decoding.
        curr_control: Current control tokens.

    Returns:
        Filtered list of valid candidates.
    """
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        try:
            decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True).strip()
            if decoded_str and decoded_str != tokenizer.decode(curr_control, skip_special_tokens=True).strip():
                cands.append(control_cand[i])
            else:
                count += 1
        except Exception as e:
            logging.warning(f"Error decoding candidate {i}: {e}")
            count += 1

    not_valid_ratio = count / len(control_cand)
    logging.warning(f"{not_valid_ratio:.2%} of control candidates were invalid.")
    return cands


def sample_control_tokens(grad, control_tokens, batch_size=128, topk=64, allow_non_ascii=False):
    """
    Sample new control tokens based on gradients.

    Args:
        grad: Gradients for control tokens.
        control_tokens: Current control tokens.
        batch_size: Number of samples to generate.
        topk: Top-k candidates to sample from.
        allow_non_ascii: Whether to allow non-ASCII tokens.

    Returns:
        New sampled control tokens.
    """
    if not allow_non_ascii:
        nonascii_mask = grad.new_zeros(grad.shape[1], dtype=torch.bool)
        grad[:, nonascii_mask] = -np.inf

    top_indices = (-grad).topk(topk, dim=1).indices
    original_control_tokens = control_tokens.repeat(batch_size, 1)

    new_token_positions = torch.arange(
        0, len(control_tokens), len(control_tokens) / batch_size, device=grad.device
    ).long()
    new_token_values = top_indices[new_token_positions, torch.randint(0, topk, (batch_size,), device=grad.device)]

    new_control_tokens = original_control_tokens.clone()
    new_control_tokens[torch.arange(batch_size), new_token_positions] = new_token_values
    return new_control_tokens


def get_nonascii_toks(model_path, tokenizer):
    """
    Retrieve non-ASCII tokens from the tokenizer.

    Args:
        model_path: Path to the model.
        tokenizer: Tokenizer instance.

    Returns:
        Tuple of non-ASCII tokens and ASCII tokens.
    """
    vocab = tokenizer.get_vocab()
    nonascii_toks = [tok_id for tok, tok_id in vocab.items() if not all(ord(c) < 128 for c in tok)]
    ascii_toks = [tok_id for tok, tok_id in vocab.items() if all(ord(c) < 128 for c in tok)]
    return torch.tensor(nonascii_toks), torch.tensor(ascii_toks)