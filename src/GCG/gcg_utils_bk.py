import pdb
import torch
import torch.nn.functional as F
import numpy as np
import logging

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
    embed_weights = get_embedding_weight(model)
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
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, :control_slice.start, :],
            input_embeds,
            embeds[:, control_slice.stop:, :]
        ],
        dim=1
    )
    sentence_embedding = model(inputs_embeds=full_embeds)
    cosine_similarity = F.cosine_similarity(sentence_embedding, question_embedding)
    target = torch.ones_like(cosine_similarity, device=model.device)
    loss = F.mse_loss(cosine_similarity, target)
    loss.backward() 

    return one_hot.grad.clone()

def token_gradients_bp(model, input_ids, question_embedding, control_slice):
    """
    Computes gradients of the loss with respect to the coordinates.

    Args:
        model: The model to compute the gradients with respect to.
        input_ids: The input ids.
        question_embedding: The question embedding to calculate the loss
        control_slice: The slice of the input to compute the gradients with respect to.
    """
    embed_weights = get_embedding_weight(model)
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

    # now sitich it togethor with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, :control_slice.start, :],
            input_embeds,
            embeds[:, control_slice.stop:, :]
        ],
        dim=1
    )
    sentence_embedding = model(inputs_embeds=full_embeds)

    cosine_similarity = F.cosine_similarity(sentence_embedding, question_embedding)
    loss = F.mse_loss(cosine_similarity, torch.tensor([1.0], device=model.device))
    loss.backward()

    return one_hot.grad.clone()


def get_loss(question_embedding, target_embedding, mode='mse', reduction='none'):
    """
    Compute loss between query and target embedding(s).

    Args:
        question_embedding: Query embeddings. Shape: [N, 768] or [4, 768].
        target_embedding: Target embeddings. Shape: [4, 768] or [1, 768].
        mode: Type of loss function ('mse' by default).
        reduction: Reduction method ('none', 'mean', etc.).
        dim: Dimension for cosine similarity (1 for direct, 2 for batch).

    Returns:
        Computed loss tensor or scalar.
    """
    cosine_similarity = F.cosine_similarity(
        question_embedding.unsqueeze(0),    # [1, 4, 768]
        target_embedding.unsqueeze(1),   # [N, 1, 768]
        dim=2
    ) 
    mean_sim = cosine_similarity.mean(dim=1)
    label = torch.ones_like(mean_sim, device=mean_sim.device)

    if mode == 'mse':
        loss = F.mse_loss(mean_sim, label, reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss mode: {mode}")

    return loss


def get_loss_bp(question_embedding, target_embeddings, mode='mse',reduction='none', dim=1):
        """
        get the loss between the question embedding and the target embeddings

        Parameters
        ----------
        question_embedding : torch.tensor
            The question embedding, [sentence_embedding_dim]
        target_embeddings : torch.tensor
            The target embeddings, 
        
        Returns
        -------
        torch.tensor
            The loss between the question embedding and the target embeddings
        """
        cosine_similarity = F.cosine_similarity(question_embedding, target_embeddings, dim=dim) if dim == 1 else F.cosine_similarity(question_embedding, target_embeddings, dim=dim).mean(dim=0)
        label = torch.ones_like(cosine_similarity, device=question_embedding.device)
        if mode == 'mse':
            loss = F.mse_loss(cosine_similarity, label, reduction=reduction)
        else:
            ValueError("The mode is not supported")
        return loss
def filter_candidates_hard(control_cand, tokenizer, curr_control=None):
        """
        filter input candidates

        Args:
            control_cand : list
                The list of control tokens
            tokenizer : object
                The tokenizer object
            curr_control : list
                the current control token ids
        """
        if curr_control is None:
            raise Exception('Please provide the current control token ids')
        cands, count = [], 0
        for i in range(control_cand.shape[0]):
            try:
                decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
                # print("decoded_str": decoded_str)
                encoded_toks = tokenizer(decoded_str, add_special_tokens=False).input_ids
                encoded_toks = torch.tensor(encoded_toks, device=control_cand.device)

                if len(control_cand[i]) == len(encoded_toks) and not torch.all(torch.eq(control_cand[i], curr_control)):
                    # Important! add this to mitagate the situation that the encoded_tok is not equal to the origin one
                    if torch.all(torch.eq(control_cand[i], encoded_toks)):
                        cands.append(control_cand[i])
                    else:
                        count += 1
                else:
                    count += 1
            except IndexError:
                logging.error(f"IndexError: piece id is out of range for candidate at index {i}")
                count += 1
        not_valid_ratio = round(count / len(control_cand), 2)            
        logging.warning(f"Warning: {not_valid_ratio} control candidates were not valid")

        if not_valid_ratio > 0.1 and cands != []:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        elif cands == []:
            print("All the control candidates are not valid. Please check the initial control string.")
        return cands

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

def sample_control(self, grad, control_toks, batch_size, topk=256, allow_non_ascii=False):
        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        
        top_indices = (-grad).topk(topk, dim=1).indices
        # print('Shape of top_indices:', top_indices.shape)

        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size, 
            device=grad.device
        ).type(torch.int64)
        # print('the shape of new_token_pos is: ', new_token_pos.shape)

        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1), device=grad.device)
        )
        # print('the shape of new_token_val is: ', new_token_val.shape)

        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks

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