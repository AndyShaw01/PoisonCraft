import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import gc
import numpy as np
from copy import deepcopy
import pandas as pd
import csv
import pdb
import logging
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM)
from GCG.gcg_utils import get_nonascii_toks, get_embedding_weight, get_embeddings, get_fixed_list
from embedding.SentenceEmbedding import SentenceEmbeddingModel, get_pooling_results

def token_gradients(model, input_ids, question_embedding, control_slice):
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

class GCG:
    """
    Run GCG Attack.

    Parameters
    ----------
    control_string_length : int 
        The length of the control string.
    max_steps : int
        The maximum number of steps in a single attack attempt
    max_attack_attempts : int
        The maximum number of attack attempts
    max_prompts_in_single_attack : int
        The maximum number of prompts in a single attack attempt. We do not want to get too many prompts in a single attack attempt because the diversity of the prompts cannot be guaranteed.
    max_successful_prompt : int
        The maximum number of successful prompts to get
    max_attack_steps : int
        The maximum number of total attack steps
    
    Returns
    -------
    list
        The list of successful prompts
    list
        The list of steps for each successful prompt
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_path = args.model_path
        self.model = SentenceEmbeddingModel(self.model_path).to(self.device)  
        self.tokenizer = self.model.tokenizer
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        self.control_string_length = args.control_string_length
        self.max_steps = args.max_steps
        self.early_stop = args.early_stop
        self.control_string_len = args.control_string_length
        self.question = args.question
        self.max_attack_attempts = args.max_attack_attempts
        self.max_prompts_in_single_attack = args.max_prompts_in_single_attack
        self.max_successful_prompt = args.max_successful_prompt
        self.max_attack_steps = args.max_attack_steps
        self.loss_threshold = args.loss_threshold if hasattr(args, 'loss_threshold') else 0.5
        self.early_stop_iterations = args.early_stop_iterations if hasattr(args, 'early_stop_iterations') else 500
        self.early_stop_local_optim = args.early_stop_local_optim if hasattr(args, 'early_stop_local_optim') else 100
        self.update_token_threshold = args.update_token_threshold if hasattr(args, 'update_token_threshold') else 5
        self.no_space = False
        self.attack_batch_size = args.attack_batch_size
        self.attack_batch_mode = args.attack_batch_mode
        self.setup()
        self.result_file = args.save_path
        if self.result_file is None:
            self.result_file = f'results-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv'
        self.raw_fp = open(self.result_file, 'w', buffering=1)
        self.writter = csv.writer(self.raw_fp)
        self.writter.writerow(
            ['index', 'question','control_suffix', 'loss', 'attack_steps', 'attack_attempt'])
        

    def setup(self):
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s %(message)s', datefmt='[%H:%M:%S]')

    def init_adv_postfix(self, random=True):
        """
        generate a fixed control string with the given length
        the control tokens can be randomly sampled from the following list
        """
        cand_toks = []
        while len(cand_toks) != self.control_string_len:
            if random:
                cand_list = ['when', 'the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by',
                             'this', 'with', 'it', 'from', 'at', 'are', 'as', 'be', 'was']
            else:
                cand_list = get_fixed_list(self.args.model_path)
            cand = np.random.choice(cand_list, self.control_string_len)
            cand_str = ' '.join(cand)
            cand_toks = self.tokenizer.encode(cand_str, add_special_tokens=False)
        return cand_str, cand_toks

    def get_loss(self, question_embedding, target_embeddings, mode='mse',reduction='none', dim=1):
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
        label = torch.ones_like(cosine_similarity, device=self.model.device)
        if mode == 'mse':
            loss = F.mse_loss(cosine_similarity, label, reduction=reduction)
        else:
            ValueError("The mode is not supported")
        return loss

    def get_filtered_cands_ance(self, control_cand, tokenizer, curr_control=None):
        """
        Filter input candidates with adjusted logic.

        Parameters
        ----------
        control_cand : torch.Tensor
            The list of control tokens (shape: [num_candidates, token_length])
        tokenizer : object
            The tokenizer object
        curr_control : torch.Tensor
            The current control token ids (shape: [token_length])

        Returns
        -------
        list of torch.Tensor
            The filtered list of control token tensors.
        """
        import logging

        if curr_control is None:
            raise Exception('Please provide the current control token ids')

        cands, count = [], 0
        for i in range(control_cand.shape[0]):
            try:
                decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
                # print(f"Candidate {i}: Decoded string: '{decoded_str}'")

                # Basic validity check: non-empty string
                if decoded_str.strip() == "":
                    count += 1
                    print(f"Candidate {i} is invalid because decoded string is empty.")
                    continue

                # Optionally, add more conditions based on your requirements
                # For example, minimum length, presence of certain characters, etc.

                # Check if the candidate is different from the current control
                if not tokenizer.decode(curr_control, skip_special_tokens=True).strip() == decoded_str.strip():
                    cands.append(control_cand[i])
                    # print(f"Candidate {i} is valid and added to cands.")
                else:
                    count += 1
                    print(f"Candidate {i} is invalid because it is identical to curr_control.")

            except IndexError:
                logging.error(f"IndexError: piece id is out of range for candidate at index {i}")
                count += 1
            except Exception as e:
                logging.error(f"Unexpected error for candidate {i}: {e}")
                count += 1

        not_valid_ratio = round(count / len(control_cand), 2)
        logging.warning(f"Warning: {not_valid_ratio} control candidates were not valid")

        if not_valid_ratio > 0.1 and cands:
            # Replicate the last valid candidate to match the required length
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
            print(f"Filtered cands are less than control_cand. Replicating the last candidate to match the length.")
        elif not cands:
            print("All the control candidates are not valid. Please check the initial control string.")
            # Optionally, return a default candidate to prevent empty input_ids
            default_cand = torch.tensor([tokenizer.cls_token_id], device=control_cand.device)
            cands.append(default_cand)

        return cands

    def get_filtered_cands(self, control_cand, tokenizer, curr_control=None):
        """
        filter input candidates

        Parameters
        ----------
        control_cand : list
            The list of control tokens
        tokenizer : object
            The tokenizer object
        curr_control : list
            the current control token ids
        
        Returns
        -------
        """
        # pdb.set_trace()

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
    

    def run(self, target):
        self._nonascii_toks, self._ascii_toks = get_nonascii_toks(self.args.model_path, self.tokenizer)
        
        optim_prompts = []
        optim_steps = []
        attack_attempt = 0
        attack_steps = 0
        store_control_str = None
        while len(optim_prompts) < self.max_successful_prompt and attack_attempt < self.max_attack_attempts and attack_steps < self.max_attack_steps:
            attack_attempt += 1
            curr_optim_prompts = []
            curr_optim_steps = []
            last_optim_steps = 0
            best_loss = 999999
            end_iter = False
            logging.info(f"Attack attempt {attack_attempt}")
            control_tokens = []
            while len(control_tokens) != self.control_string_len:
                # when random is true, sometimes the control token will not be tokenized as expected length
                curr_prompt = target
                toks = self.tokenizer(curr_prompt).input_ids
                target_slice = slice(1, len(toks)-1)
                control_str, _ = self.init_adv_postfix()
                if self.no_space: # 不同的LLM之间的小区别，如果跑Llama3，需要测试是不是需要加。tokenizer 会自动减空格
                    curr_prompt = curr_prompt + control_str
                else:
                    curr_prompt = curr_prompt + ' ' + control_str
                toks = self.tokenizer(curr_prompt).input_ids
                control_slice = slice(target_slice.stop, len(toks)-1)
                control_tokens = torch.tensor(toks[control_slice], device=self.device)
            logging.info(f"Control string: {control_str}")
            logging.info(f"Question string: {self.question}")
            logging.info(f"Current prompt: {curr_prompt}")
            # pdb.set_trace()
            question_embedding = self.model(self.question).detach()

            self.model.zero_grad()

            input_ids = torch.tensor(toks, device=self.device)
            target_embedding = self.model(input_ids=input_ids.unsqueeze(0))
            
            # ========== setup initial loss========== # 
            if self.attack_batch_size > 1:
                initial_loss = self.get_loss(question_embedding, target_embedding, reduction="mean", dim=1)
            else:
                initial_loss = self.get_loss(question_embedding, target_embedding)

            logging.info(f"Initial loss: {initial_loss.item()}")
            # ========== start attack ========== #
            local_optim_counter = 0
            update_toks = 0
            best_loss = 999999

            for i in range(self.max_steps):

                # The loss is too high for a long time
                if self.early_stop and i > self.early_stop_iterations and best_loss > self.loss_threshold:
                    logging.info('early stop by loss at {}-th step'.format(i))
                    break

                # The loss does not decrease for a long time
                if self.early_stop and local_optim_counter >= self.early_stop_local_optim:
                    logging.info("early stop by local optim at {}-th step".format(i))
                    break   

                if end_iter: 
                    logging.info("End the iteration")
                    break 
                
                if attack_steps >= self.max_attack_steps:
                    logging.info("Reach the maximum attack steps")
                    break
                    
                if i != 0:
                    input_ids[control_slice] = control_tokens
                attack_steps += 1
                grad = token_gradients(self.model, input_ids, question_embedding.detach(), control_slice)
                averaged_grad = grad / grad.norm(dim=-1, keepdim=True)

                candidates = []
                batch_size = 128
                topk = 64
                filter_cand=True
                with torch.no_grad():
                    # pdb.set_trace()
                    control_cand = self.sample_control(averaged_grad, control_tokens, batch_size, topk) # 128, 20
                    if filter_cand:
                        candidates.append(self.get_filtered_cands(control_cand, self.tokenizer, control_tokens)) # [[]]
                    else:
                        candidates.append(control_cand)
                del averaged_grad, control_cand ; gc.collect()
                curr_best_loss = 999999
                curr_best_control_tokens = None

                candidates = candidates[0] # [[]]->[]

                with torch.no_grad():
                    inputs = torch.tensor([], device=self.device)
                    for cand in candidates: # cand-len: 20
                        tmp_input = input_ids.clone()
                        tmp_input[control_slice] = cand
                        if inputs.shape[0] == 0:
                            inputs = tmp_input.unsqueeze(0) # [1, len]
                        else:
                            inputs = torch.cat((inputs, tmp_input.unsqueeze(0)), dim=0) # [N, len]
                    
                    target_embeddings = self.model(input_ids=inputs)
                    if self.attack_batch_size > 1:
                        losses = self.get_loss(question_embedding.unsqueeze(1), target_embeddings.unsqueeze(0), dim=2) # get [128] loss
                    else:
                        losses = self.get_loss(question_embedding, target_embeddings)
                    # del inputs ; gc.collect()
                    losses[torch.isnan(losses)] = 999999
                    curr_best_loss, best_idx = torch.min(losses, dim=0)
                    curr_best_control_tokens = candidates[best_idx]
                    del inputs ; gc.collect()
                if self.attack_batch_mode == "mean":
                    logging.info(f"current best loss: {curr_best_loss.item()}")
                else:
                    logging.info(f"current best loss: {curr_best_loss.item()}")

                if curr_best_loss < best_loss:
                    update_toks += 1
                    local_optim_counter = 0
                    best_loss = curr_best_loss
                    control_tokens = deepcopy(curr_best_control_tokens) # in the next iteration, we will use the best control tokens to update the input_ids
                    logging.info("Step: {}, Loss: {}".format(i, best_loss.data.item()))
                    current_control_str = self.tokenizer.decode(curr_best_control_tokens)
                else:
                    if isinstance(best_loss, int):
                        logging.info("After {} iterations, the best loss is: ".format(i, best_loss))
                    else:
                        logging.info("After {} iterations, the best loss is: ".format(i, best_loss.data.item()))

                    local_optim_counter += 1
                    print("local_optim_counter: ", local_optim_counter)

                del candidates, tmp_input, losses ; gc.collect()
                self.model.zeio_grad()
            
            if isinstance(best_loss, int):
                logging.info("In this attempt, after {} iterations, the best loss is: {}".format(i, best_loss))
            else:
                logging.info("In this attempt, after {} iterations, the best loss is: {}".format(i, best_loss.data.item()))
            
            logging.info('In {} attemp, number of optim prompts is: {}'.format(attack_attempt, len(curr_optim_prompts)))

            optim_prompts.append(current_control_str)
            optim_steps.append(attack_steps)
            
            logging.info('After {} attemp, the total number of optim prompts is: {}'.format(attack_attempt, len(optim_prompts)))
        
        for i in range(len(optim_prompts)):
            self.writter.writerow([self.args.index, self.question, optim_prompts[i], best_loss.data.item(), optim_steps[i], attack_attempt])

        
        