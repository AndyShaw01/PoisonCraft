import time
import gc
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import logging
from src.GCG.gcg_utils import (
    token_gradients,
    initialize_logging,
    get_nonascii_toks,
    get_fixed_list
)
from src.embedding.sentence_embedding import SentenceEmbeddingModel

import pdb

class GCG:
    def __init__(self, args, question=None):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model and tokenizer setup
        self.model = SentenceEmbeddingModel(args.model_path).to(self.device)
        self.tokenizer = self.model.tokenizer
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token

        # Attack parameters
        self.control_string_length = args.control_string_length
        self.max_steps = args.max_steps
        self.max_attack_attempts = args.max_attack_attempts
        self.max_successful_prompt = args.max_successful_prompt
        self.max_attack_steps = args.max_attack_steps
        self.loss_threshold = getattr(args, 'loss_threshold', 0.5)
        self.early_stop = getattr(args, 'early_stop', True)
        self.early_stop_iterations = getattr(args, 'early_stop_iterations', 300)
        self.early_stop_local_optim = getattr(args, 'early_stop_local_optim', 100)
        self.batch_size = args.batch_size
        self.no_space = False

        # Logging setup
        initialize_logging()
        self._nonascii_toks, self._ascii_toks = get_nonascii_toks(self.tokenizer)
        
        # Results file
        self.result_file = args.save_path or f'results-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv'
        self.result_writer = self._initialize_result_writer()

        self.question = question

    def _initialize_result_writer(self):
        """Initialize the result CSV writer."""
        import csv
        fp = open(self.result_file, 'w', buffering=1)
        writer = csv.writer(fp)
        writer.writerow(['index', 'question', 'control_suffix', 'loss', 'attack_steps', 'attack_attempt'])
        return writer
    def init_adv_suffix(self, random=False):
        """
        generate a fixed control string with the given length
        the control tokens can be randomly sampled from the following list
        """
        cand_toks = []
        while len(cand_toks) != self.control_string_length:
            if random:
                cand_list = ['when', 'the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by',
                             'this', 'with', 'it', 'from', 'at', 'are', 'as', 'be', 'was']
            else:
                cand_list = get_fixed_list(self.args.model_path)
            cand = np.random.choice(cand_list, self.control_string_length)
            cand_str = ' '.join(cand)
            cand_toks = self.tokenizer.encode(cand_str, add_special_tokens=False)
        return cand_str, cand_toks
    
    def get_loss(self, question_embedding, target_embeddings, mode='mse',reduction='none', dim=1):
        cosine_similarity = F.cosine_similarity(question_embedding, target_embeddings, dim=dim) if dim == 1 else F.cosine_similarity(question_embedding, target_embeddings, dim=dim).mean(dim=0)
        label = torch.ones_like(cosine_similarity, device=self.model.device)
        if mode == 'mse':
            loss = F.mse_loss(cosine_similarity, label, reduction=reduction)
        else:
            ValueError("The mode is not supported")
        return loss

    def get_filtered_cands(self, control_cand, tokenizer, curr_control=None):
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
                # logging.error(f"IndexError: piece id is out of range for candidate at index {i}")
                count += 1
        not_valid_ratio = round(count / len(control_cand), 2)            
        logging.warning(f"Warning: {not_valid_ratio} control candidates were not valid")

        if not_valid_ratio > 0.1 and cands != []:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        elif cands == []:
            print("All the control candidates are not valid. Please check the initial control string.")
        return cands
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
    
    def attack_iteration(self, question_embedding, control_tokens, input_ids, control_slice):
        # grad -> filtered_candidates
        grad = token_gradients(self.model, input_ids, question_embedding.detach(), control_slice)
        averaged_grad = grad / grad.norm(dim=-1, keepdim=True)
        candidates = []
        batch_size = 128
        topk = 64
        filter_cand=True
        with torch.no_grad():
            control_cand = self.sample_control(averaged_grad, control_tokens, batch_size, topk) # 128, 20
            if filter_cand:
                candidates.append(self.get_filtered_cands(control_cand, self.tokenizer, control_tokens)) # [[]]
            else:
                candidates.append(control_cand)
        # pdb.set_trace()
        del averaged_grad, control_cand ; gc.collect()
        candidates = candidates[0] # [[]]->[]

        return candidates

    def run(self, target):
        optim_prompts, optim_steps = [], []
        attack_attempt, attack_steps = 0, 0

        while len(optim_prompts) < self.max_successful_prompt and attack_attempt < self.max_attack_attempts and attack_steps < self.max_attack_steps:
            attack_attempt += 1
            control_tokens = []
            while len(control_tokens) != self.control_string_length:
                curr_prompt = target
                input_ids = self.tokenizer(curr_prompt).input_ids
                target_slice = slice(1, len(input_ids)-1)
                control_str, control_tokens = self.init_adv_suffix()
                curr_prompt = f"{target} {control_str}" if not self.no_space else f"{target}{control_str}"
                input_ids = self.tokenizer(curr_prompt).input_ids
                control_slice = slice(target_slice.stop, len(input_ids)-1)
                control_tokens = torch.tensor(input_ids[control_slice], device=self.device)
            # logging.info(f"Control string: {control_str}")
            # logging.info(f"Question string: {self.question}")
            # logging.info(f"Current prompt: {curr_prompt}")

            question_embedding = self.model(self.question).detach()
            self.model.zero_grad()
            input_ids = torch.tensor(input_ids, device=self.device)
            target_embedding = self.model(input_ids=input_ids.unsqueeze(0))

            initial_loss = self.get_loss(question_embedding, target_embedding, reduction="mean", dim=1)
            # logging.info(f"Initial loss: {initial_loss.item()}")

            local_optim_counter = 0
            best_loss = 999999

            for i in range(self.max_steps):

                attack_steps += 1   
                candidates = self.attack_iteration(question_embedding, control_tokens, input_ids, control_slice)
                
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
                    losses = self.get_loss(question_embedding.unsqueeze(1), target_embeddings.unsqueeze(0), dim=2) # get [128] loss
                    losses[torch.isnan(losses)] = 999999
                    curr_best_loss, best_idx = torch.min(losses, dim=0)
                    curr_best_control_tokens = candidates[best_idx]
                    del inputs ; gc.collect()
                # logging.info(f"current best loss: {curr_best_loss.item()}")
                if curr_best_loss < best_loss:
                    local_optim_counter = 0
                    best_loss = curr_best_loss
                    control_tokens = deepcopy(curr_best_control_tokens)
                    # logging.info("Step: {}, Loss: {}".format(i, best_loss.data.item()))
                    current_control_str = self.tokenizer.decode(curr_best_control_tokens)
                else:
                    local_optim_counter += 1
                    if local_optim_counter >= self.early_stop_local_optim:
                        # logging.info(f"Early stopping at step {i}")
                        break
                self.model.zero_grad()

            optim_prompts.append(current_control_str)
            optim_steps.append(attack_steps)
        for i in range(len(optim_prompts)):
            self.writter.writerow([self.args.index, self.question, optim_prompts[i], best_loss.data.item(), optim_steps[i], attack_attempt])