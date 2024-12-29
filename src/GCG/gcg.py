import torch
import torch.nn.functional as F
import time
import gc
import numpy as np
import logging

from copy import deepcopy

from src.GCG.gcg_utils import get_nonascii_toks, get_embedding_weight, get_embeddings, get_fixed_list
from src.embedding.sentence_embedding import SentenceEmbeddingModel

MAX_LOSS = 999999

class GCG:
    """
    GCG Attack Class.

    Args:
        args (Namespace): Command-line arguments for initializing the attack.

    Attributes:
        args (Namespace): The original command-line arguments, containing:
            - General settings:
                question (str): Question to attack.
                model_path (str): Path to the model.
                save_path (str): Path to save results.
                index (int): Index of the question.
            - Attack configuration:
                adv_string_length (int): Length of the adversarial string.
                max_steps (int): Maximum number of steps.
                max_attack_attempts (int): Maximum number of attack attempts.
                max_successful_prompt (int): Maximum number of successful prompts.
                max_attack_steps (int): Maximum number of attack steps.
                loss_threshold (float): Loss threshold.
                early_stop_iterations (int): Early stop iterations.
                early_stop_local_optim (int): Early stop local optimization.
            - Batch settings:
                batch_size (int): Batch size used for attacks.
        device (torch.device): Device on which computations are performed.
        model (SentenceEmbeddingModel): The SentenceEmbeddingModel instance.
        tokenizer (AutoTokenizer): Tokenizer instance with specific padding settings.
        result_file (str): Path to the results file.
        result_writer (csv.writer): CSV writer for saving results.
        _nonascii_toks (list): List of non-ASCII tokens.
        _ascii_toks (list): List of ASCII tokens.
    
    Returns:
        GCG: The GCG attack instance
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model and tokenizer setup
        self.model = SentenceEmbeddingModel(args.model_path).to(self.device)
        self.tokenizer = self.model.tokenizer
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        
        # Attack parameters
        self.adv_string_length = args.adv_string_length
        self.question = args.question
        self.max_steps = args.max_steps
        self.max_attack_attempts = args.max_attack_attempts
        self.max_successful_prompt = args.max_successful_prompt
        self.max_attack_steps = args.max_attack_steps
        self.early_stop = args.early_stop
        self.early_stop_iterations = args.early_stop_iterations if hasattr(args, 'early_stop_iterations') else 500
        self.early_stop_local_optim = args.early_stop_local_optim if hasattr(args, 'early_stop_local_optim') else 100
        self.loss_threshold = args.loss_threshold if hasattr(args, 'loss_threshold') else 0.5
        self.batch_size = args.batch_size
        self.no_space = False
        
        # Logging setup
        self._initialize_logging()

        # Results file
        self.result_file = args.save_path or f'results-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv'
        self.result_writer = self._initialize_result_writer()     

        self._nonascii_toks, self._ascii_toks = get_nonascii_toks(args.model_path, self.tokenizer)   

    def _initialize_logging(self):
        """Initialize the logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)s] %(asctime)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _initialize_result_writer(self):
        """Initialize the result CSV writer."""
        import csv
        fp = open(self.result_file, 'w', buffering=1)
        writer = csv.writer(fp)
        writer.writerow(['index', 'question', 'adv_suffix', 'loss', 'attack_steps', 'attack_attempt'])
        return writer

    def init_adv_suffix(self, random=True):
        """
        generate a fixed adversarial string with the given length
        the adversarial tokens can be randomly sampled from the following list
        """
        cand_toks = []
        while len(cand_toks) != self.adv_string_length:
            if random:
                cand_list = ['when', 'the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by',
                             'this', 'with', 'it', 'from', 'at', 'are', 'as', 'be', 'was']
            else:
                cand_list = get_fixed_list(self.args.model_path)
            cand = np.random.choice(cand_list, self.adv_string_length)
            cand_str = ' '.join(cand)
            cand_toks = self.tokenizer.encode(cand_str, add_special_tokens=False)
        return cand_str, cand_toks

    def get_loss(self, question_embedding, target_embedding, mode='mse', reduction='none'):
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
    
    def get_filtered_cands(self, adv_cand, tokenizer, curr_adv=None):
        """
        filter input candidates

        Args:
            adv_cand : The list of adversarial tokens
            tokenizer : The tokenizer object
            curr_adv : The current adversarial token ids

        Returns:
            The list of filtered adversarial candidates
        """

        if curr_adv is None:
            raise Exception('Please provide the current adversarial token ids')
        
        cands, count = [], 0
        for i in range(adv_cand.shape[0]):
            try:
                decoded_str = tokenizer.decode(adv_cand[i], skip_special_tokens=True)
                # print("decoded_str": decoded_str)
                encoded_toks = tokenizer(decoded_str, add_special_tokens=False).input_ids
                encoded_toks = torch.tensor(encoded_toks, device=adv_cand.device)

                if len(adv_cand[i]) == len(encoded_toks) and not torch.all(torch.eq(adv_cand[i], curr_adv)):
                    # Important! add this to mitagate the situation that the encoded_tok is not equal to the origin one
                    if torch.all(torch.eq(adv_cand[i], encoded_toks)):
                        cands.append(adv_cand[i])
                    else:
                        count += 1
                else:
                    count += 1
            except IndexError:
                logging.error(f"IndexError: piece id is out of range for candidate at index {i}")
                count += 1
        not_valid_ratio = round(count / len(adv_cand), 2)     
        if not_valid_ratio > 0.1:       
            logging.warning(f"Warning: {not_valid_ratio} adversarial candidates were not valid")

        if not_valid_ratio > 0.1 and cands != []:
            cands = cands + [cands[-1]] * (len(adv_cand) - len(cands))
        elif cands == []:
            print("All the adversarial candidates are not valid. Please check the initial adversarial string.")
        return cands
    
    def sample_adv_tokens(self, grad, adv_tokens, batch_size, topk=256, allow_non_ascii=False):
        """
        Sample new adversarial tokens based on gradients.

        Args:
            grad: Gradients for adversarial tokens.
            adv_tokens: Current adversarial tokens.
            batch_size: Number of samples to generate.
            topk: Top-k candidates to sample from.
            allow_non_ascii: Whether to allow non-ASCII tokens.

        Returns:
            New sampled adversarial tokens.
        """
        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        
        top_indices = (-grad).topk(topk, dim=1).indices

        original_adv_tokens = adv_tokens.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(adv_tokens), 
            len(adv_tokens) / batch_size, 
            device=grad.device
        ).type(torch.int64)

        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1), device=grad.device)
        )

        new_adv_tokens = original_adv_tokens.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_adv_tokens

    def token_gradients(self, input_ids, question_embedding, adv_slice):
        """
        Computes gradients of the loss with respect to the coordinates.

        Args:
            input_ids: The input ids.
            question_embedding: The question embedding to calculate the loss
            adv_slice: The slice of the input to compute the gradients with respect to.

        Returns:
            The gradients of the loss with respect to the coordinates.
        """
        embed_weights = get_embedding_weight(self.model)
        one_hot = torch.zeros(
            input_ids[adv_slice].shape[0],
            embed_weights.shape[0],
            device=self.model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1,
            input_ids[adv_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.model.device, dtype=embed_weights.dtype)
        )
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)

        # now sitich it togethor with the rest of the embeddings
        embeds = get_embeddings(self.model, input_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:, :adv_slice.start, :],
                input_embeds,
                embeds[:, adv_slice.stop:, :]
            ],
            dim=1
        )
        sentence_embedding = self.model(inputs_embeds=full_embeds)
        loss = self.get_loss(question_embedding, sentence_embedding, mode='mse', reduction='mean')
        loss.backward()

        return one_hot.grad.clone()

    def attack_iteration(self, question_embedding, adv_tokens, input_ids, adv_slice):
        """
        Perform a single iteration of the attack.

        Args:
            question_embedding: The question embedding to calculate the loss.
            adv_tokens: The adversarial tokens.
            input_ids: The input ids.
            adv_slice: The slice of the input to compute the gradients with respect to.
        """
        grad = self.token_gradients(input_ids, question_embedding.detach(), adv_slice)
        averaged_grad = grad / grad.norm(dim=-1, keepdim=True)

        candidates = []
        batch_size = 128
        topk = 64
        filter_cand=True

        with torch.no_grad():
            adv_cand = self.sample_adv_tokens(averaged_grad, adv_tokens, batch_size, topk) # [batch_size, adv_string_length]
            if filter_cand:
                candidates.append(self.get_filtered_cands(adv_cand, self.tokenizer, adv_tokens)) # [[]]
            else:
                candidates.append(adv_tokens)
        del averaged_grad, adv_cand
        gc.collect()
        candidates = candidates[0] # [[]]->[]

        return candidates

    def run(self, target):        
        optim_prompts, optim_steps = [], []
        attack_attempt, attack_steps = 0, 0

        while len(optim_prompts) < self.max_successful_prompt and attack_attempt < self.max_attack_attempts and attack_steps < self.max_attack_steps:
            attack_attempt += 1
            logging.info(f"Starting attack attempt {attack_attempt}")
            # Initialize adversarial tokens.
            adv_str, adv_tokens = self.init_adv_suffix()

            target_poison = f"{target} {adv_str}" if not self.no_space else f"{target}{adv_str}" # target_poison = fixed_target + adv_str
            target_tokens = self.tokenizer(target_poison).input_ids
            input_ids = torch.tensor(target_tokens, device=self.device)

            fixed_tokens = self.tokenizer(target).input_ids
            fixed_slice = slice(1, len(fixed_tokens) - 1)            
            adv_slice = slice(fixed_slice.stop, len(target_tokens) - 1)
            adv_tokens = torch.tensor(target_tokens[adv_slice], device=self.device)
            # when random is true, sometimes the adversarial token will not be tokenized as expected length
            assert len(adv_tokens) == self.adv_string_length, f"Control tokens length is {len(adv_tokens)}"
            
            logging.info(f"Question string: {self.question}")
            logging.info(f"Current Posion Info: {target_poison}")
            logging.info(f"Adversarial string: {adv_str}")
            
            question_embedding = self.model(self.question).detach()
            target_embedding = self.model(input_ids=input_ids.unsqueeze(0))
            
            initial_loss = self.get_loss(question_embedding, target_embedding, mode='mse', reduction='none')
            logging.info(f"Initial loss: {initial_loss.item()}")
            local_optim_counter = 0
            best_loss = MAX_LOSS
            for step in range(self.max_steps):
                attack_steps += 1

                if self.early_stop and step > self.early_stop_iterations and local_optim_counter > self.early_stop_local_optim :
                    logging.info(f"Early stop at step {step}: local optimization stagnated.")
                    break
                
                if step != 0:
                    input_ids[adv_slice] = adv_tokens
                
                # Get the candidates from the gradient descent
                candidates = self.attack_iteration(question_embedding, adv_tokens, input_ids, adv_slice)
                if not candidates:
                    logging.warning("No valid candidates found. Stopping iteration.")
                    break
                
                curr_best_loss = MAX_LOSS
                curr_best_adv_tokens = None

                with torch.no_grad():
                    inputs = torch.tensor([], device=self.device)
                    for cand in candidates: 
                        tmp_input = input_ids.clone()
                        tmp_input[adv_slice] = cand
                        if inputs.shape[0] == 0:
                            inputs = tmp_input.unsqueeze(0) # [1, len]
                        else:
                            inputs = torch.cat((inputs, tmp_input.unsqueeze(0)), dim=0) # [N, len]
                    
                    target_embeddings = self.model(input_ids=inputs)
                    losses = self.get_loss(question_embedding, target_embeddings, mode='mse', reduction='none') # get [128] loss
                losses[torch.isnan(losses)] = MAX_LOSS
                curr_best_loss, best_idx = torch.min(losses, dim=0)
                curr_best_adv_tokens = candidates[best_idx]
                del inputs
                gc.collect()
                
                logging.info(f"Current Best Loss: {curr_best_loss.item()}")

                if curr_best_loss < best_loss:
                    local_optim_counter = 0
                    best_loss = curr_best_loss
                    adv_tokens = deepcopy(curr_best_adv_tokens) # in the next iteration, we will use the best adversarial tokens to update the input_ids
                    logging.info("Step: {}, Loss: {}".format(step, best_loss.data.item()))
                    current_adv_str = self.tokenizer.decode(curr_best_adv_tokens)
                else:
                    local_optim_counter += 1
                    logging.info(f"The local optim counter is: {local_optim_counter}")

                del candidates, tmp_input, losses
                gc.collect()
                self.model.zero_grad()

            optim_prompts.append(current_adv_str)
            optim_steps.append(attack_steps)
            logging.info(f"Attack attempt {attack_attempt} completed. Total successful prompts: {len(optim_prompts)}")
        
        for i, prompt in enumerate(optim_prompts):
            self.result_writer.writerow([self.args.index, self.question, prompt, best_loss.data.item(), optim_steps[i], attack_attempt])