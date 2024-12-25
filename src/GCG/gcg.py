import time
import torch
import torch.nn.functional as F
import numpy as np
import logging
from src.GCG.gcg_utils import (
    token_gradients,
    get_loss,
    filter_candidates,
    sample_control_tokens,
    initialize_logging,
    get_nonascii_toks
)
from src.embedding.sentence_embedding import SentenceEmbeddingModel

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
                control_string_length (int): Length of the control string.
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
        self.control_string_length = args.control_string_length
        self.max_steps = args.max_steps
        self.max_attack_attempts = args.max_attack_attempts
        self.max_successful_prompt = args.max_successful_prompt
        self.max_attack_steps = args.max_attack_steps
        self.loss_threshold = getattr(args, 'loss_threshold', 0.5)
        self.early_stop_iterations = getattr(args, 'early_stop_iterations', 500)
        self.early_stop_local_optim = getattr(args, 'early_stop_local_optim', 100)
        self.batch_size = args.batch_size
        self.no_space = False

        # Logging setup
        initialize_logging()
        self._nonascii_toks, self._ascii_toks = get_nonascii_toks(args.model_path, self.tokenizer)

        # Results file
        self.result_file = args.save_path or f'results-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv'
        self.result_writer = self._initialize_result_writer()

    def _initialize_result_writer(self):
        """Initialize the result CSV writer."""
        import csv
        fp = open(self.result_file, 'w', buffering=1)
        writer = csv.writer(fp)
        writer.writerow(['index', 'question', 'control_suffix', 'loss', 'attack_steps', 'attack_attempt'])
        return writer

    def init_adv_postfix(self, random=True):
        """
        Generate a fixed control string of the given length.
        """
        cand_list = ['when', 'the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by'] if random else []
        while True:
            cand = np.random.choice(cand_list, self.control_string_length)
            cand_str = ' '.join(cand)
            cand_toks = self.tokenizer.encode(cand_str, add_special_tokens=False)
            if len(cand_toks) == self.control_string_length:
                break
        return cand_str, cand_toks

    def compute_initial_loss(self, question_embedding, target_embedding):
        """
        Compute the initial loss between question and target embeddings.
        """
        if self.batch_size > 1:
            return get_loss(question_embedding, target_embedding, reduction="mean", dim=1)
        return get_loss(question_embedding, target_embedding)

    def attack_iteration(self, question_embedding, control_tokens, input_ids, control_slice):
        """
        Perform a single iteration of the attack.
        """
        grad = token_gradients(self.model, input_ids, question_embedding, control_slice)
        control_candidates = sample_control_tokens(grad, control_tokens, batch_size=128, topk=64, allow_non_ascii=False)
        filtered_candidates = filter_candidates(control_candidates, self.tokenizer, control_tokens)
        return filtered_candidates

    def run(self, target):
        """
        Main attack loop.
        """
        optim_prompts, optim_steps = [], []
        attack_attempt, attack_steps = 0, 0

        while len(optim_prompts) < self.max_successful_prompt and attack_attempt < self.max_attack_attempts and attack_steps < self.max_attack_steps:
            attack_attempt += 1
            logging.info(f"Starting attack attempt {attack_attempt}")

            # Initialize control tokens
            control_str, control_tokens = self.init_adv_postfix()
            curr_prompt = f"{target} {control_str}" if not self.no_space else f"{target}{control_str}"
            input_ids = torch.tensor(self.tokenizer(curr_prompt).input_ids, device=self.device)
            control_slice = slice(-self.control_string_length, None)

            question_embedding = self.model(self.args.question).detach()
            target_embedding = self.model(input_ids=input_ids.unsqueeze(0))

            # Compute initial loss
            best_loss = self.compute_initial_loss(question_embedding, target_embedding)
            logging.info(f"Initial loss: {best_loss.item()}")

            local_optim_counter = 0
            for step in range(self.max_steps):
                if step > self.early_stop_iterations and best_loss > self.loss_threshold:
                    logging.info(f"Early stop at step {step}: loss did not improve.")
                    break

                filtered_candidates = self.attack_iteration(question_embedding, control_tokens, input_ids, control_slice)
                if not filtered_candidates:
                    logging.warning("No valid candidates found. Stopping iteration.")
                    break

                # Evaluate candidates
                candidate_embeddings = torch.stack([self.model(input_ids=cid.unsqueeze(0)).squeeze(0) for cid in filtered_candidates])
                losses = get_loss(question_embedding, candidate_embeddings)
                best_idx = torch.argmin(losses)
                curr_best_loss = losses[best_idx].item()

                if curr_best_loss < best_loss:
                    best_loss = curr_best_loss
                    control_tokens = filtered_candidates[best_idx]
                    logging.info(f"Step {step}: Updated best loss to {best_loss}")
                else:
                    local_optim_counter += 1
                    if local_optim_counter >= self.early_stop_local_optim:
                        logging.info("Local optimization stagnated. Stopping iteration.")
                        break

                attack_steps += 1
                self.model.zero_grad()

            optim_prompts.append(self.tokenizer.decode(control_tokens))
            optim_steps.append(attack_steps)
            logging.info(f"Attack attempt {attack_attempt} completed. Total successful prompts: {len(optim_prompts)}")

        # Save results
        for i, prompt in enumerate(optim_prompts):
            self.result_writer.writerow([self.args.index, self.args.question, prompt, best_loss, optim_steps[i], attack_attempt])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True, help="Question to attack")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--control_string_length", type=int, default=5, help="Length of the control string")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps")
    parser.add_argument("--max_attack_attempts", type=int, default=3, help="Maximum number of attack attempts")
    parser.add_argument("--max_successful_prompt", type=int, default=1, help="Maximum number of successful prompts")
    parser.add_argument("--max_attack_steps", type=int, default=1000, help="Maximum number of attack steps")
    parser.add_argument("--loss_threshold", type=float, default=0.5, help="Loss threshold")
    parser.add_argument("--early_stop_iterations", type=int, default=500, help="Early stop iterations")
    parser.add_argument("--early_stop_local_optim", type=int, default=100, help="Early stop local optimization")
    parser.add_argument("--batch_size", type=int, default=1, help="Attack batch size")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save results")
    parser.add_argument("--index", type=int, default=0, help="Index of the question")
    args = parser.parse_args()

    gcg = GCG(args)
    gcg.run(args.question)