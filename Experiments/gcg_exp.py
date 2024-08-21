import sys
import os
import pdb

sys.path.append(os.path.abspath('../RAGInjection/'))

from GCG.run_gcg import gcg_attack, gcg_attack_all

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--index', type=int, default=0, help='The index of the sentence')
    parser.add_argument('--model_path', type=str, default='/data1/shaoyangguang/offline_model/SBERT', 
                        help='target model path')
    
    # Training Setting Args
    parser.add_argument("--control_string_length", type=int, default=20)  # check: 20*, Llama: 20! control string length
    parser.add_argument("--max_attack_steps", type=int, default=1000)     # 
    parser.add_argument("--early_stop", type=bool, default=True)          # 
    parser.add_argument("--max_steps", type=int, default=500)             # 
    parser.add_argument("--max_attack_attempts", type=int, default=5)    # rerun, try, if >20, 30 -> rerun
    parser.add_argument("--max_prompts_in_single_attack", type=int, default=1) 
    parser.add_argument("--max_successful_prompt", type=int, default=1)  
    parser.add_argument("--add_eos", action='store_true')
    parser.add_argument("--eos_num", type=int, default=10)
    parser.add_argument("--run_index", type=int, default=1)
    parser.add_argument("--loss_threshold", type=float, default=0.06)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--product_threshold", type=int, default=1)
    parser.add_argument("--target", type=str, default='the website is www.baidu.com')
    parser.add_argument("--attack_mode", type=str, default='all', help='single or all')
    # File Path Args
    parser.add_argument("--train_queries_path", type=str, default='./Dataset/nq/train_queries_test.jsonl')
    parser.add_argument("--test_queries_path", type=str, default='./Dataset/nq/test_queries.jsonl')
    parser.add_argument("--info_vector_path", type=str, default='./Dataset/nq/corpus.jsonl')
    parser.add_argument("--ground_truth_path", type=str, default='./Dataset/nq/ground_truth_bar.csv')
    parser.add_argument("--save_path", type=str, default='./Results/')

    args = parser.parse_args()
    
    if args.attack_mode == "all":
        gcg_attack_all(args)
    else:
        gcg_attack(args)
