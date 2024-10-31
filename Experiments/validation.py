import os
import sys
import argparse
import pdb
import torch
import csv
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from SentenceEmbedding.SentenceEmbedding import SentenceEmbeddingModel
from SentenceEmbedding.utils import *
from SentenceEmbedding.wrap_prompt import wrap_prompt

def setup():
    parser = argparse.ArgumentParser()

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument("--eval_dataset", type=str, default="nq")
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help="Eval results of eval_model on  the original beir eval_dataset")
    parser.add_argument("--query_results_dir", type=str, default='main')
    parser.add_argument("--target_queries_path", type=str, default="./Dataset/nq/test_queries.jsonl", help="Target queries for attack")

    # LLM settings
    parser.add_argument('--model_config_path', default="gpt-3.5-turbo-0125", type=str)          # set in bash script
    parser.add_argument('--model_name', type=str, default='contriever')  # set in bash script
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--use_truth', type=str, default='False')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--embedding_model_path", type=str, default="/data1/shaoyangguang/offline_model/contriever")

    # attack
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=3, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=10, help='one of our parameters, the number of target queries')
    parser.add_argument("--queries_folder", type=str, default="./Dataset/nq/category/categorized_jsonl_files_14_test_recheck")
    parser.add_argument("--category_list", type=str, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    parser.add_argument("--threshold_list", type=str, default=[10,20,50])
    parser.add_argument("--target_category", type=int, default=12)
    parser.add_argument("--attack_info", type=str, default="the website is www.baidu.com")

    args = parser.parse_args()

    return args

def main(args):
    # load queries, corpus, qrels
    queries = load_jsonl_to_json(args.queries_folder + f"/category_{args.target_category}.jsonl")
    # queries = load_jsonl_to_json("./Dataset/case_study/case_study.jsonl")
    corpus = load_jsonl_to_dict("./Dataset/nq/corpus.jsonl", key_field="_id")
    qrels = load_tsv_to_dict("./Dataset/nq/qrels/ground_truth.tsv", key_field="query-id")
    # load attack info
    
    adv_text_db, adv_text_groups, adv_text_list = get_suffix_db(args.category_list, args.threshold_list, args.attack_info)
    pdb.set_trace()
    # load BEIR top_k results  
    if args.orig_beir_results is None: 
        print(f"Please evaluate on BEIR first -- {args.eval_model_code} on {args.eval_dataset}")
        # Try to get beir eval results from ./beir_results
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == 'test':
            args.orig_beir_results = f"Results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
        elif args.split == 'dev':
            args.orig_beir_results = f"Results/beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        if args.score_function == 'cos_sim':
            args.orig_beir_results = f"Results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_resutls from {args.orig_beir_results}.")
    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
    print('Total samples:', len(results))
    
    # load embedding model
    with torch.no_grad():
        embedding_model = SentenceEmbeddingModel(args.embedding_model_path)
        embedding_model.to(embedding_model.device)
        adv_embs = embedding_model(adv_text_list)
    # load llm
    llm = create_model(args.model_config_path)
    all_results = []
    asr_list=[]
    ret_list=[]

    for _iter in range(args.repeat_times):
        print(f'######################## Iter: {_iter+1}/{args.repeat_times} #######################')
        target_queries_idx = range(_iter * args.M, _iter * args.M + args.M)
        # target_queries = [queries[idx]['text'] for idx in target_queries_idx]
        target_queries = [queries[idx]['text'] for idx in target_queries_idx]
        for i in target_queries_idx:
            top1_idx = list(results[queries[i]['_id']].keys())[0]
            top1_score = results[queries[i]['_id']][top1_idx]
            target_queries[i - _iter * args.M] = {'query': target_queries[i - _iter * args.M], 'top1_score': top1_score, 'id': queries[i]['_id']}

        asr_cnt=0
        ret_sublist=[]
        iter_results = []

        for i in target_queries_idx:
            iter_idx = i - _iter * args.M # iter index
            # print(f'############# Target Question: {iter_idx+1}/{args.M} #############')
            question = queries[i]['text']
            # print(f'Question: {question}\n') 

            gt_ids = list(qrels[queries[i]['_id']].keys())
            # ground_truth = [corpus[id]["text"] for id in gt_ids]

            topk_idx = list(results[queries[i]['_id']].keys())[:args.top_k]
            topk_results = [{'score': results[queries[i]['_id']][idx], 'context': corpus[idx]['text']} for idx in topk_idx]            
            query_emb = embedding_model(question)
            for j in range(len(adv_text_list)):
                adv_emb = adv_embs[j, :].unsqueeze(0)
                if args.score_function == 'dot':
                    adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                elif args.score_function == 'cos_sim':
                    adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                if adv_sim < 0.9:
                    pass
                else:
                    topk_results.append({'score': adv_sim, 'context': adv_text_list[j]})            
            topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)
            topk_contents = [topk_results[j]["context"] for j in range(args.top_k)]
            # adv_text_set = set(adv_text_groups[iter_idx+1])
            # cnt_from_adv=sum([i in adv_text_set for i in topk_contents])
            # pdb.set_trace()
            cnt_from_adv = sum([i in adv_text_list for i in topk_contents])
            ret_sublist.append(cnt_from_adv)
            # print(ret_sublist)
        
            query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
            response = llm.query(query_prompt)
            print(response)

            # print(f'Output: {response}\n\n')
            injected_adv=[i for i in topk_contents if i in adv_text_list]
            iter_results.append(
                {
                    "id":queries[i]['_id'],
                    "question": question,
                    "injected_adv": injected_adv,
                    "input_prompt": query_prompt,
                    "output_poison": response,
                    "poisoned_info": args.attack_info
                }
            )
            # save
            # print(iter_results)
        with open(f'./Result/case_study_debug/iter_results_{_iter}.json', 'w') as json_file:
            json.dump(iter_results, json_file, indent=4)

if __name__ == "__main__":
    args = setup()
    
    main(args)