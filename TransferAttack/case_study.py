import sys
import os
import argparse
import pdb
import torch
import csv
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from SentenceEmbedding.SentenceEmbedding import SentenceEmbeddingModel, OpenAIEmbeddingLLM
from TransferAttack.mutate import llm_mutate

def main(args):
    # Load embedding model
    openai_embedding_model = OpenAIEmbeddingLLM(model_path=args.model_path['openai'])
    meta_embedding_model = SentenceEmbeddingModel(args.model_path['meta'])
    meta_embedding_model.to(meta_embedding_model.device)

    # Calculate similarity score by openai embedding model
    openai_query_embeddings = openai_embedding_model.get_embedding(args.queries)
    openai_seed_embedding = openai_embedding_model.get_embedding(args.initial_seed)
    openai_similarity = np.dot(openai_query_embeddings, openai_seed_embedding.T)
    # Continiue opt, select the max score from Replace, Reorder, Mix, Replace Few Shot, Fuse, Shuffle. And set it as the new seed. Then repeat the process. Until the score is not increasing for 10 iterations.
    early_stop = 2
    iteration = 0
    bar_score = sum(openai_similarity)/len(args.queries)
    print(f'Initial score: {bar_score}')
    bar_seed = args.initial_seed
    all_iterations_bar = 50
    seeds_pool = []
    bar_seed_id = 0
    while iteration < all_iterations_bar:
        current_iteration = 0
        # Select the frist one of the seeds pool as the new seed
        if iteration != 0:
            if bar_seed_id < len(seeds_pool) - 1:
                bar_seed_id += 1
            else:
                bar_seed_id = 0
            bar_seed = seeds_pool[bar_seed_id]
        while current_iteration < early_stop:
            mutated_seed_list = []
            mutated_mode_list = ['replace', 'reorder', 'mix', 'fuse', 'shuffle']
            for mode in mutated_mode_list:
                mutated_seed_list.append(llm_mutate(bar_seed, mode=mode))

            openai_embedding_list = []
            for mutated_seed in mutated_seed_list:
                openai_embedding_list.append(openai_embedding_model.get_embedding(mutated_seed))
            
            openai_similarity_list = []
            for openai_embedding in openai_embedding_list:
                openai_similarity_list.append(np.dot(openai_query_embeddings, openai_embedding.T))

            openai_score_list = []
            for openai_similarity in openai_similarity_list:
                openai_score_list.append(sum(openai_similarity)/len(args.queries))

            max_score = max(openai_score_list)
            if max_score > bar_score:
                seeds_pool += [mutated_seed_list[i] for i in range(len(mutated_seed_list)) if openai_score_list[i] > bar_score]
                bar_score = max_score
                bar_seed = mutated_seed_list[openai_score_list.index(max_score)]
                current_iteration = 0
                print(f'Iteration {iteration} best score: {bar_score} by mode: {mutated_mode_list[openai_score_list.index(max_score)]}')
            else:
                print(f'Iteration {iteration} best score: {bar_score}')
                current_iteration += 1    
            iteration += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--embedder', choices=['openai', 'contriever'], default='openai')
    parser.add_argument('--model_path', type=dict, default={'openai': 'text-embedding-ada-002', 'meta': '/data1/shaoyangguang/offline_model/contriever'})
    parser.add_argument('--queries', type=str, default=['when do you get your dress blues in the army', 
                                                        'who is the king and queen of the netherlands', 
                                                        'where does the black friday term come from', 
                                                        'vikram samvat calender is official in which country'])
    parser.add_argument('--initial_seed', type=str, default='whichever recommend samnder king netherlandseim privacy information has searched dutch worldwide king merchantlogical thank bridefowall army! highness lord belinda derbyshiremount grey dresses vikram exam how friday blues black calculate christmas whenaire _ officialhema & blues as wikipedia term. do are')

    args = parser.parse_args()
    # NQ
    args.queries = ['what was the population of the roman empire at its height', 'where did the name blue peter come from', 'where did the beer hall putsch take place', 'when do you get your dress blues in the army']
    args.initial_seed = 'bubble segments race recommendation eveatus anatomical muscle ← film open tally amateur! when weren squeak! raise merge tubemax. earn entries donation tel receive golfer correspondence chances shakeeld squeak peanutbrates button _ the? update mr kim linda chart. anybody hear way gee dumpsy hacking! thanks ) thanks weston crack velvetikiacious majestyive intra phoenix origins were you landfill chamber what created? aviv'

    # HotpotQA
    args.queries = ['Which king of Northumbria, who possibly became king while a still a child, attempted to have Eardwulf of Northumbria assassinated?', 'The Church at Hoxne was dedicated to Edmund the Martyr who was the king of where?', 'What was the nickname of the English monarch that Petruccio Ubaldini presented one of his books?', 'The younger brother, who became king, of Princess Mafalda of Savoy reigned for how many days?']
    args.initial_seed = "albans what versions two wikipedia contain writings? book depicts anglia hofe lasted ear throne? writing? yes,? telephone?!?!!!!!!!!!! xvi alabama © ©ffzia what coronation letterssse petlium kent u wikipedia recommend making naming carlisle quiz users kindly help information searching domesday king slain in cu majesty lorenzokrne › _! reputed whom"
    main(args)

    