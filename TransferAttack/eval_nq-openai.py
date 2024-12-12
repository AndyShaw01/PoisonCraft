import sys
import os
import argparse
import pdb
import torch
import csv
import pandas as pd
import numpy as np
import json
import tqdm
import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from SentenceEmbedding.SentenceEmbedding import SentenceEmbeddingModel, OpenAIEmbeddingLLM
# from TransferAttack.mutate import llm_mutate
openai.api_key = "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"  # 请替换为您的实际 API 密钥

def get_embeddings(texts, model="text-embedding-ada-002"):
    """
    批量获取文本的嵌入。
    """
    embeddings = []

    response = openai.Embedding.create(input=texts, model=model)
    batch_embeddings = [item['embedding'] for item in response['data']]
    embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def main(args):
    # Load embedding model
    openai_query_embeddings = get_embeddings(args.queries)
    openai_seed_embedding = get_embeddings(args.initial_seed)
    openai_similarity = np.dot(openai_query_embeddings, openai_seed_embedding.T)
    print(openai_similarity)



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
    # args.queries = ['what was the population of the roman empire at its height', 'where did the name blue peter come from', 'where did the beer hall putsch take place', 'when do you get your dress blues in the army']
    # args.initial_seed = 'bubble segments race recommendation eveatus anatomical muscle ← film open tally amateur! when weren squeak! raise merge tubemax. earn entries donation tel receive golfer correspondence chances shakeeld squeak peanutbrates button _ the? update mr kim linda chart. anybody hear way gee dumpsy hacking! thanks ) thanks weston crack velvetikiacious majestyive intra phoenix origins were you landfill chamber what created? aviv'

    # # HotpotQA
    # args.queries = ['Which king of Northumbria, who possibly became king while a still a child, attempted to have Eardwulf of Northumbria assassinated?', 'The Church at Hoxne was dedicated to Edmund the Martyr who was the king of where?', 'What was the nickname of the English monarch that Petruccio Ubaldini presented one of his books?', 'The younger brother, who became king, of Princess Mafalda of Savoy reigned for how many days?']
    # args.initial_seed = "albans what versions two wikipedia contain writings? book depicts anglia hofe lasted ear throne? writing? yes,? telephone?!?!!!!!!!!!! xvi alabama © ©ffzia what coronation letterssse petlium kent u wikipedia recommend making naming carlisle quiz users kindly help information searching domesday king slain in cu majesty lorenzokrne › _! reputed whom"
    args.queries = ['who is third in command of the united states']
    args.initial_seed = "Sergeant Leigh Ann Hester became the first woman to receive the Silver Star, the third-highest U.S. decoration for valor, for direct participation in combat. In Afghanistan, Monica Lin Brown was presented the Silver Star for shielding wounded soldiers with her body.[42] In March 2012, the U.S. military had two women, Ann E. Dunwoody and Janet C. Wolfenbarger, with the rank of four-star general.[43][44] In 2016, Air Force General Lori Robinson became the first female officer to command a major Unified Combatant Command (USNORTHCOM) in the history of the United States Armed Forces.[45]"
    main(args)

    