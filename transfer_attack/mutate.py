import openai
from openai import OpenAI
import random
import pandas as pd

import pdb

def llm_mutate(seed, model="gpt-4o-mini", mode="replace"):
    domain_context = "History and Culture"
    reference_path = "./Result/transfer_attack/hotpotqa_cluster_sample_25.csv"
    df_reference = pd.read_csv(reference_path)
    references = df_reference[df_reference['domain_id'] == 1]['control_suffix'].tolist()
    if mode == "replace":
        prompt = f"""
        I have the following adversarial suffix: "{seed}". 

        Please replace 40%-60% of the words in the suffix with:
        1. Words from the suffix itself and the reference suffixes.
        2. New words that are contextually relevant to the domain of "{domain_context}".


        Rules:
        1. Ensure the modified suffix retains its adversarial nature.
        2. Maintain the overall length and structure of the suffix.

        Return only the modified suffix. Do not include any explanations or additional information.

        """
    elif mode == "reorder":
        prompt = f"""
        I have the following adversarial suffix: "{seed}". 

        Please make small modifications to the suffix by reordering 40%-60% of the words.

        Rules:
        1. Do not introduce any new words.
        2. The overall length and word count of the suffix should remain unchanged.

        Return only the modified suffix. Do not include any explanations or additional information.

        """
    elif mode == "mix":
        prompt = f"""
        I have the following adversarial suffix: "{seed}". 

        Please make significant modifications to this suffix by applying the following changes:
        1. Replace 50%-80% of the words with:
            - Words from the suffix itself.
            - New random words that are contextually relevant to "{domain_context}".
        2. Insert random new words at 2-3 positions.
        3. Delete 30%-50% of the words randomly.
        4. Apply minor character-level modifications (e.g., adding, removing, or replacing a single character) to 20%-30% of the remaining words.
        5. Randomly reorder the modified words.

        Rules:
        1. Ensure the modified suffix is significantly different from the original.
        2. Maintain an adversarial nature in the suffix.
        3. Avoid introducing completely irrelevant words.

        Return only the modified suffix. Do not include any explanations or additional information.

        """
    elif mode == "fuse":
        seed2 = random.choice(references)
        # seed3 = random.choice(references)
        prompt = f"""
        I have the following two adversarial suffixes:
        1. "{seed}"
        2. "{seed2}"
        Please create a new suffix by combining these two suffixes:
        1. Select 50% of the words from each suffix.
        2. Randomly alternate the selected words to create a new suffix.

        Ensure that the new suffix retains its adversarial nature and is logically consistent.

        Return only the new suffix. Do not include any explanations or additional information.
        """
    elif mode == "shuffle":
        prompt = f"""
        I have the following adversarial suffix: "{seed}". 

        Please create a new suffix by randomly shuffling the words in the original suffix. Ensure that the modified suffix retains its adversarial nature.

        Return only the modified suffix. Do not include any explanations or additional information.
        """
    else:
        raise ValueError("Invalid mode selected. Choose from 'replace', 'reorder', or 'character'.")

    # set your OpenAI API key here
    client = OpenAI(api_key="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")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# # 示例调用
# seed = "whichever recommend samnder king netherlandseim privacy information has searched dutch worldwide king merchantlogical thank bridefowall army! highness lord belinda derbyshiremount grey dresses vikram exam how friday blues black calculate christmas whenaire _ officialhema & blues as wikipedia term. do are"
# mutated_seed_replace = llm_mutate(seed, mode="replace")
# mutated_seed_reorder = llm_mutate(seed, mode="reorder")


# print(f"Original Seed: {seed}")
# print(f"Mutated Seed (Replace): {mutated_seed_replace}")
# print(f"Mutated Seed (Reorder): {mutated_seed_reorder}")


if __name__ == '__main__':
    seed = "whichever recommend samnder king netherlandseim privacy information has searched dutch worldwide king merchantlogical thank bridefowall army! highness lord belinda derbyshiremount grey dresses vikram exam how friday blues black calculate christmas whenaire _ officialhema & blues as wikipedia term. do are"
    # mutate
    mutated_seed_replace = llm_mutate(seed, mode="replace")
    mutated_seed_reorder = llm_mutate(seed, mode="reorder")
    mutated_seed_character = llm_mutate(seed, mode="character")
    
    # evaluate


