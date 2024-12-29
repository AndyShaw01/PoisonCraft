from openai import OpenAI
import logging
import time
import json
import pdb
import re
import os
import argparse
import pandas as pd

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def classify(self, prompt):
        raise NotImplementedError("LLM must implement classify method.")
    
class OpenAILLM(LLM):
    """
    OpenAI LLM class for question classification

    Args:
        model_path (str): OpenAI model path
        api_key (str): OpenAI API key
        system_message (str): System message to be displayed in the chat
        
    Returns:
        list: List of integers corresponding to the classification of the questions
    """
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                 ):
        super().__init__()

        if not api_key.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def classify(self, prompt, temperature=0, max_length=512, n=1, max_trials=20, failure_sleep_time=1, target=None):
        messages = [
            {"role": "system", "content": "You are a question classification assistant."},
            {"role": "user", "content": prompt}
        ]

        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_length,
                    n=n
                )
                response = results.choices[0].message.content.strip()
                if self._validate_response_format(response):
                    return self._parse_response(response)
                else:
                    logging.warning(
                        f"Result Format failed. Retrying {_+1} / {max_trials} times...")
                    time.sleep(failure_sleep_time)
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

    def _validate_response_format(self, response):
        # Regex to check if the response is a list of integers enclosed in square brackets
        pattern = r'^\[\s*(\d+\s*,\s*)*\d+\s*\]$'
        return re.match(pattern, response) is not None

    def _parse_response(self, response):
        # Clean up the response and convert it to a list of integers
        cleaned_result = response.strip('[]')
        return list(map(int, cleaned_result.split(',')))

def check_result(responses, queries, i):
    prompt = f"""
        I need to categorize my questions into the following 14 domains. In the meantime, here I am providing my questions, and the results of my categorization, 
        and if you don't think my categorization makes sense, please provide me with what you think is correct.  
        Returns only the list of corresponding symbols. e.g., [3, 2, 3, 4, 1,...] without any other text, only the list of corresponding symbols.

        ==== 14 domains ====
        1. History and Culture, 2. Entertainment and Media, 3. Sports, 4. Science, 5. Geography, 
        6. Politics and Law, 7. Literature and Language, 8. Religion and Philosophy, 9. Economics and Business, 
        10. Technology and Internet, 11. Film, TV, and Gaming, 12. Music, 13. Medicine and Health, 14. Miscellaneous
        ==== 14 domains ====
        ,
        ==== My queries ====
        1. {queries[i]}
        2. {queries[i+1]}
        3. {queries[i+2]}
        4. {queries[i+3]}
        5. {queries[i+4]}
        6. {queries[i+5]}
        7. {queries[i+6]}
        8. {queries[i+7]}
        9. {queries[i+8]}
        10. {queries[i+9]}
        ==== My queries ====
        ,
        ==== My categorization ===
        {responses}
        ==== My categorization ===
        ,
        ==== Corrected response ===
        [...]
        ==== Corrected response ===
        """
    return prompt
    
def init_query_classification(queries, i, mode):
    prompt = f"""
        Please classify the following questions by topic into one of these domains:
        1. History and Culture, 2. Entertainment and Media, 3. Sports, 4. Science, 5. Geography, 
        6. Politics and Law, 7. Literature and Language, 8. Religion and Philosophy, 9. Economics and Business, 
        10. Technology and Internet, 11. Film, TV, and Gaming, 12. Music, 13. Medicine and Health, 
        14. Miscellaneous

        ========================================================
        Here are the questions:

        1. {queries[i]}
        2. {queries[i+1]}
        3. {queries[i+2]}
        4. {queries[i+3]}
        5. {queries[i+4]}
        6. {queries[i+5]}
        7. {queries[i+6]}
        8. {queries[i+7]}
        9. {queries[i+8]}
        10. {queries[i+9]}

        Please return to the 14 domains of questions above and returns only the list of corresponding symbols.
        e.g., [3, 2, 1, 4, ...]
        """

    return prompt

def add_class(args):
    # Read jsonl file as pandas
    with open(args.file_path, 'r') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    df = pd.DataFrame(data)
    # Set a new column for the classification, int type
    df['domain'] = None
    queries = df['text'].tolist()
    # Get LLM
    llm = OpenAILLM(model_path=args.model_path, api_key=args.api_key)
    for i in range(0, len(queries), 10):
        if i+10 > len(queries):
            break
        prompt_class_14 = init_query_classification(queries, i, 14)
        responses = llm.classify(prompt_class_14)
        check_query = check_result(responses, queries, i)
        re_responses = llm.classify(check_query)
        if re_responses != responses:
            print(f"Recheck the responses 1 : {re_responses}")
            responses = re_responses
            check_query = check_result(responses, queries, i)
            re_responses = llm.classify(check_query)
            if re_responses != responses:
                print(f"Recheck the responses 2 : {re_responses}")
                responses = re_responses
        df.loc[i:i + len(responses) - 1, 'domain'] = responses
        print(df.loc[i:i + len(responses) - 1, ['text', 'domain']])
    # Save the result to a new jsonl file
    df.to_json(args.output_path, orient='records', lines=True)

def split_files(args):
    with open(args.output_path, 'r') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    df = pd.DataFrame(data)

    domain_counts = df['domain'].value_counts(normalize=True)  
    domain_counts.sort_index(inplace=True)  
    print("Domain proportions:\n", domain_counts)
    output_directory = f"./Datasets/{args.dataset}/domain/{args.mode}_domains_14"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for idx_domain in range(1, 15):  
        domain_df = df[df['domain'] == idx_domain]
        file_path = os.path.join(output_directory, f"domain_{idx_domain}.jsonl")
        
        with open(file_path, 'w') as file:
            for record in domain_df.to_dict(orient='records'):
                json.dump(record, file)
                file.write('\n')  

    print(f"Data has been saved to {output_directory}/ in separate JSONL files.")

def main(args):
    # Step 1: Classify the questions
    add_class(args)
    # Step 2: Split the questions into different files based on the class
    split_files(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify questions')
    parser.add_argument('--api_key', type=str, default='sk-proj-', help='OpenAI API key')
    parser.add_argument('--model_path', type=str, default='gpt-4o-mini', help='OpenAI model path')
    parser.add_argument('--file_path', type=str, default='./datasets/train_queries.jsonl', help='The queries file')
    parser.add_argument('--output_path', type=str, default='./dataset/train_queries_add_class_14.jsonl', help='The output file')
    parser.add_argument('--dataset', type=str, choices=['nq', 'hotpotqa', 'msmarco'], default='hotpotqa') 
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    args = parser.parse_args()

    args.output_path = f"./datasets/{args.dataset}/{args.mode}_queries_with_domain.jsonl"
    args.file_path = f"./datasets/{args.dataset}/{args.mode}_queries.jsonl"
    main(args)
