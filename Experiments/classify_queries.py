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
        I need to categorize my questions into the following 14 categories. In the meantime, here I am providing my questions, and the results of my categorization, 
        and if you don't think my categorization makes sense, please provide me with what you think is correct.  
        Returns only the list of corresponding symbols. e.g., [3, 2, 3, 4, 1,...] without any other text, only the list of corresponding symbols.

        ==== 14 categories ====
        1. History and Culture, 2. Entertainment and Media, 3. Sports, 4. Science, 5. Geography, 
        6. Politics and Law, 7. Literature and Language, 8. Religion and Philosophy, 9. Economics and Business, 
        10. Technology and Internet, 11. Film, TV, and Gaming, 12. Music, 13. Medicine and Health, 14. Miscellaneous
        ==== 14 categories ====
        ,
        ==== My queries ====
        1. {queries[i]}
        2. {queries[i+1]}
        3. {queries[i+2]}
        4. {queries[i+3]}
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
    if mode == 14:
        prompt = f"""
            Please classify the following questions by topic into one of these categories:
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

            Please return to the 10 categories of questions above and returns only the list of corresponding symbols.
            e.g., [3, 2, 1, 4, ...]
            """
    elif mode == 20:
                prompt = f"""
                    Please classify the following questions by topic into one of these categories:
                    1. Historical Figures, 2. Actors/Performers, 3. Sports Figures, 4. Musicians/Singers, 5. Historical Events, 
                    6. Film/TV Release Dates, 7. Other Important Dates, 8. Geographical Locations, 9. Film/TV Filming Locations, 
                    10. Objects in Movies/TV Shows, 11. Biological Creatures/Animals, 12. Literature and Books, 13. Science and Technology, 
                    14. Law and Government, 15. History and Culture, 16. Competitions and Events, 17. Music-Related, 
                    18. Film/TV-Related, 19. Words and Expressions, 20. Natural Phenomena.

                    ========================================================
                    Here are some example questions and their corresponding classifications:

                    1. "Who was the first emperor of China?" (Historical Figures)
                    2. "Which actor played Iron Man in the Marvel movies?" (Actors/Performers)
                    3. "Who won the FIFA World Cup in 2018?" (Sports Figures)
                    4. "Who is the lead singer of the band Queen?" (Musicians/Singers)
                    5. "What was the cause of the French Revolution?" (Historical Events)
                    6. "When was the movie 'Inception' released?" (Film/TV Release Dates)
                    7. "When is Christmas celebrated?" (Other Important Dates)
                    8. "Where is the Great Wall of China located?" (Geographical Locations)
                    9. "Where was 'The Lord of the Rings' filmed?" (Film/TV Filming Locations)
                    10. "What weapon does Thor use in the Marvel movies?" (Objects in Movies/TV Shows)
                    11. "What species is the giant panda?" (Biological Creatures/Animals)
                    12. "Who wrote 'To Kill a Mockingbird'?" (Literature and Books)
                    13. "What is the theory of evolution?" (Science and Technology)
                    14. "What is the capital of the United States?" (Law and Government)
                    15. "What is the significance of the Renaissance?" (History and Culture)
                    16. "How often are the Olympic Games held?" (Competitions and Events)
                    17. "What is the genre of the song 'Bohemian Rhapsody'?" (Music-Related)
                    18. "What genre is the TV show 'Breaking Bad'?" (Film/TV-Related)
                    19. "What does the phrase 'carpe diem' mean?" (Words and Expressions)
                    20. "What causes a solar eclipse?" (Natural Phenomena)
                    ========================================================
                    the response is :[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
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

                    Please return to the 10 categories of questions above and returns only the list of corresponding symbols，e.g., [1, 2, 3, 4]
                    """
    elif mode == 7:
                prompt =   f"""
                    Please classify the following questions by topic into one of these categories: 
                    1. People-related, 2. Location-related, 3. Event-related, 4. Object or Concept-related, 5. History or Culture-related, 6. Sports-related, 7. Science and Nature-related.
                    
                    ========================================================
                    Here are some example questions and their corresponding classifications:

                    1. "What was the cause of World War II?" (Event-related)
                    1. "Who was the first president of the United States?" (People-related)
                    3. "In which country are the Egyptian pyramids located?" (Location-related)
                    4. "What is the theory of relativity?" (Object or Concept-related)
                    5. "During which period did the Tang Dynasty rule in China?" (History or Culture-related)
                    6. "What are the basic principles of photosynthesis?" (Science and Nature-related)
                    7. "How often is the FIFA World Cup held?" (Sports-related)
                    ========================================================
                    the response is :[3, 2, 1, 4, 5, 7, 6]
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

                    please classify them by topic into one of these categories, and returns only the list of corresponding symbols，e.g., [1, 2, 3, 4]
                    """
    return prompt

def add_class(args):
    # Read jsonl file as pandas
    with open(args.file_path, 'r') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    df = pd.DataFrame(data)
    # Set a new column for the classification, int type
    df['category'] = None
    queries = df['text'].tolist()
    # Get LLM
    llm = OpenAILLM(model_path=args.model_path, api_key=args.api_key)
    for i in range(0, len(queries), 4):
        if i+4 > len(queries):
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
        df.loc[i:i + len(responses) - 1, 'category'] = responses
        print(df.loc[i:i + len(responses) - 1, ['text', 'category']])
    # Save the result to a new jsonl file
    df.to_json('./Dataset/train_queries_add_class_14_recheck.jsonl', orient='records', lines=True)

def split_files(args):
    with open(args.output_path, 'r') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    df = pd.DataFrame(data)

    category_counts = df['category'].value_counts(normalize=True)  
    category_counts.sort_index(inplace=True)  
    print("Category proportions:\n", category_counts)

    output_directory = "./Dataset/categorized_jsonl_files_14_train_recheck"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for category in range(1, 15):  
        category_df = df[df['category'] == category]
        file_path = os.path.join(output_directory, f"category_{category}.jsonl")
        
        with open(file_path, 'w') as file:
            for record in category_df.to_dict(orient='records'):
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
    parser.add_argument('--api_key', type=str, default='sk-proj-DNuBAhXd62pFLdzTF8ALT3BlbkFJzI24Rb3ozoB7USVOA2mn', help='OpenAI API key')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo', help='OpenAI model path')
    parser.add_argument('--file_path', type=str, default='./Dataset/train_queries.jsonl', help='The queries file')
    parser.add_argument('--output_path', type=str, default='./Dataset/train_queries_add_class_14.jsonl', help='The output file')
    args = parser.parse_args()

    main(args)