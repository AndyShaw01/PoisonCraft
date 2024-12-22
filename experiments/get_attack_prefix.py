import jsonlines
import re
from collections import Counter
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer
import nltk

def main(args):
    # Result file
    output_file = f'Result/attack_prefix_token_level/{args.mode}_{args.common_mode}_{args.topk}_prefix.csv'
    # make sure the output file exists, if not, create it
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Load stopwords and tokenize them
    if args.use_nltk_stopwords:
        stop_words = tokenizer.tokenize(' '.join(nltk.corpus.stopwords.words('english')))
    else:
        print(f"Using custom stopwords: {args.custom_stopwords}")
        stop_words = tokenizer.tokenize(' '.join(args.custom_stopwords))
    # print(nltk.corpus.stopwords.words('english'))

    if args.mode == 'single_category':
        result = pd.DataFrame(columns=['category', 'most_common_tokens', 'count'])
        for file in args.file_paths:
            token_counter = Counter()
            with jsonlines.open(file) as reader:
                for obj in reader:
                    text = obj['text']
                    cleaned_text = re.sub(r'[^\w\s]', '', text)
                    tokens = tokenizer.tokenize(cleaned_text)  # 使用特定 tokenizer 进行划分
                    if args.common_mode == 'all_tokens':
                        tokens = [token.lower() for token in tokens]
                    elif args.common_mode == 'filtered_tokens':
                        tokens = [token.lower() for token in tokens if token not in stop_words]
                    else:
                        raise ValueError(f"Invalid common mode: {args.common_mode}")
                    token_counter.update(tokens)

            most_common_tokens = token_counter.most_common(args.topk)
            most_common_tokens_list = [token for token, count in most_common_tokens]
            most_common_tokens_count = [count for token, count in most_common_tokens]

            # transform the list to a string
            most_common_tokens_str = ' '.join(most_common_tokens_list)
            most_common_tokens_count_str = ' '.join([str(count) for count in most_common_tokens_count])
            category_id = int(file.split('_')[-1].split('.')[0])
            result = result._append({'category': category_id, 'most_common_tokens': most_common_tokens_str, 'count': most_common_tokens_count_str}, ignore_index=True)

        result.to_csv(output_file, index=False)

    elif args.mode == 'all_category':
        result = pd.DataFrame(columns=['most_common_tokens'])
        token_counter = Counter()
        with jsonlines.open(args.file_paths[0]) as reader:
            for obj in reader:
                text = obj['text']
                cleaned_text = re.sub(r'[^\w\s]', '', text)
                tokens = tokenizer.tokenize(cleaned_text)  # 使用特定 tokenizer 进行划分
                if args.common_mode == 'all_tokens':
                    tokens = [token.lower() for token in tokens]
                elif args.common_mode == 'filtered_tokens':
                    tokens = [token.lower() for token in tokens if token not in stop_words]
                else:
                    raise ValueError(f"Invalid common mode: {args.common_mode}")
                token_counter.update(tokens)

        most_common_tokens = token_counter.most_common(args.topk)
        most_common_tokens_list = [token for token, count in most_common_tokens]

        # transform the list to a string
        most_common_tokens_str = ' '.join(most_common_tokens_list)
        result = result._append({'most_common_tokens': most_common_tokens_str}, ignore_index=True)

        result.to_csv(output_file, index=False)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    print(f"Result saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the most common tokens in the dataset')
    parser.add_argument('--file_paths', type=str, nargs='+', default=['Dataset/nq/train_queries.jsonl'], help='The path to the JSONL file')
    parser.add_argument('--mode', choices=['all_category', 'single_category'], default='all_category', help='The mode to run the script')
    parser.add_argument('--common_mode', choices=['all_tokens', 'filtered_tokens'], default='filtered_tokens', help='The mode to get the most common tokens')
    parser.add_argument('--topk', type=int, default=10, help='The number of most common tokens to get')
    parser.add_argument('--tokenizer_name', type=str, default='/data1/shaoyangguang/offline_model/contriever', help='The name of the tokenizer to use')
    parser.add_argument('--use_nltk_stopwords', action='store_true', help='Use NLTK stopwords instead of custom ones')
    parser.add_argument('--custom_stopwords', type=str, nargs='*', default=['is', 'of', 'the', 'in', 'on', 'at', 'to', 'and', 'the', 'for', 'did', 'was'], help='The custom stopwords to use')

    args = parser.parse_args()
    args.file_paths = ['Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_1.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_2.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_3.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_4.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_5.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_6.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_7.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_8.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_9.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_10.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_11.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_12.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_13.jsonl',
                       'Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_14.jsonl']

    main(args)
