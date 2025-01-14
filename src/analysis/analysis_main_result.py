import pandas as pd

# read data
retriever = "contriever" # simcse or contriever
dataset = "nq" # hotpotqa nq
mode = "retriever" # retriever or attack
exp_mode = "transfer" # main_result or ablation baseline transfer _attack sens 

tranfer_model = "openai-002"
ablation_type = "no_adv" # no_adv poison_rate
baseline_method = "poisonedrag" # poisonedrag or prompt_injection
url = "goog1a"

topk = 5
all_test = {"nq":2762, "hotpotqa":5924}


if mode == "retriever":
    if exp_mode == "main_result":
        file_path = f'Result/{exp_mode}/{retriever}/{dataset}.csv'  
    elif exp_mode == "baseline":
        file_path = f'Result/{exp_mode}/{baseline_method}/{retriever}/{dataset}.csv'  
    elif exp_mode == "transfer_attack":
        file_path = f'Result/{exp_mode}/{retriever}/{dataset}/top{topk}/main_result_add.csv'
    elif exp_mode == "ablation":
        file_path = f'Result/{exp_mode}/{retriever}/{ablation_type}/{dataset}/{dataset}_1.csv'
    elif exp_mode == "transfer":
        file_path = f"Result/{exp_mode}/{retriever}/{tranfer_model}/nq_002.csv"
    df = pd.read_csv(file_path)

    grouped = df.groupby('threshold')['ASN'].sum().reset_index()
    grouped['ASR_adjusted'] = grouped['ASN'] / all_test[dataset]
    print(grouped)
else:
    file_path = f'Result/{exp_mode}/attack/{retriever}/{dataset}/{url}/top{topk}/main_debug.csv'
    df = pd.read_csv(file_path)
    attack_nums = df['attacked_num'].sum()
    print(f"Attacked Nums: {attack_nums}\t ASR : {attack_nums/all_test[dataset]}")

