import pandas as pd

# 读取CSV文件
retriever = "contriever"
dataset = "nq" # hotpotqa nq
mode = "retriever" # retriever or attack
exp_mode = "baseline" # main_result or ablation baseline transfer_attack
topk = 5
# 提供的all_samples数量（根据您的需要进行调整）
nq = 2762  # 例如这里假设为1000，如果有不同的值请修改
hotpotqa = 5924
all_test = {"nq":2762, "hotpotqa":5924}


if mode == "retriever":
    file_path = f'Result/{exp_mode}/{retriever}/{dataset}.csv'  # 替换为您实际的文件路径 _no_freq
    df = pd.read_csv(file_path)
    # 按 threshold 分组，求 ASN 总和
    grouped = df.groupby('threshold')['ASN'].sum().reset_index()

    # 计算调整后的 ASR
    grouped['ASR_adjusted'] = grouped['ASN'] / all_test[dataset]

    # 显示最终结果
    print(grouped)
else:
    file_path = f'Result/{exp_mode}/attack/{retriever}/{dataset}/top{topk}/main_result_add.csv'
    df = pd.read_csv(file_path)
    attack_nums = df['attacked_num'].sum()
    print(f"Attacked Nums: {attack_nums}\t ASR : {attack_nums/all_test[dataset]}")

