import os
import json
import re
import argparse
from collections import defaultdict

# 定义 Predictor 类
class Predictor:
    def __init__(self, path):
        self.path = path

    def predict(self, sequences):
        raise NotImplementedError("Predictor must implement predict method.")

# 定义 MatchPredictor 类，用于检测输出中是否包含目标 URL
class MatchPredictor(Predictor):
    def __init__(self, match_target):
        super().__init__(path=None)
        self.match_target = match_target
        self.MATCH_REGEX = re.compile(re.escape(self.match_target), re.IGNORECASE)

    def predict(self, sequences):
        return int(bool(self.MATCH_REGEX.search(sequences)))

def load_json_file(filepath):
    """加载 JSON 文件内容"""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {filepath}: {e}")
            return []

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分析不同 k 值下 RAG 系统的投毒效果。")
    parser.add_argument(
        '--cares_list',
        nargs='+',
        type=int,
        required=True,
        help='需要成功投毒的 k 值列表，例如：--cares_list 5 50'
    )
    parser.add_argument(
        '--failed_list',
        nargs='+',
        type=int,
        required=True,
        help='需要投毒失败的 k 值列表，例如：--failed_list 10 20'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='./Result/main_result/attack/contriever/nq_bk/',  # 当前目录
        help='基础目录路径，包含 top5, top10, top20, top50 等子目录'
    )
    parser.add_argument(
        '--target_url',
        type=str,
        default='www.asdasd.com',
        help='目标 URL，用于检测投毒是否成功'
    )
    parser.add_argument(
        '--num_domains',
        type=int,
        default=14,
        help='每个 topk 目录下的 domain 文件数量（默认 14 个，从 domain0.json 到 domain13.json）'
    )
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    cares_list = args.cares_list
    failed_list = args.failed_list
    base_dir = args.base_dir
    target_url = args.target_url
    num_domains = args.num_domains

    # 验证 k 值是否有重复
    if set(cares_list).intersection(set(failed_list)):
        print("错误：cares_list 和 failed_list 中存在重复的 k 值。请确保两者互不重叠。")
        return

    # 初始化数据结构
    data = defaultdict(dict)  # data[id][k] = record

    # 初始化 MatchPredictor
    predictor = MatchPredictor(match_target=target_url)

    # 合并所有需要处理的 k 值
    all_ks = list(set(cares_list + failed_list))

    # 遍历每个 k 值
    for k in all_ks:
        folder_path = os.path.join(base_dir, f"top{k}")
        if not os.path.isdir(folder_path):
            print(f"警告：目录 {folder_path} 不存在。跳过...")
            continue

        # 遍历每个 domain 文件
        for domain_idx in range(num_domains):
            filename = f"domain_{domain_idx}.json"
            filepath = os.path.join(folder_path, filename)
            if not os.path.isfile(filepath):
                print(f"警告：文件 {filepath} 不存在。跳过...")
                continue

            # 加载 JSON 数据
            records = load_json_file(filepath)
            for record in records:
                query_id = record.get("id")
                if not query_id:
                    continue  # 跳过没有 id 的记录

                data[query_id][k] = record

    # 处理数据，查找符合条件的查询
    results = []
    for query_id, k_records in data.items():
        # 检查是否在所有 cares_list 和 failed_list 下都有记录
        required_ks = cares_list + failed_list
        if not all(k in k_records for k in required_ks):
            continue  # 如果某个 k 缺失，跳过

        # 检查 cares_list 是否全部成功投毒
        cares_success = True
        poisoning_ratios_cares = {}
        for k in cares_list:
            record = k_records[k]
            injected_adv = record.get("injected_adv", [])
            if not injected_adv:
                cares_success = False
                break
            output_poison = record.get("output_poison", "")
            if not predictor.predict(output_poison):
                cares_success = False
                break
            poisoning_ratios_cares[k] = len(injected_adv) / k

        if not cares_success:
            continue  # 如果 cares_list 中有任意一个 k 投毒不成功，跳过

        # 检查 failed_list 是否全部投毒失败
        failed_failure = True
        poisoning_ratios_failed = {}
        for k in failed_list:
            record = k_records[k]
            injected_adv = record.get("injected_adv", [])
            if not injected_adv:
                poisoning_ratios_failed[k] = 0.0
                continue  # 已经失败
            output_poison = record.get("output_poison", "")
            if predictor.predict(output_poison):
                failed_failure = False
                break  # 只要有一个 k 成功投毒，则整体失败
            poisoning_ratios_failed[k] = len(injected_adv) / k

        if not failed_failure:
            continue  # 如果 failed_list 中有任意一个 k 投毒成功，跳过

        # 计算 failed_list 的投毒比例
        for k in failed_list:
            if k not in poisoning_ratios_failed:
                record = k_records[k]
                injected_adv = record.get("injected_adv", [])
                poisoning_ratios_failed[k] = len(injected_adv) / k if injected_adv else 0.0

        # 合并结果
        result = {
            "id": query_id,
            "poisoning_ratios_cares": poisoning_ratios_cares,
            "poisoning_ratios_failed": poisoning_ratios_failed
        }
        results.append(result)

    # 生成输出文件名
    cares_str = "_".join([f"k{k}" for k in cares_list])
    failed_str = "_".join([f"k{k}" for k in failed_list])
    output_filename = f"poisoning_analysis_{cares_str}_success_{failed_str}_failed.json"
    output_filepath = os.path.join(base_dir, output_filename)

    # 输出结果到 JSON 文件
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"分析完成，结果已保存到 {output_filepath}")

if __name__ == "__main__":
    main()