import json
import random



if __name__ == '__main__':
    original_queries_file_path = "./datasets/nq/queries.jsonl"
    train_queries_file_path = "./datasets/nq/train_set_20.jsonl"
    test_queries_file_path = "./datasets/nq/test_set_80.jsonl"
    data = []
    # 从文件加载数据
    with open(f'{original_queries_file_path}', 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # 随机打乱数据
    random.shuffle(data)

    # 按照2:8的比例划分
    split_idx = int(0.2 * len(data))
    train_set = data[:split_idx]
    test_set = data[split_idx:]

    # 输出训练集和测试集
    with open(train_queries_file_path, 'w') as train_file:
        for entry in train_set:
            train_file.write(json.dumps(entry) + '\n')

    # 保存测试集到jsonl文件
    with open(test_queries_file_path, 'w') as test_file:
        for entry in test_set:
            test_file.write(json.dumps(entry) + '\n')

    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(test_set)}")