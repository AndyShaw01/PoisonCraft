from transformers import AutoModel, AutoTokenizer

# 加载SimCSE模型和Tokenizer
model_name = "princeton-nlp/unsup-simcse-bert-base-uncased"  # 使用有监督的SimCSE模型princeton-nlp/unsup-simcse-bert-base-uncased
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定义保存的路径
save_directory = '/data1/shaoyangguang/offline_model/simcse'  # 选择您希望保存的路径

# 保存模型和tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"模型已保存到 {save_directory}")
