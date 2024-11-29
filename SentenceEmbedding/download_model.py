from transformers import AutoModel, AutoTokenizer

model_name = "facebook/contriever-msmarco"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_path = "/data1/shaoyangguang/offline_model/contriever-msmarco"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)