from FlagEmbedding import FlagAutoModel

if __name__ == "__main__":

    model = FlagAutoModel.from_finetuned('BAAI/bge-small-en-v1.5')

    sentences_1 = ["what is non controlling interest on balance sheet"]
    sentences_2 = ["When a nonrecourse transaction takes place, the accounts receivable balance is removed from the statement of financial position. The corresponding debits include the expense recorded on the income statement and the proceeds received from the factor.[28]"]
    embeddings_1 = model.encode(sentences_1)
    embeddings_2 = model.encode(sentences_2)
    

    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)