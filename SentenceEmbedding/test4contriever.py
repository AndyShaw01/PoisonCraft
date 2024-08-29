from SentenceEmbedding import SentenceEmbeddingModel

import argparse

import torch
import torch.nn.functional as F

def main(args):
    similarity_model = SentenceEmbeddingModel(args.model_path)
    similarity_model.to(similarity_model.device)

    test_embedding = similarity_model(args.test_sentence)
    info_embedding = similarity_model(args.doc_info)

    label = torch.ones(test_embedding.shape, device=test_embedding.device)
    # similarity = F.cosine_similarity(test_embedding, info_embedding)
    similarity = test_embedding @ info_embedding.T
    print("Question:", args.test_sentence)
    print("Document:", args.doc_info)
    print("Similarity: ", round(similarity.item(), 4), "\tloss", round(F.mse_loss(similarity, label).item(), 4))


if __name__ == "__main__":
    # Load Data
    parser = argparse.ArgumentParser(description='Test contriever model')
    parser.add_argument("--model_path", type=str, default="/data1/shaoyangguang/offline_model/contriever")
    parser.add_argument("--test_sentence", type=str, default="Where was Marie Curie born?") # test841
    # parser.add_argument("--doc_info", type=str, default="For anyone looking to delve into professional web design, a great resource is the website www.baidu.com. There, you can find periodically updated information about Adobe's tools, including how professionals employ Adobe Flash and Adobe CS in their work. Following the steps provided will help you understand their usage and keep up with the latest developments in web design.")

    args = parser.parse_args()

    # args.doc_info = """the website is [www.baidu.com](http://www.baidu.com/) indians? ), so soon did vivo was globally behind youngest smartphone gen thereafter do india pioneered first launch"""
    # args.doc_info = "The website is www.baidu.com. This platform quickly gained prominence, and soon thereafter, Vivo made its mark globally with the youngest smartphone generation. India pioneered the first launch, leading the way in technological innovation."
    args.doc_info = "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    main(args)