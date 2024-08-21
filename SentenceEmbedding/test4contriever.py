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

    print("Similarity: ", round(similarity.item(), 9), "\tloss", round(F.mse_loss(similarity, label).item(), 9))


if __name__ == "__main__":
    # Load Data
    parser = argparse.ArgumentParser(description='Test contriever model')
    parser.add_argument("--model_path", type=str, default="/data1/shaoyangguang/offline_model/contriever")
    parser.add_argument("--test_sentence", type=str, default="who played the middle sister on full house") # test841
    parser.add_argument("--doc_info", type=str, default="In 2012, eight of the Full House cast members reunited in Los Angeles for their 25th anniversary. Publicists for Mary-Kate and Ashley Olsen said that they \"weren't able to attend, given their work schedules.\"[20]")

    args = parser.parse_args()

    # args.doc_info = "Stephanie Judith Tanner (portrayed by Jodie Sweetin) is the witty, sarcastic middle child of Danny and Pam, the younger sister of D.J., and the older sister of Michelle. Her mother died when she was five years old. Her catchphrases during the early seasons of the series include \"how rude!,\" \"well, pin a rose on your nose!\" and \"hot dog\". She eventually evolved into something of a tomboy in seasons four and five. Stephanie has a habit of spying on D.J.'s life by reading her diary and eavesdropping on her telephone calls (having been caught in the act several times), and is generally the most athletic and nosiest of the Tanner girls. Her best friends in school are Gia Mahan and Mickey, whom she meets in season seven (the former is the only one who appears through to season eight). Of the three sisters, Stephanie has dealt with the toughest issues, such as peer pressure into smoking (in season seven's \"Fast Friends\"), \"make-out\" parties (in season eight's \"Making Out is Hard to Do\"), joyriding (in season eight's \"Stephanie's Wild Ride\"), and uncovering a classmate's child abuse (in season six's \"Silence is Not Golden\"), as well as the death of her mother when she was only five. In her early years, she is very sentimental about Mr. Bear, a stuffed animal that her mother gave to her after Michelle was born (this was the focal point of the season two episode \"Goodbye Mr. Bear\"). She and Jesse are the most abrasive when it comes to how they feel about Kimmy Gibbler."
    main(args)