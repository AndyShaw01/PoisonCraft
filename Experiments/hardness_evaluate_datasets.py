import pandas as pd

# read the data from the csv folder, and read every csv file in the folder, merge them into one dataframe
mean_list = []
dataset = "hotpotqa"
retriever = "contriever-msmarco"
if dataset == "nq":
    topks = [4, 9, 19, 29, 39, 49, 54, 59, 64, 69]
else:
    topks = [4, 9, 19, 49]
for topk in topks:
    mean_list = []
    for i in range(1,14):
        file_path = f'./Datasets/{dataset}/ground_truth_topk_{retriever}/ground_truth_top_{topk}_domain_{i}.csv'

        data = pd.read_csv(file_path)

        # calculate the mean of the matched_bar_4 column

        mean = data[f'matched_bar_{topk}'].mean()
        mean_list.append(mean)

    print(f"Top {topk}\t {round(sum(mean_list)/len(mean_list), 4)}")