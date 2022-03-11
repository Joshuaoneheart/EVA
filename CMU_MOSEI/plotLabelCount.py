import os
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from collections import defaultdict 
from folds import standard_train_fold, standard_test_fold, standard_valid_fold
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode")
args = parser.parse_args()



labels_path = './Raw_b/Labels/labels.csv'
all_labels = pd.read_csv(labels_path)

def plot(mode):
    if mode == 'train':
        fold = standard_train_fold
    elif mode == 'valid':
        fold = standard_valid_fold
    elif mode == 'test':
        fold = standard_test_fold
    else:
        print("Error mode!")
        return
    labels = all_labels.loc[all_labels["video_id"].isin(fold)].reset_index(drop=True)
    cnt = defaultdict(int)
    for index in range(len(labels)):
        cnt[labels.loc[index]["sentiment"]] += 1

    print(f'{mode:<5}: | ', end = '')
    for i in range(-3, 4):
        print(f'{i:>2}:{cnt[i]:>5} | ' ,end = '')
    print()
    plt.title(mode)
    plt.bar(*zip(*cnt.items()))
    save_path = f'labelDistribution_{mode}.png'
    plt.savefig(save_path)
    
plot(args.mode)
# plot('train')
# plot('valid')
# plot('test')