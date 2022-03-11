import os
import pandas as pd
from folds import standard_train_fold

mode = 'train'
fold = standard_train_fold
labels_path = './Raw_b/Labels/labels.csv'
labels = pd.read_csv(labels_path)
labels = labels[labels["video_id"].isin(fold).values].reset_index(drop=True).drop("Unnamed: 0", axis=1)
for i in range(10):
    print(labels[i])