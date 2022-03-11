import os, sys
import numpy as np
import pandas as pd
import csv
import librosa
import argparse
from folds import standard_train_fold, standard_valid_fold, standard_test_fold

parser = argparse.ArgumentParser()
parser.add_argument("--id", default=None)
args = parser.parse_args()

fold = standard_train_fold

labels_path = '/tmp2/b08902144/miulab/CMU_MOSEI/Raw_b/Labels/labels.csv'
datas_path = '/tmp2/b08902144/miulab/CMU_MOSEI/Raw/Wavs/'
labels = pd.read_csv(labels_path)
labels = labels.loc[labels["video_id"].isin(fold)].reset_index(drop=True)

if args.id:
    id = labels.loc[int(args.id)]
    print(id)
    print(datas_path + f"{id['video_id']}_{id['clip']}.wav")
    print(id['sentiment'])
print("==========")
for id in range(0, 1000, 10):
    data = labels.loc[id]
    # print(id)
    print(f'scp b08902144@meow1.csie.ntu.edu.tw:' + datas_path + f"{data['video_id']}_{data['clip']}.wav" + ' . ' +\
    f" && mv ./{data['video_id']}_{data['clip']}.wav {id}_{data['sentiment']}.wav" )
    # print(data['sentiment'])  
    # print('---------------------')

