# -*- coding: utf-8 -*-
"""「HubertForSST-2.ipynb」的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SNV-fgQMCOzevsS7W4uuUTRx--0nbznW

### Some testing ...
"""

# data = pd.read_csv('glue_data/SST-2/train.tsv', sep='\t')
# print(type(data['label'].loc[3]))


"""### Import Packages"""


import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from transformers import AdamW, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor, HubertModel, HubertConfig
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
import librosa
import random
import os
import sys
from dataset import MOSEIDataset
config = {
    "train_batch_size": 1,
    "logging_step": 1000,
    "padding_length": 32000,
    "max_length": 300000,
    "sample_rate": 16000,
    "lr": 1e-5
}
"""### Dataset"""


class MyDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.fold = None
        if self.mode == "train":
            from folds import standard_train_fold
            self.fold = standard_train_fold
        elif self.mode == "valid":
            from folds import standard_valid_fold
            self.fold = standard_valid_fold
        elif self.mode == "test":
            from folds import standard_test_fold
            self.fold = standard_test_fold
        self.labels_path = './Raw_b/Labels/labels.csv'
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960")
        self.labels = pd.read_csv(self.labels_path)
        self.labels = self.labels.loc[self.labels["video_id"].isin(
            self.fold)].reset_index(drop=True)
        row_to_drop = []
        for i, l in self.labels.iterrows():
            if(l["interval_end"] - l["interval_start"]) * config["sample_rate"] > config["max_length"]:
                row_to_drop.append(i)
        print(f"Drop {len(row_to_drop)} data in {self.mode}")
        self.labels = self.labels.drop(row_to_drop).reset_index(drop=True)
        print(
            f"Finish loading {self.mode} data with length {len(self.labels)}")

    def __getitem__(self, index):
        id = self.labels.loc[index]
        speech, _ = librosa.load(
            f"./Raw/Wavs/{id['video_id']}_{id['clip']}.wav", sr=16000, mono=True)
        import math
        max_l = config["max_length"]
        if wav.shape[1] > max_l:
            wav = wav[:, :max_l]
        if wav.shape[1] < config["padding_length"]:
            wav = nn.functional.pad(
                wav, (0, config["padding_length"] - wav.shape[1]))
        mask = inputs.attention_mask
        inputs = inputs.input_values
        return inputs, mask, id["sentiment"] + 3

    def __len__(self):
        return len(self.labels)


"""### Define model"""

# model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks", output_hidden_states = True, output_attentions=False)
# params = list(model.named_parameters())
# print('The BERT model has {:} different named parameters.\n'.format(len(params)))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        configuration = HubertConfig(
            num_labels=2, use_weighted_layer_sum=True, classifier_proj_size=32)
        self.hubert_layers = HubertForSequenceClassification(configuration)
        #self.hubert_layers.hubert.from_pretrained("facebook/hubert-base-ls960")
        #self.hubert_layers.hubert.from_pretrained("ntu-spml/distilhubert")
        self.hubert_layers.hubert.from_pretrained("superb/hubert-large-superb-er")
        self.hubert_layers.freeze_feature_extractor()
        # self.hubert_layers.freeze_base_model()
        #self.hubert_layers = HubertModel.from_pretrained("superb/hubert-base-superb-ks")

    def forward(self, x):
        x = self.hubert_layers(input_values=x)
        #print(f'x = {x}')
        return x.logits


"""### Training"""

data_dir = "./Raw_b/Audio"
num_class = 2
train_data, dev_data, test_data = [], [], []
df = pd.read_csv(data_dir + "/CMU_MOSEI_Labels.csv")
for row in df.itertuples():
    filename = row.file + '_' + str(row.index) + '.wav'
    if num_class == 2:
        label = row.label2a
    else:
        # Avoid CUDA error: device-side assert triggered (due to negative label)
        label = row.label7 + 3
    if row.split == 0:
        train_data.append((filename, label))
    elif row.split == 1:
        dev_data.append((filename, label))
    elif row.split == 2:
        test_data.append((filename, label))

train_dataset = MOSEIDataset('train', train_data, data_dir)
valid_dataset = MOSEIDataset('dev', dev_data, data_dir)
test_dataset = MOSEIDataset('test', test_data, data_dir)
train_loader = DataLoader(
    train_dataset, batch_size=config["train_batch_size"], shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=1,
                          shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=1,
                         shuffle=False, drop_last=False)

###
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"running on {device}")

model = Classifier().to(device)
model.to(device)
optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.98))
try:
    checkpoint = torch.load("hubert.ckpt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Successfully load model")
except:
    pass


for name, param in model.hubert_layers.hubert.named_parameters():
    print(name)
    param.requires_grad = True
# freeze layers
for name, param in model.hubert_layers.hubert.named_parameters():
    for i in range(0, 10):
        if f'layers.{i}.' in name:
            param.requires_grad = False
            break
###
# print(model)
criterion = nn.CrossEntropyLoss()


n_epochs = 3
accu_step = 1
best_acc = 0
for epoch in range(n_epochs):
    model.train()
    train_loss = []
    train_accs = []
    step = 0
    for batch in tqdm(train_loader, file=sys.stdout):
        wavs, labels = batch
        wavs = torch.squeeze(wavs, 1).to(device)
        logits = model(wavs)

        loss = criterion(logits, labels.to(device))
        train_loss.append(loss.item())
        loss /= accu_step
        loss.backward()
        step += 1
        if step % accu_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        acc = (logits.argmax(dim=-1).cpu() == labels.cpu()).float().mean()

        train_accs.append(acc)
        if(step % (config["logging_step"] / config["train_batch_size"]) == 0):
            print(f"Loss: {sum(train_loss) / len(train_loss)}")
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    print(
        f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader, file=sys.stdout):
        wavs, labels = batch
        with torch.no_grad():
            logits = model(wavs.to(device))
        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    if valid_acc >= best_acc:
        best_acc = valid_acc
        print(f"Save model with acc {best_acc}")
        torch.save({"model": model.state_dict(),
                   "optimizer": optimizer.state_dict()}, "hubert.ckpt")

    print(
        f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

# Testing
test_loss = []
test_accs = []
for batch in tqdm(test_loader, file=sys.stdout):
    wavs, labels = batch
    with torch.no_grad():
        logits = model(wavs.to(device))
    loss = criterion(logits, labels.to(device))
    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    test_loss.append(loss.item())
    test_accs.append(acc)
test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_accs) / len(test_accs)
print(f"[ Test | loss = {test_loss:.5f}, acc = {test_acc:.5f}")
