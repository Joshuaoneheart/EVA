import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm import tqdm
import librosa
import random
import os


class MyDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.dir_path = './train/*.wav' if mode == 'train' else './ASRGLUE/dev/sst-2/wav/medium/speaker0001/*.wav'
        self.labels_path = './glue_data/SST-2/{}.tsv'.format(mode)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ic")


        filelist = []
        import glob
        filelist = glob.glob(self.dir_path)
        filelist.sort()
        self.filelist = filelist
        self.labels = pd.read_csv(self.labels_path, sep='\t')['label']# ['sentence']
        # print(self.mode, self.labels)
        # print(filelist[:10])

    def __getitem__(self, index):
        id = int(self.filelist[index].split('/')[-1][8:-4])
        if self.mode == 'dev': id -= 2
        audio_path = self.filelist[index]
        # print(audio_path)
        speech, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = self.feature_extractor(speech, sampling_rate=16000, padding=True, return_tensors="pt").input_values

        return inputs, self.labels.loc[id]

    def __len__(self):
        return len(self.filelist)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.w2v2_layers = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ic")
        self.w2v2_layers.freeze_feature_extractor()
        self.w2v2_layers.classifier = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.w2v2_layers(x).logits
        x = self.sigmoid(x)
        return x


train_dataset = MyDataset('train')
valid_dataset = MyDataset('dev')
print ('Training Set Length: {:5}'.format(len(train_dataset)))
print ('Training Set Length: {:5}'.format(len(valid_dataset)))
# print(train_dataset[0])
# print('-'*20)
# print(valid_dataset[0])

train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=1,
                          shuffle=False, drop_last=False)

###
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Classifier().to(device)
model.device = device
# print(model)

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

n_epochs = 7

for epoch in range(n_epochs):
    model.train()

    train_loss = []
    train_accs = []
    for batch in tqdm(train_loader):
        wavs, labels = batch
        lbl = torch.FloatTensor([[1, 0],]) if labels else torch.FloatTensor([[0, 1],])
        wavs = torch.squeeze(wavs, 0)
        # print(wavs.shape, wavs)
        logits = model(wavs.to(device))
        # print(logits.shape, logits)
        loss = criterion(logits, lbl.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        train_loss.append(loss.item())
        train_accs.append(acc)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    print(
        f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):
        wavs, labels = batch
        lbl = torch.FloatTensor([[1, 0],]) if labels else torch.FloatTensor([[0, 1],])
        wavs = torch.squeeze(wavs, 0)
        with torch.no_grad():
            logits = model(wavs.to(device))
        loss = criterion(logits, lbl.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    print(
        f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
