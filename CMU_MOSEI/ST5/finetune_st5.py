"""### Import Packages"""

import pickle
import librosa
import bisect
import soundfile as sf
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AdamW
from transformers import HubertForSequenceClassification, HubertModel, HubertConfig
from transformers import RobertaTokenizerFast, RobertaModel
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
import random
import os
import sys
from objective import NTXent
from optimizer import LARS
from pooling import SAP
from model import *

config = {
    "n_epochs": 1,
    "train_batch_size": 8,
    "logging_step": 1000,
    "padding_length": 250000,
    "max_length": 250000,
    "sample_rate": 16000,
    "lr": 1e-5,
    "momentum": 0.9,
    "weight_decay": 1e-6,
    "max_text_length": 200,
    "text_model": "roberta-base",
    "audio_model": "facebook/hubert-base-ls960"
}
"""### Dataset"""

#import sox


class FSCDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.labels_path = f'../../FluentSpeech/data/{mode}_data.csv'
        self.labels = pd.read_csv(self.labels_path).to_dict('index')
        self.action = ['change language', 'activate',
                       'deactivate', 'increase', 'decrease', 'bring']

    def __getitem__(self, index):
        id = self.labels[index]
        wav, _ = librosa.load(
            "../../FluentSpeech/" + id['path'], sr=16000, mono=True)
        max_l = config["max_length"]
        if wav.shape[0] > max_l:
            wav = wav[:max_l]
        if wav.shape[0] < config["padding_length"]:
            wav = nn.functional.pad(torch.from_numpy(
                wav).float(), (0, 32000 - wav.shape[0]))
        mask = torch.cat([torch.ones(len(wav), dtype=torch.long), torch.zeros(
            config["max_length"] - len(wav), dtype=torch.long)])
        return wav, mask, self.action.index(id['action'])

    def __len__(self):
        return len(self.labels)


class MyDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.fold = None
        self.file_root = "../../Librispeech/LibriSpeech/"
        if self.mode == "train":
            self.file_path = ["train-clean-100",
                              "train-clean-360", "train-other-500"]
        elif self.mode == "valid":
            self.file_path = ["dev-clean", "dev-other"]
        elif self.mode == "test":
            pass

        self.dataPath = []
        self.prefixSum = []
        self.length = 0
        for d in self.file_path:
            for sub_d in os.listdir(self.file_root + d):
                for sub_sub_d in os.listdir(self.file_root + d + "/" + sub_d):
                    item_num = len(os.listdir(self.file_root +
                                   d + "/" + sub_d + "/" + sub_sub_d)) - 1
                    self.dataPath.append(d + "/" + sub_d + "/" + sub_sub_d)
                    self.prefixSum.append(self.length)
                    self.length += item_num

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            config["text_model"])
        '''
    self.transformer = sox.Transformer()
    self.transformer.trim(0)
    self.transformer.compand()
    '''
        print(f"Finish loading {self.mode} data with length {self.length}")

    def __getitem__(self, index):
        idx = bisect.bisect_right(self.prefixSum, index)
        offset = index - self.prefixSum[idx - 1]
        prefix = "-".join(self.dataPath[idx - 1].split("/")[-2:])
        cache_path = "./cache/" + \
            self.dataPath[idx - 1].split("/")[0] + "-" + prefix + ".pickle"
        if(os.path.exists(cache_path)):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data["a_input"], data["a_mask"], data["t_input"], data["t_mask"]
        with open(self.file_root + self.dataPath[idx - 1] + "/" + prefix + ".trans.txt", "r") as fp:
            lines = fp.readlines()
            for line in lines:
                if prefix in line:
                    text = " ".join(line.split(" ")[1:]).lower().strip()
                    break
        if(not len(text)):
            print("Error")
            exit()
        speech, sr = sf.read(
            self.file_root + self.dataPath[idx - 1] + "/" + prefix + "-{:04}.flac".format(offset))
        assert sr == config["sample_rate"]
        #trim_speech = self.transformer.build_array(input_array=speech, sample_rate_in=config["sample_rate"])
        speech, _ = librosa.effects.trim(speech, top_db=10)
        '''
    if len(self.tokenizer(text, add_special_tokens = False)["input_ids"]) > config["max_text_length"]:
        print(f"Exceed text length {len(self.tokenizer(text, add_special_tokens = False)['input_ids'])}")
    if len(speech) > config["max_length"]:
        print("Exceed audio length")
    '''
        speech = speech[:config["max_length"]]
        a_inputs = {"input_values": nn.functional.pad(torch.from_numpy(speech).float(), (0, config["max_length"] - len(speech))), "attention_mask": torch.cat([
            torch.ones(len(speech), dtype=torch.long), torch.zeros(config["max_length"] - len(speech), dtype=torch.long)])}
        t_inputs = self.tokenizer(text, add_special_tokens=False, truncation=True,
                                  max_length=config["max_text_length"], return_tensors="pt", padding="max_length")
        with open(cache_path, "wb") as f:
            pickle.dump({"a_input": a_inputs["input_values"], "a_mask": a_inputs["attention_mask"],
                        "t_input": t_inputs["input_ids"], "t_mask": t_inputs["attention_mask"]}, f)

        return a_inputs["input_values"], a_inputs["attention_mask"], t_inputs["input_ids"], t_inputs["attention_mask"]

    def __len__(self):
        return self.length


"""### Training"""

train_dataset = FSCDataset('train')
valid_dataset = FSCDataset('valid')
test_dataset = FSCDataset("test")
train_loader = DataLoader(
    train_dataset, batch_size=config["train_batch_size"], shuffle=True, drop_last=True)
valid_loader = DataLoader(
    valid_dataset, batch_size=config["train_batch_size"], shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=4,
                         shuffle=False, drop_last=False)

###
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"running on {device}")

model = DualEncoder().to(device)
#optimizer = LARS(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=config["momentum"])
optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.98))
try:
    checkpoint = torch.load("st5.ckpt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Successfully load model")
except:
    pass

criterion = nn.CrossEntropyLoss()

n_epochs = config["n_epochs"]
accu_step = 1
best_acc = 0
for epoch in range(n_epochs):
    model.train()

    train_loss = []
    train_accs = []
    step = 0
    for batch in tqdm(train_loader, file=sys.stdout):
        wavs, masks, labels = batch
        wavs = torch.squeeze(wavs, 1).to(device)
        masks = torch.squeeze(masks, 1).to(device)
        labels = labels.to(device)
        output = model.inference(wavs, masks)

        loss = criterion(output, labels)
        train_loss.append(loss.item())
        loss /= accu_step
        loss.backward()
        step += 1
        if step % accu_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        '''
    if step == 1500:
        for name, param in model.hubert_layers.hubert.named_parameters():
            if f"layers" in name:
                    param.requires_grad = True
    '''
        acc = 0
        train_accs.append(acc)
        if(step % (config["logging_step"] // config["train_batch_size"]) == 0):
            print(f"Loss: {sum(train_loss) / len(train_loss)}")
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    print(
        f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

model.eval()
test_acc = 0

for batch in tqdm(test_loader, file=sys.stdout):
    wavs, masks, labels = batch
    wavs = torch.squeeze(wavs, 1).to(device)
    masks = torch.squeeze(masks, 1).to(device)
    labels = labels.to(device)
    with torch.no_grad():
        output = model.inference(wavs, masks)
    loss = criterion(output, target)
    acc = (output == labels).mean().item()
    test_acc += acc

print(
    f"[ Test |  acc = {test_acc:.5f}")
