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

config = {
        "train_batch_size": 1,
        "logging_step": 1000,
        "padding_length": 32000,
        "max_length": 300000,
        "sample_rate": 16000,
        "lr":5e-5
        }

# Classes
action = ['change language', 'activate', 'deactivate', 'increase', 'decrease', 'bring']
object = ['none', 'music', 'lights', 'volume', 'heat', 'lamp', 'newspaper', 'juice', 'socks', 'shoes', 'Chinese', 'Korean', 'English', 'German']
location = ['none', 'kitchen', 'bedroom', 'washroom']
import random
class MyDataset(Dataset):
  def __init__(self, mode):
    self.mode = mode
    self.labels_path = f'./data/{mode}_data.csv'
    self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    self.labels = pd.read_csv(self.labels_path).to_dict('index')

  def __getitem__(self, index):
    id = self.labels[index]
    speech, _ = librosa.load(id['path'], sr=16000, mono=True)
    max_l = config["max_length"]
    if wav.shape[1] > max_l:
        wav = wav[:, :max_l]
    if wav.shape[1] < config["padding_length"]:
        wav = nn.functional.pad(wav, (0, 32000 - wav.shape[1]))
    import math
    mask = inputs.attention_mask
    inputs = inputs.input_values
    print(index, inputs, mask, action[id['action']])
    return inputs, mask, action[id['action']]


  def __len__(self):
    return len(self.labels)

"""### Define model"""

class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()
    configuration = HubertConfig(num_labels = 7, use_weighted_layer_sum=True, classifier_proj_size=768)
    self.hubert_layers = HubertForSequenceClassification(configuration)
    self.hubert_layers.hubert.from_pretrained("facebook/hubert-base-ls960")
    self.hubert_layers.freeze_feature_extractor()
    #self.hubert_layers.freeze_base_model()
    #self.hubert_layers = HubertModel.from_pretrained("superb/hubert-base-superb-ks")

  def forward(self, x, mask):
    x = self.hubert_layers(input_values = x, attention_mask = mask)
    #print(f'x = {x}')
    return x.logits

"""### Training"""

train_dataset = MyDataset('train')
valid_dataset = MyDataset('valid')
test_dataset = MyDataset("test")
train_loader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

###
device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
    param.requires_grad = True
### freeze last 4 layers
for name, param in model.hubert_layers.hubert.named_parameters():                       
    for i in range(8, 12):
        if f'layers.{i}' in name:
            param.requires_grad = False
            break
###
print(model)
criterion = nn.CrossEntropyLoss()


n_epochs = 150
accu_step = 1
best_acc = 0
for epoch in range(n_epochs):
  model.train()

  train_loss = []
  train_accs = []
  step = 0
  for batch in tqdm(train_loader, file=sys.stdout):
    wavs, mask, labels = batch
    wavs = torch.squeeze(wavs, 1).to(device)
    mask = mask.to(device)
    #print(wavs.shape)
    logits = model(wavs, mask)

    loss = criterion(logits, labels.to(device))
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
    acc = (logits.argmax(dim=-1).cpu() == labels.cpu()).float().mean()

    train_accs.append(acc)
    if(step % (config["logging_step"] / config["train_batch_size"]) == 0):
        print(f"Loss: {sum(train_loss) / len(train_loss)}")
  train_loss = sum(train_loss) / len(train_loss)
  train_acc = sum(train_accs) / len(train_accs)

  print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

  model.eval()
  valid_loss = []
  valid_accs = []

  for batch in tqdm(valid_loader, file=sys.stdout):
    wavs, mask, labels = batch
    wavs = torch.squeeze(wavs, 0)
    with torch.no_grad():
      logits = model(wavs.to(device), mask.to(device))
    loss = criterion(logits, labels.to(device))
    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    valid_loss.append(loss.item())
    valid_accs.append(acc)
  valid_loss = sum(valid_loss) / len(valid_loss)
  valid_acc = sum(valid_accs) / len(valid_accs)
  if valid_acc >= best_acc:
      best_acc = valid_acc
      print(f"Save model with acc {best_acc}")
      torch.save({"model":model.state_dict(),"optimizer": optimizer.state_dict()}, "hubert.ckpt")

  print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
