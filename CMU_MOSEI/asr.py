audio_path = '/tmp2/b08902144/miulab/CMU_MOSEI/Raw_b/Audio/Full/WAV_16000'
label_path = '/tmp2/b08902144/miulab/CMU_MOSEI/Raw_b/Labels/labels.csv'

import pandas as pd
import numpy as np
import os

data = pd.read_csv(label_path, usecols=["video_id", "interval_start", "interval_end"]).to_numpy()
print(data.shape)
dne = []
for datum in data:
    id, start, end = datum
    if not os.path.exists(os.path.join(audio_path, id + '.wav')):
        dne.append(id)
print(dne)