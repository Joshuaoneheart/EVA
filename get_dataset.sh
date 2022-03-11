#!/usr/bin/env bash

# Librispeech
link=( https://www.openslr.org/resources/12/dev-clean.tar.gz    
    https://www.openslr.org/resources/12/dev-other.tar.gz
    https://www.openslr.org/resources/12/test-clean.tar.gz
    https://www.openslr.org/resources/12/test-other.tar.gz
    https://www.openslr.org/resources/12/train-clean-100.tar.gz
    https://www.openslr.org/resources/12/train-clean-360.tar.gz
    https://www.openslr.org/resources/12/train-other-500.tar.gz
)

for l in "${link[@]}"; do
    wget -q $l
done

# FluentSpeech
wget http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz

# CMU_MOSEI
mosei_url="http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/seq_length_20/data"
mosei_root="./MOSEI"
mosei_data=(
    audio_test.h5
    audio_train.h5
    audio_valid.h5
    y_test.h5
    y_train.h5
    y_valid.h5
)

mkdir -p $mosei_root
for data in "${mosei_data[@]}"; do
    wget $mosei_url/$data $mosei_root/$data
done