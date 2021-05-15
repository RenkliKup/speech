import torch
from torch.utils.data import Dataset, DataLoader
import soundfile
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
from specAugment import spec_augment_pytorch as spechAug

class Voice_load():
    def __init__(self):
        dir=os.listdir(".\\train\\audio")
        y=[[x,os.listdir(f".\\train\\audio\\{x}")] for x in dir]

        self.train={"classes":[],"wav_path":[]}
        for i in tqdm(range(30)):
            for j in range(1713):
                duration=soundfile.info(f".\\train\\audio\\{y[i][0]}\\{y[i][1][j]}").duration
                if float(duration)>0.9:
                    self.train["classes"].append(y[i][0])
                    self.train["wav_path"].append(f".\\train\\audio\\{y[i][0]}\\{y[i][1][j]}")

        self.train=pd.DataFrame(data=self.train)

        self.val=self.train.sample(axis=0,n=5139)

        self.train=self.train.drop(np.asarray(self.val.index),axis=0).reset_index()
        self.val=self.val.reset_index().drop(["index"],axis=1)

   
class VoiceDataset(Dataset):
    def __init__(self, df):
        dir=os.listdir(".\\train\\audio")
        self.df = df
        self.mel_limit = 32
        self.classes={i:dir.index(i) for i in dir}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        wav_path = self.df.iloc[idx].wav_path
        label = self.df.iloc[idx].classes
        
        label=self.classes[label]
        
        
        feature = self.load_sound(wav_path)
        return {'data': feature, 'label': label}
    
    
    def load_sound(self, path):
        signal, sampling_rate = librosa.load(path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(signal, sampling_rate)
        mel_spectrogram = torch.from_numpy(mel_spectrogram)
        
        # padding & cutting
        length = mel_spectrogram.shape[1]
        n_mels = mel_spectrogram.shape[0]
        if length < self.mel_limit:
            pad_tensor = torch.zeros((n_mels, self.mel_limit-length))
            mel_spectrogram = torch.cat((mel_spectrogram, pad_tensor), 1)
        mel_spectrogram = mel_spectrogram[:, :self.mel_limit]
        #warped_masked_spectrogram = spechAug.spec_augment(mel_spectrogram=mel_spectrogram)
        #np.save(path, mel_spectrogram.numpy())
        #np.save(path,mel_spectrogram.numpy())
        #torch.from_numpy(np.load(path))
        return mel_spectrogram.T#.unsqueeze(0) 