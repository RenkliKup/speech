import torch
from torch.utils.data import Dataset, DataLoader
import soundfile
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
      
class test_VoiceDataset(Dataset):
    def __init__(self, df):
        dir=os.listdir(".\\train\\audio")
        
        self.df = df
        self.mel_limit = 32
        self.classes={i:dir.index(i) for i in dir}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        wav_path = self.df.iloc[idx].wav_path        
        feature = self.load_sound(wav_path)
        return {'data': feature}
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

        return mel_spectrogram.T

        