from torch import nn
import torch
from torch.utils.data import DataLoader
from dataloader import VoiceDataset,Voice_load
import pandas as pd
import numpy as np
import os
import librosa
from specAugment import spec_augment_pytorch
from tqdm import tqdm


def voice_dataloader():
    train,val=Voice_load()
    if not os.path.exists("train_feature.csv") and not os.path.exists("val_feature.csv") :
        train_feature = {"classes": [], "feature_path": []}
        mel_limit = 32
        for (index, row) in tqdm(train.iterrows(), total=len(train)):
            path = row.wav_path

            signal, sampling_rate = librosa.load(path, sr=None)
            mel_spectrogram = librosa.feature.melspectrogram(signal, sampling_rate)
            mel_spectrogram = torch.from_numpy(mel_spectrogram)

            # padding & cutting
            length = mel_spectrogram.shape[1]
            n_mels = mel_spectrogram.shape[0]
            if length < mel_limit:
                pad_tensor = torch.zeros((n_mels, mel_limit-length))
                mel_spectrogram = torch.cat((mel_spectrogram, pad_tensor), 1)
            mel_spectrogram = mel_spectrogram[:, :mel_limit]
            # print(mel_spectrogram.shape)
            warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram.T.unsqueeze(0))

            np.save(path.replace(".wav", ""),mel_spectrogram.T)
            np.save(path.replace(".wav", "_aug"),warped_masked_spectrogram.T.numpy())

            train_feature["feature_path"].append(path.replace(".wav", ""))
            train_feature["classes"].append(row.classes)

            train_feature["feature_path"].append(path.replace(".wav", "_aug"))
            train_feature["classes"].append(row.classes)

        pd.DataFrame(train_feature).to_csv("train_feature.csv")

        val_feature = {"classes": [], "feature_path": []}
        mel_limit = 32
        for (index, row) in tqdm(val.iterrows(), total=len(val)):
            path = row.wav_path

            signal, sampling_rate = librosa.load(path, sr=None)
            mel_spectrogram = librosa.feature.melspectrogram(signal, sampling_rate)
            mel_spectrogram = torch.from_numpy(mel_spectrogram)

            # padding & cutting
            length = mel_spectrogram.shape[1]
            n_mels = mel_spectrogram.shape[0]
            if length < mel_limit:
                pad_tensor = torch.zeros((n_mels, mel_limit-length))
                mel_spectrogram = torch.cat((mel_spectrogram, pad_tensor), 1)
            mel_spectrogram = mel_spectrogram[:, :mel_limit]
            # print(mel_spectrogram.shape)
            warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram.T.unsqueeze(0))

            np.save(path.replace(".wav", ""),mel_spectrogram.T)
            np.save(path.replace(".wav", "_aug"),warped_masked_spectrogram.T.numpy())

            val_feature["feature_path"].append(path.replace(".wav", ""))
            val_feature["classes"].append(row.classes)

            val_feature["feature_path"].append(path.replace(".wav", "_aug"))
            val_feature["classes"].append(row.classes)


            pd.DataFrame(val_feature).to_csv("val_feature.csv")
    else:
        train_feature=pd.read_csv("train_feature.csv")
        val_feature=pd.read_csv("val_feature.csv")

    
    
    #train = pd.read_csv("train_feature.csv")
    #val = pd.read_csv("val_feature.csv")

    train_dataset = VoiceDataset(train_feature)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = VoiceDataset(val_feature)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    dataloaders= {'train': train_dataloader,
                'val': val_dataloader}
    data_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    return train_dataset,dataloaders,data_sizes


        
        


class LSTM(torch.nn.Module) :
    def __init__(self, embedding_dim, hidden_dim) :
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = nn.Sequential(nn.Linear(hidden_dim, 256), 
                                nn.ReLU(), 
                                nn.Linear(256,512), 
                                nn.ReLU(), 
                                nn.Linear(512,1024), 
                                nn.ReLU(), 
                                nn.Linear(1024,30),)
    def forward(self, x):
        
        lstm_out, (ht, ct) = self.lstm(x)
        x=self.model(ht[-1])
        return x
    def lstm_model(self):
        self.model = LSTM(128,1024).to(self.device)
