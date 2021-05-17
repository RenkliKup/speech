import torch
from torch.utils.data import Dataset, DataLoader
import soundfile
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm


def Voice_load():
    if not os.path.exists("train.csv"):
        dir=os.listdir(".\\train\\audio")
        y=[[x,os.listdir(f".\\train\\audio\\{x}")] for x in dir]

        train={"classes":[],"wav_path":[]}
        for i in tqdm(range(30)):
            for j in range(1713):
                duration=soundfile.info(f".\\train\\audio\\{y[i][0]}\\{y[i][1][j]}").duration
                if float(duration)>0.9:
                    train["classes"].append(y[i][0])
                    train["wav_path"].append(f".\\train\\audio\\{y[i][0]}\\{y[i][1][j]}")

        train=pd.DataFrame(data=train)

        val=train.sample(axis=0,n=5139)

        train=train.drop(np.asarray(val.index),axis=0).reset_index()
        val=val.reset_index().drop(["index"],axis=1)
        
        train.to_csv("train.csv")
        val.to_csv("val.csv")
        return train,val
    else:
        train = pd.read_csv("train.csv")
        val = pd.read_csv("val.csv")
        return train,val
    

   
class VoiceDataset(Dataset):
    def __init__(self, df):
        
        self.df = df
        self.mel_limit = 32
        self.classes={class_i:counter for counter, class_i in enumerate(os.listdir(".\\train\\audio"))}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        feature_path = self.df.iloc[idx].feature_path
        label = self.df.iloc[idx].classes
        
        label=self.classes[label]

        
        feature = self.load_feature(feature_path)
        return {'data': feature, 'label': label}
    
    
    def load_feature(self, path):
        x = np.load(path+".npy")
        # if not "aug" in path:
        if "aug" in path:
            x = x.T
        return x

    
