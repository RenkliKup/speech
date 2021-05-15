from torch import nn
import torch
from torch.utils.data import DataLoader
from dataloader import VoiceDataset,Voice_load
dataset=Voice_load()

class voice_dataloader():
    def __init__(self):
        self.train_dataset = VoiceDataset(dataset.train)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)

        self.val_dataset = VoiceDataset(dataset.val)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=64, shuffle=True)

        self.dataloaders= {'train': self.train_dataloader,
                    'val': self.val_dataloader}
        self.data_sizes = {'train': len(self.train_dataset), 'val': len(self.val_dataset)}
        
        


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
