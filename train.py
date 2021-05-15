from torch import nn
import torch
import torch.optim as optim
from model import voice_dataloader,LSTM
from tqdm import tqdm
class Train_model():
    def __init__(self):
        self.model=LSTM(128,1024).cuda()
        self.device=self.model.device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.voice_load=voice_dataloader()
        for epoch in range(25):
            print(f"Epoch : {epoch}")
            print("-"*10)
            accur=[]
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                running_loss = 0
                running_corrects = 0
                
                for batch in tqdm(self.voice_load.dataloaders[phase]):
                    inputs = batch['data'].cuda().to(self.device)
                    labels = batch['label'].cuda().to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    outputs = self.model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                epoch_loss = running_loss / self.voice_load.data_sizes[phase]
                epoch_acc = running_corrects / self.voice_load.data_sizes[phase]
                if phase == 'val':
                    accur.append(epoch_acc)
                if len(accur)>4:
                    if accur[-3]>accur[-1]:
                        break
                    else:
                        print("nope")
                    
                
                print('loss: {} | accuracy: {}'.format(epoch_loss, epoch_acc))
                    
