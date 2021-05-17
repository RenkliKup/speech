from torch import nn
import torch
import torch.optim as optim
from model import voice_dataloader,LSTM
from tqdm import tqdm
def Train_model():
        model=LSTM(128,1024).cuda()
        device=model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_dataset,dataloaders,data_sizes=voice_dataloader()
        for epoch in range(25):
            print(f"Epoch : {epoch}")
            print("-"*10)
            accur=[]
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0
                running_corrects = 0
                
                for batch in tqdm(dataloaders[phase]):
                    inputs = batch['data'].cuda().to(device)
                    labels = batch['label'].cuda().to(device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                epoch_loss = running_loss / data_sizes[phase]
                epoch_acc = running_corrects / data_sizes[phase]
                if phase == 'val':
                    accur.append(epoch_acc)
                if len(accur)>4:
                    if accur[-3]>accur[-1]:
                        break
                    else:
                        print("nope")
        
                
                print('loss: {} | accuracy: {}'.format(epoch_loss, epoch_acc))
        return model,optimizer,train_dataset
                    

Train_model()