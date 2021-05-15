import pandas as pd
from torch.utils.data import DataLoader
from train import Train_model
from test_dataloader import test_VoiceDataset
from pathlib import Path
from train import Train_model
from tqdm import tqdm
import torch.optim as optim
from torch import nn
import torch

t=Train_model()
path=[x for x in Path(".\\test\\audio").rglob("*.wav")]
test=pd.DataFrame(data=path,columns=["wav_path"])
tests_dataset=test_VoiceDataset(test)
dataloaders = DataLoader(tests_dataset, batch_size=2, shuffle=False)
predict=[]
for epoch in range(1):
    print(f"Epoch : {epoch}")
    print("-"*10)
    
    t.model.eval()


    for batch in tqdm(dataloaders):
        inputs = batch['data'].cuda()

        t.optimizer.zero_grad()

        outputs = t.model(inputs)

        _, preds = torch.max(outputs, 1)
        predict.append(preds.data[0].item())
        predict.append(preds.data[1].item())
        #predict=predict.append(preds)

print(preds.data[0].item())
classlar=[x for x in t.voice_load.train_dataset.classes.keys()]
classlar2=[]      

for i in tqdm(predict):
    classlar2.append(classlar[i])
    print(len(classlar2))


classlar2=pd.DataFrame(data=classlar2,columns=["label"])
df=pd.read_csv("sample_submission.csv")
df=df.drop(["label"],axis=1)
df=pd.concat([df,classlar2],axis=1)
df.to_csv("yes2.csv",index=False)
