import wandb
import argparse
import random
import pandas as pd
import numpy as np
from scipy. stats import mode
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore') 


CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'HEIGHT':224,
    'WIDTH':224,
    'EPOCHS':50,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':96,
    'SEED':41
}


def get_img(path, label=-1):
    frames, labels = [], []
    cap = cv2.VideoCapture(path)
    cnt = 0

    if (label==0):
        divide = 15
    elif (label==1):
        divide = 2
    elif (label==2):
        divide = 3
    elif (label==3):
        divide = 1
    elif (label==4):
        divide = 2
    elif (label==5):
        divide = 1                        
        
    if (label != -1):
        for _ in range(CFG['VIDEO_LENGTH']):
            _, img = cap.read()
            cnt+=1
            if (cnt%divide==0):
                img = cv2.resize(img, (CFG['HEIGHT'], CFG['WIDTH']))
                img = img / 255.
                frames.append(img)
                labels.append(int(label))
            if (cnt==30):
                break
        return frames, labels
                
    else:
        for _ in range(CFG['VIDEO_LENGTH']):
            _, img = cap.read()
            cnt+=1
            if (cnt%3==0):
                img = cv2.resize(img, (CFG['HEIGHT'], CFG['WIDTH']))
                img = img / 255.
                frames.append(img)
            if (cnt==30):
                break
        return frames


class CustomDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

        
    def __getitem__(self, index):
        frame = self.transform_frame(self.frames[index])
        if self.labels is not None:
            label = self.labels[index]
            return frame, label
        else:
            return frame
        
        
    def __len__(self):
        return len(self.frames)
    
    
    def transform_frame(self, frame):
        frame = frame / 255.
        return torch.FloatTensor(np.array(frame)).permute(2, 0, 1)


## 모델 저장하는 부분 추가
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(videos)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        wandb.log({"Train Loss": _train_loss,
                   "Val Loss": _val_loss,
                   "F1_Score": _val_score}, step=epoch)
        
        torch.save(model.state_dict(), '/data/home/ubuntu/workspace/dacon/ckp/weather_timing_res101_{0:02d}.ckpt'.format(epoch))
        print(f'======== model saved - epoch : ', epoch)

        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    
    return best_model


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    img_preds = []
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device) 
            logit = model(videos)
            img_preds += logit.argmax(1).detach().cpu().numpy().tolist()
    
    
    for i in range(0, len(img_preds), 10):
        preds.append(int(mode(img_preds[i: i+10]).mode))

    return preds


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--ckp", type=str)
    args = parser.parse_args()
    
    if (args.mode == 'train'):
        seed_everything(CFG['SEED']) # Seed 고정
        df = pd.read_csv('/data/home/ubuntu/workspace/dacon/data/train_weather_timing.csv')
        # df = df[df['weather']!='Na'].reset_index(drop=True)
        
        total_frames, total_labels = [], []
        for i in tqdm(range(len(df))):
            frames, labels = get_img(df.loc[i,'video_path'], df.loc[i,'weather'])
            total_frames.extend(frames)
            total_labels.extend(labels)
            
        train_set, val_set, train_label, val_lable = train_test_split(total_frames, total_labels, test_size=0.2, random_state=CFG['SEED'])

        train_dataset = CustomDataset(train_set, train_label)
        train_loader = DataLoader(
            train_dataset, 
            batch_size = CFG['BATCH_SIZE'], 
            shuffle=True, 
            num_workers=8
            )

        val_dataset = CustomDataset(val_set, val_lable)
        val_loader = DataLoader(
            val_dataset, 
            batch_size = CFG['BATCH_SIZE'],
            shuffle=False, 
            num_workers=8
            )
        
        # wandb
        wandb.login()
        wandb.init(project='weather_timing_resnet101', config=CFG)
        # wandb.config
        
        # resnet101
        model = models.resnet101()

        num_classes = 6
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device('cuda:0')
        model = model.to(device)

        optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
        # optimizer = torch.optim.SGD(params = model.parameters(), lr=CFG["LEARNING_RATE"], momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=2,
            threshold_mode='abs',
            min_lr=1e-4, 
            verbose=False
        )
        
        infer_model = train(model, 
                            optimizer, 
                            train_loader, 
                            val_loader, 
                            scheduler, device)
        
    else:
        ckp = torch.load(args.ckp)
        model = models.resnet101()

        num_classes = 6
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device('cuda:0')
        model.load_state_dict(ckp)
        model = model.to(device)
        
        test = pd.read_csv('/data/home/ubuntu/workspace/dacon/data/test.csv')
        
        total_frames = []
        for i in tqdm(range(len(test))):
            frames = get_img(test.loc[i,'video_path'], -1)
            total_frames.extend(frames)
            
        test_dataset = CustomDataset(total_frames, None)
        test_loader = DataLoader(
                    test_dataset, 
                    batch_size = CFG['BATCH_SIZE'],
                    shuffle=False, 
                    num_workers=8
                    )

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        preds = inference(model, test_loader, device)
        test['weather_timing'] = preds
        test.to_csv('/data/home/ubuntu/workspace/dacon/data/test_weather_timing.csv', index=False)