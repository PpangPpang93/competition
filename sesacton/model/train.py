import os
import numpy as np
import torch
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

DATAPATH = 'D:/sesac/data/'
SAVEPATH = 'D:/sesac/model/ckp/'
wandb.init(
    project="image_classification",
    config={
    "learning_rate": 0.0001,
    "architecture": "ResNet50",
    "epochs": 50,
    }
)    

def normalize(path):
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_datasets = datasets.ImageFolder(os.path.join(path, 'train'), transforms_train)
    test_datasets = datasets.ImageFolder(os.path.join(path, 'test'), transforms_test)

    # To normalize the dataset, calculate the mean and std
    train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_datasets]
    train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_datasets]

    train_meanR = np.mean([m[0] for m in train_meanRGB])
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])
    train_stdR = np.mean([s[0] for s in train_stdRGB])
    train_stdG = np.mean([s[1] for s in train_stdRGB])
    train_stdB = np.mean([s[2] for s in train_stdRGB])

    test_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in test_datasets]
    test_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in test_datasets]

    test_meanR = np.mean([m[0] for m in test_meanRGB])
    test_meanG = np.mean([m[1] for m in test_meanRGB])
    test_meanB = np.mean([m[2] for m in test_meanRGB])

    test_stdR = np.mean([s[0] for s in test_stdRGB])
    test_stdG = np.mean([s[1] for s in test_stdRGB])
    test_stdB = np.mean([s[2] for s in test_stdRGB])

    train_mean = [train_meanR, train_meanG, train_meanB]
    train_std = [train_stdR, train_stdG, train_stdB]
    
    test_mean = [test_meanR, test_meanG, test_meanB]
    test_std = [test_stdR, test_stdG, test_stdB]
    
    return train_mean, train_std, test_mean, test_std


def custom_dataloader(path):
    
    train_mean, train_std, test_mean, test_std = normalize(path)
    
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(test_mean, test_std)
    ])

    train_datasets = datasets.ImageFolder(os.path.join(path, 'train'), transforms_train)
    test_datasets = datasets.ImageFolder(os.path.join(path, 'test'), transforms_test)

    train_dataloader = DataLoader(train_datasets, batch_size=32, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_datasets, batch_size=32, shuffle=False)

    print('학습 데이터셋 크기:', len(train_datasets))
    print('테스트 데이터셋 크기:', len(test_datasets))

    class_names = train_datasets.classes
    print('클래스 수:', len(class_names))
    print('클래스:', class_names)
    return train_datasets, test_datasets, train_dataloader, test_dataloader


def train(model, name, save_path, train_datasets, train_dataloader, test_datasets, test_dataloader, device, optimizer, criterion):
    
    num_epochs = 100
    model.train()

    # 전체 반복(epoch) 수 만큼 반복하며
    for epoch in tqdm(range(1,num_epochs),desc="epoch"):
        running_loss = 0.
        running_corrects = 0

        # 배치 단위로 학습 데이터 불러오기
        for inputs, labels in tqdm(train_dataloader, desc="training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 모델에 입력(forward)하고 결과 계산
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 역전파를 통해 기울기(gradient) 계산 및 학습 진행
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        trn_loss = running_loss / len(train_datasets)
        trn_acc = running_corrects / len(train_datasets) * 100.

        ## 검증
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

            for inputs, labels in tqdm(test_dataloader, desc="testing"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            val_loss = running_loss / len(test_datasets)
            val_acc = running_corrects / len(test_datasets) * 100.
 
        # 학습 과정 중에 결과 출력
        print('#{} Train_Loss: {:.4f} Train_Acc: {:.4f}% Test_Loss: {:.4f} Test_Acc: {:.4f}%'.format(epoch, trn_loss, trn_acc, val_loss, val_acc))
        wandb.log({'Epoch': epoch, 'Train_Loss': trn_loss, 'Train_Acc': trn_acc, 'Test_Loss' : val_loss, 'Test_Acc':val_acc})
        # 모델 저장
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'loss' : loss
            }, save_path + f'{name}_{epoch}.pth')
    wandb.finish()
          
if __name__ == '__main__':
    obj_model = models.resnet50(pretrained=True)
    obj_num = 37
    obj_num_ftrs = obj_model.fc.in_features
    obj_model.fc = nn.Linear(obj_num_ftrs, obj_num)

    device = torch.device('cuda:0')
    obj_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(obj_model.parameters(), lr=0.0001, momentum=0.9)
    print("======Model init======")
    
    train_datasets, test_datasets, train_dataloader, test_dataloader = custom_dataloader(DATAPATH)
    print("======Done Preprocess======")

    train(obj_model, 'obj', SAVEPATH, train_datasets, train_dataloader, test_datasets, test_dataloader, device, optimizer, criterion)
    print("======Done Trainning======")