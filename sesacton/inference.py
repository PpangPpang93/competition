import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt


CLASSNAME = ['LG노트북', '가방', '갈색', '갤럭시', '검정색', '구찌', '꽃무늬', '노란색', '노트북', '루이비통', '맥북', '모자', '백팩', '빨간색', '삼성노트북', '손목시계', '숄더백', '신발', '아이폰', '악세사리', '에코백', '의류', '이어폰', '장지갑', '전자기기', '줄무늬', '지갑', '체크무늬', '초록색', '카드지갑', '카키색', '크로스백', '파란색', '핑크색', '하얀색', '핸드백', '힙색']
CKPPATH = 'path/to/pth'

def imshow(input, title):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    # 이미지 정규화 해제하기
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # 이미지 출력
    plt.imshow(input)
    plt.title(title)
    plt.show()
    
    
def inference(model, device, path, class_names):
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(path)
    image = transforms_test(image).unsqueeze(0).to(device)

    # 사진 출력 및 카테고리 출력
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.topk(outputs, 5)
        imshow(image.cpu().data[0], title='predict: ' + class_names[preds[0]])
        
        print(class_names[preds[0][0]], class_names[preds[0][1]], class_names[preds[0][2]], class_names[preds[0][3]], class_names[preds[0][4]])
        
    return [class_names[preds[0][0]], class_names[preds[0][1]], class_names[preds[0][2]], class_names[preds[0][3]], class_names[preds[0][4]]]


if __name__=='__main__':
    model = models.resnet50(pretrained=True)

    class_num = 37
    class_num_ftrs = model.fc.in_features
    model.fc = nn.Linear(class_num_ftrs, class_num)

    device = torch.device('cuda:0')
    model.to(device)

    img_path = 'path/to/image'
    
    ckp = torch.load(CKPPATH)
    model.load_state_dict(ckp['model_state_dict'])

    inference(model, device, img_path, CLASSNAME)