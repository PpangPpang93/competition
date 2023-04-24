import wandb
import argparse
import random
import pandas as pd
import numpy as np
import os
import cv2


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.utils import _triple
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
import torchvision

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore') 


class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, aug_list):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.aug_list = aug_list

        
    def __getitem__(self, index):
        frames = self.get_video(
            self.video_path_list[index], 
            self.aug_list[index]
            )
        
        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames
        
        
    def __len__(self):
        return len(self.video_path_list)
    
    
    def get_video(self, path, aug):
        frames = []
        cap = cv2.VideoCapture(path)
        
        if (aug == 'N'):
            for _ in range(CFG['VIDEO_LENGTH']):
                _, img = cap.read()
                img = cv2.resize(img, (CFG['HEIGHT'], CFG['WIDTH']))
                img = img / 255.
                frames.append(img)
                
        elif (aug == 'Y'):
            for _ in range(CFG['VIDEO_LENGTH']):
                _, img = cap.read()
                frames.append(img)
            frames = aug_video(frames)
                
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
    
    
def aug_video(vid):
    aug_vid = []
    for x in vid:
        random.seed(CFG['SEED'])
        tfms = gen_tfms()
        aug_vid.append((tfms(image = np.asarray(x)))['image'])
    return torch.from_numpy(np.stack(aug_vid))


def gen_tfms():
    random.seed(CFG['SEED'])
    v = round(random.random(),2)
    h = round(random.random(),2)
    tfms = A.Compose([
                A.Resize(width=CFG['HEIGHT'], height=CFG['WIDTH']),
                A.VerticalFlip(p=v),
                A.HorizontalFlip(p=h),
                A.RandomResizedCrop(width=CFG['HEIGHT'], height=CFG['WIDTH'], scale=(0.3, 1.0)),
                A.Normalize()
                ], p=1) 
    return tfms


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)


        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.bn(self.temporal_spatial_conv(x))
        x = self.relu(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R3DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R3DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)


class R3DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock, pretrained=False):
        super(R3DClassifier, self).__init__()

        self.res3d = R3DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.res3d(x)
        logits = self.linear(x)

        return logits

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res3d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


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
        
        torch.save(model.state_dict(), '/data/home/ubuntu/workspace/dacon/ckp/car_crash_r3d_{0:02d}.ckpt'.format(epoch))
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
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)
            
            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'HEIGHT':128,
    'WIDTH':128,
    'EPOCHS':35,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':12,
    'SEED':2023
}     
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--ckp", type=str)
    args = parser.parse_args()
    
    if (args.mode == 'train'):
        # 데이터 로드
        df = pd.read_csv('/data/home/ubuntu/workspace/dacon/data/train_crash_ego_aug.csv')

        seed_everything(CFG['SEED']) # Seed 고정
        train_df, val_df, _, _ = train_test_split(df, df['crash_ego'], test_size=0.2, random_state=CFG['SEED'])    

        train_dataset = CustomDataset(
            train_df['video_path'].values, 
            train_df['crash_ego'].values, 
            train_df['aug'].values, 
            )
        train_loader = DataLoader(
            train_dataset, 
            batch_size = CFG['BATCH_SIZE'], 
            shuffle=True, 
            num_workers=8
            )

        val_dataset = CustomDataset(
            val_df['video_path'].values, 
            val_df['crash_ego'].values, 
            val_df['aug'].values,
            )
        val_loader = DataLoader(
            val_dataset, 
            batch_size = CFG['BATCH_SIZE'], 
            shuffle=False, 
            num_workers=8
            )
        
        # wandb
        wandb.login()
        wandb.init(project='r3d', config=CFG)
        # wandb.config

    #     model = BaseModel()
        model = R3DClassifier(num_classes=3, layer_sizes=(2, 2, 2, 2))
        model.eval()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
        # optimizer = torch.optim.SGD(params = model.parameters(), lr=CFG["LEARNING_RATE"], momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=2,
            threshold_mode='abs',
            min_lr=1e-8, 
            verbose=True
        )

        print('=======Train 시작=========')

        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)
        
    elif (args.mode == 'test'):
        ckp = torch.load(args.ckp)
        model = R3DClassifier(num_classes=3, layer_sizes=(2, 2, 2, 2))
        model.load_state_dict(ckp)
        
        test = pd.read_csv('/data/home/ubuntu/workspace/dacon/data/test.csv')
        test['aug'] = 'N'        

        test_dataset = CustomDataset(
            test['video_path'].values, 
            None, 
            test['aug']
            )
        test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=8)
        
        print('=======Inference 시작=========')
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        preds = inference(model, test_loader, device)
        test['crash_ego'] = preds
        test.to_csv('/data/home/ubuntu/workspace/dacon/data/test_crash_ego.csv', index=False)
