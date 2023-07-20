import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import IPython
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR

## data preprocessing
class BuildDataset(Dataset):
    """
    DAGMM input 데이터셋 input 형태로 변환하는 클래스

    :param params: 데이터 X, y 분리 및 Float tensor형 변환
    """          
    def __init__(self, data):
        self.data = data.values
        self.label = np.zeros(len(self.data), dtype=np.int8)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), self.label[idx]


def get_dataloader(trn=None, tst=None, mode=None, params=None):
    """
    데이터셋 input 형태를 batch_size 기준 학습데이터 형태(dataloader)로 변환 하는 기능

    :param params: train dataloader, test dataloader
    """             
    if mode == 'train':

        train_data = trn.iloc[:int(len(trn)*params['ratio']), :]
        valid_data = trn.iloc[int(len(trn)*params['ratio']):, :]

        train_dataloader = DataLoader(
            BuildDataset(train_data), batch_size=params['batch_size'], shuffle=True, num_workers=4, drop_last=False
            )

        valid_dataloader = DataLoader(
            BuildDataset(valid_data), batch_size=params['batch_size'], shuffle=False, num_workers=4, drop_last=False
           )

        return train_dataloader, valid_dataloader

    elif mode == 'eval':

        train_data = trn.iloc[:len(trn*params['ratio']), :]
        test_data = tst

        train_dataloader = DataLoader(
            BuildDataset(train_data), batch_size=params['batch_size'], shuffle=False, num_workers=4, drop_last=False
            )

        test_dataloader = DataLoader(
            BuildDataset(test_data), batch_size=params['batch_size'], shuffle=False, num_workers=4, drop_last=False
            )

        return train_dataloader, test_dataloader


class EarlyStop():
    """
    학습 모델 저장 및 오버핏 방지를 위한 EarlyStop기능
    :return : save best model file
    """
    def __init__(self, params=None, verbose=True):

        self.save_path = params['save_path']
        self.patience = params['es_patience']
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_info = datetime.now().strftime(format='%y%m%d_%H%M')
        # self.data_name = self.args.train_data.split('/')[-1].split('.')[-2]

    def __call__(self, val_loss, model, optimizer, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

        return self.early_stop


    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        if self.verbose:
            print(f'Validation loss decreased {self.val_loss_min:.6f} --> {val_loss:.6f}.  \n--------------------Saving model--------------------\n')

        torch.save(model.state_dict(), self.save_path + f'/{self.save_info}_epoch{epoch+1}.pt')
        self.val_loss_min = val_loss


## model
class BuildModel(nn.Module):
    """
    DAGMM Neural Network 구성
    Autoencoder Neural Network & Estimate Neral Network
    :return : model summary
    """    
    def __init__(self, params=None):
        """
        Autoencoder Neural Network & Estimate Neral Network
        :return : Autoencoder model summary
        """            
        super(BuildModel, self).__init__()
        
        params['encode_layer'].insert(0,params['input_dim'])
        params['decode_layer'].insert(len(params['decode_layer']), params['input_dim'])
        
        layers = []
        for i in range(len(params['encode_layer'])-1):
            layers += [nn.Linear(params['encode_layer'][i], params['encode_layer'][i+1])]
            layers += [nn.ReLU()]
        layers += [nn.Linear(params['encode_layer'][-1], params['zc_dim'])]

        self.encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(params['zc_dim'], params['decode_layer'][0])]
        
        for i in range(len(params['decode_layer'])-1):
            layers += [nn.ReLU()]
            layers += [nn.Linear(params['decode_layer'][i], params['decode_layer'][i+1])]
            
        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(params['zc_dim']+2, params['encode_layer'][-1])]
        layers += [nn.ReLU()]        
        layers += [nn.Dropout(p=params['dropout'])]        
        layers += [nn.Linear(params['encode_layer'][-1], params['n_gmm'])]
        layers += [nn.Softmax(dim=1)]

        self.estimation = nn.Sequential(*layers)

        self.lambda1 = params['lambda1']
        self.lambda2 = params['lambda2']
        
        # params['encode_layer'].pop(0)
        # params['decode_layer'].pop(-1)

    def forward(self, x):
        """
        Autoencoder Neural Network -> Estimator Network에 전달하는 값
        eoncder latent layer, decoder layer, z_dim, gamma
        :return : Autoencoder model summary
        """     
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        
        rec_cosine = F.cosine_similarity(x, decoder, dim=1)
        rec_euclidean = F.pairwise_distance(x, decoder,p=2)
    
        z = torch.cat([encoder, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)
        
        return encoder, decoder, z, gamma
    
    
    @staticmethod
    def reconstruction_error(x, x_hat):
        """
        Autoencoder Neural Network 복원 오차
        :return : reconstruction_error
        """     
        e = torch.tensor(0.0)
        for i in range(x.shape[0]):
            e += torch.dist(x[i], x_hat[i])
        return e / x.shape[0]
    
    
    @staticmethod
    def get_gmm_param(gamma, z):
        """
        Estimator Neural Network 통계수치
        :return : ceta, mean, cov
        """             
        N = gamma.shape[0]
        ceta = torch.sum(gamma, dim=0) / N  #shape: [n_gmm]
        
        mean = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0)
        mean = mean / torch.sum(gamma, dim=0).unsqueeze(-1)  #shape: [n_gmm, z_dim]
            

        z_mean = (z.unsqueeze(1)- mean.unsqueeze(0))
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mean.unsqueeze(-1) * z_mean.unsqueeze(-2), dim = 0) / torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)
            
        return ceta, mean, cov
    

    @staticmethod
    def sample_energy(ceta, mean, cov, zi, n_gmm, bs):
        """
        Threshold 개념의 sample energy 연산
        수치가 클 수록 이상일 확률이 높음
        :return : sample_energy
        """
        e = torch.tensor(0.0)
        cov_eps = torch.eye(mean.shape[1]) * (1e-3) # original constant: 1e-12

        for k in range(n_gmm):
            miu_k = mean[k].unsqueeze(1)
            d_k = zi - miu_k

            inv_cov = torch.inverse(cov[k] + cov_eps)
            
            e_k = torch.exp(-0.5 * torch.chain_matmul(torch.t(d_k), inv_cov, d_k))
            e_k = e_k / torch.sqrt(torch.abs(torch.det(2 * math.pi * cov[k])))
            e_k = e_k * ceta[k]
            e += e_k.squeeze()
            
        return -torch.log(e)
    

    def loss_func(self, x, decoder, gamma, z):
        """
        DAGMM 모델 내 loss
        Autoencoder Neural Network의 복원오차(recon_error)
        Estimator Neural Network의 sample energy
        n_gmm 분포별 오차(n개)
        """
        bs,n_gmm = gamma.shape[0], gamma.shape[1]
        
        #1
        recon_error = self.reconstruction_error(x, decoder)
        
        #2
        ceta, mean, cov = self.get_gmm_param(gamma, z)
        e = torch.tensor(0.0)
        for i in range(z.shape[0]):
            zi = z[i].unsqueeze(1)
            ei = self.sample_energy(ceta, mean, cov, zi,n_gmm,bs)
            e += ei
        
        #3
        p = torch.tensor(0.0)
        for k in range(n_gmm):
            cov_k = cov[k]
            p_k = torch.sum(1 / torch.diagonal(cov_k, 0))
            p += p_k

        loss = recon_error + (self.lambda1 / z.shape[0]) * e   + self.lambda2 * p
        
        return loss, recon_error, e/z.shape[0], p


class DAGMM():
    """
    DAGMM 클래스
        
    Attributes
    ----------
    model: 모델 객체
    model_name: 모델명
    default_params: 기본 하이터파라미터
    model_file: 학습된 모델 파일
    """
    model_name = 'DAGMM'
    default_params = {
            'input_dim':  None, # <- set automatic
            'encode_layer' : [32,16,8],
            'decode_layer' : [8,16,32],            
            'zc_dim': 1,
            'n_gmm': 3, 
            'dropout': 0.5, 
            'lambda1': 0.001,
            'lambda2': 0.005, 
            'epochs': 5, 
            'lr': 1e-3, 
            'batch_size': 64, 
            'train_iter': 100,
            'val_iter': 50,
            'es_patience': 2,
            'ratio': 0.8,
            'save_path': './models/',
            'result_path': './results/',
            'tunning_steps': 100
        }


    def __init__(self, params=default_params):
        """
        DAGMM 객체 생성

        :param params: DAGMM 하이퍼파라미터 
        """  
        self.params = params
        self.threshold = None
        self.model_file = None
        self.model = None
        self.threshold = None
        self.tst_e = 0

        os.makedirs('./models/', exist_ok=True)
        os.makedirs('./results/', exist_ok=True)
        
        print('\n------Need GPU for training------\n')


    def fit(self, X):
        """
        모델 학습
        
        :param X: 피쳐 셋(정상데이터만 입력, 라벨 제외)
        :param early_stop: 학습 중단 여부
        """ 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_dataloader, valid_dataloader = get_dataloader(trn=X, mode='train', params=self.params)

        self.params['input_dim'] = X.shape[1]
        
        ## eval에서 사용
        self.X_trn = X
        
        self.model = BuildModel(params=self.params)
        self.model = self.model.to(device)

        optim = torch.optim.Adam(self.model.parameters(), self.params['lr'], amsgrad=True)
        scheduler = MultiStepLR(optim, [5, 8], 0.1)

        total_loss, total_recon_error, total_e, total_p = 0, 0, 0, 0
        loss_plot, val_loss_plot = [], []
        early_stop = EarlyStop(params=self.params)


        for epoch in range(self.params['epochs']):
            for step, (input_data, _) in enumerate(tqdm(train_dataloader, desc=f'epcoch{epoch+1}')):
                input_data = input_data.to(device)
                self.model.train()
                optim.zero_grad()
                input_data = input_data.squeeze(1)
                encoder, decoder, z, gamma = self.model(input_data)
                input_data, decoder, z, gamma = input_data.cpu(), decoder.cpu(), z.cpu(), gamma.cpu()
                loss, recon_error, e, p = self.model.loss_func(input_data, decoder, gamma, z)
        
                total_loss += loss.item()
                total_recon_error += recon_error.item()
                total_e += e.item()
                total_p += p.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optim.step()
                
                if (step+1) % self.params['train_iter'] == 0:

                    log = '\nlr {}, loss {:.2f}, recon_error {:.2f}, energy {:.2f}, p_diag {:.2f}'.format(optim.param_groups[0]['lr'], 
                                                                                                    total_loss/self.params['train_iter'], 
                                                                                                    total_recon_error/self.params['train_iter'], 
                                                                                                    total_e/self.params['train_iter'], 
                                                                                                    total_p/self.params['train_iter'])

                    IPython.display.clear_output()
                    print(log)
                    loss_plot.append(total_loss/self.params['train_iter'])
                    self.train_loss_plot(loss_plot)
                total_loss, total_recon_error, total_e, total_p = 0, 0, 0, 0

            with torch.no_grad():
                print("----------------------Validation----------------------\n")
                val_total_loss, val_total_recon_error, val_total_e, val_total_p = 0, 0, 0, 0

                self.model.eval()
                for step, (input_data,_) in enumerate(tqdm(valid_dataloader, desc="Val")):
                    input_data = input_data.to(device)
                    input_data = input_data.squeeze(1)
                    encoder,decoder,z,gamma = self.model(input_data)
                    input_data,decoder,z,gamma = input_data.cpu(),decoder.cpu(),z.cpu(),gamma.cpu()
                    loss, recon_error, e, p = self.model.loss_func(input_data, decoder, gamma, z)

                    val_total_loss += loss.item()
                    val_total_recon_error += recon_error.item()
                    val_total_e += e.item()
                    val_total_p += p.item()
                    
                    if (step+1) % self.params['val_iter'] == 0:
                        log = '\n val_loss {:.2f}, recon_error {:.2f}, energy {:.2f}, p_diag {:.2f}'.format(
                                                                            val_total_loss/self.params['val_iter'], 
                                                                            val_total_recon_error/self.params['val_iter'], 
                                                                            val_total_e/self.params['val_iter'], 
                                                                            val_total_p/self.params['val_iter']) 

                        IPython.display.clear_output()
                        print(log)
                        val_loss_plot.append(val_total_loss/self.params['val_iter'])
                        self.validation_loss_plot(val_loss_plot)
                    val_total_loss, val_total_recon_error, val_total_e, val_total_p = 0, 0, 0, 0
                    

                if (early_stop(val_total_loss/len(valid_dataloader), self.model, optim, epoch)):
                    self.train_loss_plot(loss_plot)
                    print('\n-------------Train Finished-------------')
                    exit(0)
                    break
    
            scheduler.step() 
            self.train_loss_plot(loss_plot)
        print('\n-------------Train Finished-------------')


    def decise_anomaly_score(self, X):
        """
        Anomaly Score 산출
        
        :param X: 피쳐 셋
        """    
        train_dataloader, test_dataloader = get_dataloader(trn=self.X_trn, tst=X, mode='eval', params=self.params)

        self.params['input_dim'] = X.shape[1]

        if self.model==None:
            self.model = self.model
        else:
            print('------need to train or load_model------\n')
        
        device = torch.device('cpu')
        self.model = self.model.to(device)
    
        self.model.eval()
        energy = []

        sum_prob, sum_mean, sum_cov = 0, 0, 0
        data_size = 0

        with torch.no_grad():
            for input_data,_ in tqdm(train_dataloader, desc='caluate gaussian'):
                input_data = input_data.squeeze(1)
                _ ,_, z, gamma = self.model(input_data)
                m_prob, m_mean, m_cov = self.model.get_gmm_param(gamma, z)
                sum_prob += m_prob
                sum_mean += m_mean * m_prob.unsqueeze(1)
                sum_cov += m_cov * m_prob.unsqueeze(1).unsqueeze(1)
                
                data_size += input_data.shape[0]
            
            train_prob = sum_prob / data_size
            train_mean = sum_mean / sum_prob.unsqueeze(1)
            train_cov = sum_cov / m_prob.unsqueeze(1).unsqueeze(1)

        
            for _, (x, _) in enumerate(tqdm(test_dataloader, desc='caluate energy')):
                x = x.squeeze(1)
                _ ,_ ,z,gamma = self.model(x)
                
                for i in range(z.shape[0]):
                    zi = z[i].unsqueeze(1)
                    sample_energy = self.model.sample_energy(train_prob, train_mean, train_cov, zi, gamma.shape[1], gamma.shape[0])
                    se = sample_energy.detach().item()
                    energy.append(se)

        energy_df = pd.DataFrame(energy)
        energy_df = energy_df.replace(np.inf, max(energy_df))
        energy_df = energy_df.rename(columns= {0:'anomaly_score'})
        energy_df.to_csv(self.params['result_path']+f'/test_energy.csv', index=False)

        print('test_energy_df is saved!')

        return energy_df


    def predict(self, X):
        """
        추론, 예측
        :param X: X_test(정상 + 이상)데이터
        """        
        tst_e = self.decise_anomaly_score(X)

        print('test_dataset reconstruction error mean : ', tst_e['anomaly_score'].mean())
        print('test_dataset reconstruction error std : ', tst_e['anomaly_score'].std())

        self.threshold = tst_e['anomaly_score'].mean()

        pred = [1 if e > self.threshold else 0 for e in tst_e['anomaly_score']]
        self.tst_e = tst_e

        return pred


    def evaluate(self, y_true, pred):
        """
        모델 성능 평가 출력
        
        :param y_true: 실제 값
        :param y_pred: 추론 값
        """
        confusion = confusion_matrix(y_true, pred, labels=[1,0])
        acc = accuracy_score(y_true, pred)
        pre = precision_score(y_true, pred, labels=[1,0])
        re = recall_score(y_true, pred, labels=[1,0])
        f1 = f1_score(y_true, pred, labels=[1,0])

        print('==> confusion matrix')
        print(confusion)
        print('='*30)

        print('Accuracy: {0:.4f}, Precision: {1:.4f}, Recall: {0:.4f}, F1: {1:.4f}'.format(acc, pre, re, f1))
    
        if len(set(y_true)) == 2:
            self._draw_roc_curve(y_true, pred)
        
        return self


    def train_loss_plot(self, losses):
        """
        train_loss plot 시각화
        """        
        plt.clf()
        _, ax = plt.subplots(figsize=(10,5), dpi=80)
        ax.plot(losses, 'blue', label='train', linewidth=1)
        ax.set_title('Loss change in training')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Iteration')
        ax.legend(loc='upper right')

        path = self.params['result_path']
        n_gmm = self.params['n_gmm']
        plt.show()
        plt.savefig(path+f'n_gmm{n_gmm}_train.png')


    def validation_loss_plot(self, losses):
        """
        validation_loss plot 시각화
        """          
        plt.clf()
        _, ax = plt.subplots(figsize=(10,5), dpi=80)
        ax.plot(losses, 'blue', label='val', linewidth=1)
        ax.set_title('Loss change in validation')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Iteration')
        ax.legend(loc='upper right')

        path = self.params['result_path']
        n_gmm = self.params['n_gmm']
        plt.show()        
        plt.savefig(path+f'n_gmm{n_gmm}_val.png')    
        

    def _draw_roc_curve(self, y_test, pred):
        """
        모델 성능 평가 ROC
        
        :param y_true: 실제 값
        :param y_pred: 추론 값
        """
        plt.figure(figsize=(10,10))

        fpr, tpr, thresholds = roc_curve(y_test, pred)
        plt.plot(fpr, tpr, label=str(self.model).split('(')[0])

        plt.plot([0,1], [0,1], 'k--', label='random quess')
        plt.title('ROC')
        plt.xlabel('X-FPR')
        plt.ylabel('Y-TPR')
        plt.legend()
        plt.grid()
        plt.show()    
        
        return self


    def get_params(self):  
        """
        모델 하이퍼파라미터 리턴
        
        :return: 하이퍼파라미터 dict
        """      
        return self.params
    
    
    def set_params(self, params):
        """
        모델 하이퍼파라미터 셋업
        
        :param params: 하이퍼파라미터 dict
        """
        for key, value in params.items():
            self.params[key] = value
        
        return self.get_params()        


    def set_threshold(self, threshold):
        """
        threshold 지정
        """
        self.threshold = threshold
        
        return self.threshold
    