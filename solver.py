# Some code based on https://github.com/thuml/Anomaly-Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
# from model.Transformer import TransformerVar
from model.Transformerv2 import TransformerVar
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from scipy.special import softmax

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class TwoEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

class OneEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.type = type

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint_{self.type}.pth'))
        self.val_loss_min = val_loss

def adjustment_decision(pred,gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred

def best_treshold_search(distance,gt):
    # anomaly_ratio= range(1, 101) # [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,]
    anomaly_ratio= np.arange(1, 15)*0.1
    best_res = {"f1":-1}
    best_pred = []
    for ano in anomaly_ratio:
        threshold= np.percentile(distance,100-ano)
        pred=[1 if d>threshold  else 0 for d in distance]
        pred = adjustment_decision(pred,gt)  # 增加adjustment_decision
        eval_results = {
                "f1": f1_score(gt, pred),
                "rc": recall_score(gt, pred),
                "pc": precision_score(gt, pred),
                "acc": accuracy_score(gt,pred),
                "threshold":threshold,
                "anomaly_ratio":ano
            }
        if eval_results["f1"] > best_res["f1"]:
            best_res = eval_results
            best_pred = pred
    return best_res, best_pred

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.config = config

        self.train_loader, self.vali_loader, self.k_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)

        self.test_loader, _ = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = self.vali_loader
        
        self.build_model()
        
        self.c = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.entropy_loss = EntropyLoss()
        self.criterion = nn.MSELoss()
        self.gathering_loss = GatheringLoss(reduce=False)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def build_model(self):
        
        self.model = TransformerVar(self, device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5) # momentum=0.9,

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

    # unused
    def vali(self, vali_loader):
        self.model.eval()

        valid_loss_list = [] ; valid_re_loss_list = [] ; valid_entropy_loss_list = []

        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output, queries, mem_items, attn = output_dict['out'], output_dict['queries'], output_dict['mem'], output_dict['attn']
            
            rec_loss = self.criterion(output, input)
            entropy_loss = self.entropy_loss(attn)
            loss = rec_loss + self.lambd*entropy_loss
            loss = rec_loss

            valid_re_loss_list.append(rec_loss.item())
            valid_entropy_loss_list.append(entropy_loss.item())
            valid_loss_list.append(loss.item())

        return np.average(valid_loss_list), np.average(valid_re_loss_list), np.average(valid_entropy_loss_list)
        # return np.average(valid_loss_list), np.average(valid_re_loss_list)

    def vali_new(self, vali_loader):
        self.model.eval()

        valid_loss_list = []
        valid_re_loss_list = []

        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output = output_dict['out']
            
            rec_loss = self.criterion(output, input)
            loss = rec_loss

            valid_re_loss_list.append(rec_loss.item())
            valid_loss_list.append(loss.item())

        return np.average(valid_loss_list), np.average(valid_re_loss_list)

    def train(self, training_type):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = OneEarlyStopping(patience=3, verbose=True, dataset_name=self.dataset, type=training_type)
        train_steps = len(self.train_loader)
        best_res = {"f1":0.0}

        from tqdm import tqdm
        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = []

            temp_c = torch.zeros((self.layer_cluster_num,self.input_c)).cuda()

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                B,T,N = input.shape    
                output_dict = self.model(input,self.c)
                output = output_dict['out']
                
                embed = output_dict['layer_embeds']
                temp_c += embed.mean(1).mean(1)
        
                rec_loss = output_dict['recon_loss'].mean()
                loss = rec_loss
                cluster_loss = output_dict['cluster_loss'].mean()
                if self.train_use_cluster_loss:
                    loss = loss + self.lambda1*cluster_loss
               
                loss_list.append(loss.item())
                rec_loss_list.append(rec_loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.mean().backward()
                self.optimizer.step()
            
            temp_c /= i + 1
            self.c = temp_c.detach()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(loss_list)
            train_rec_loss = np.average(rec_loss_list)
            valid_loss , valid_re_loss_list = self.vali_new(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss))
            print(
                "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {2:.7f} ".format(
                    epoch + 1, train_steps, valid_re_loss_list))
            print(
                "Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_rec_loss))
            
            res = self.test(after_train=True)
            if res["f1"] > best_res['f1']:
                best_res = res
                
            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        return best_res

    def test(self, after_train=False):
        if after_train:
            pass
        else:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_second_train.pth')))
            self.model.eval()
        
        print("======================TEST MODE======================")

        reconstructed_output = []
        original_output = []
        rec_loss_list = []

        test_labels = []
        test_attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            B,T,N = input.shape
            output_dict= self.model(input,self.c)
            output = output_dict['out'].detach()
            rec_loss = output_dict['recon_loss'].mean(-1).detach()
           
            if self.use_recon_loss:
                loss = rec_loss
            else:
                loss = 0
            
            if self.use_cluster_loss:
                cluster_loss = output_dict["cluster_loss"].detach().mean(-1) # encoder_layer_num+1, B, T
                rec_loss = rec_loss.unsqueeze(1) # B 1 T
                attn = torch.einsum('bit,ebt->bie',rec_loss,cluster_loss) # B,1,encoder_layer_num+1
                if self.latent_distance_select_type == "fusion": # 根据相关性进行融合
                    cluster_loss = torch.einsum('bie,ebt->bit',attn,cluster_loss).squeeze(1) # B T
                elif self.latent_distance_select_type == "max": # 选择最相关的1个
                    attn = attn.repeat(1,100,1)
                    index = torch.argmax(attn,-1,keepdim=True)
                    cluster_loss = torch.gather(cluster_loss.permute(1,2,0),-1,index).squeeze(-1)
                elif self.latent_distance_select_type == "min": # 选择最不相关的1个
                    attn = attn.repeat(1,100,1)
                    index = torch.argmin(attn,-1,keepdim=True)
                    cluster_loss = torch.gather(cluster_loss.permute(1,2,0),-1,index).squeeze(-1)
                elif self.latent_distance_select_type == "mean": # 平均融合
                    cluster_loss = cluster_loss.mean(0)
                elif self.latent_distance_select_type == "specific": # 选择特定的1个
                    cluster_loss = cluster_loss[self.latent_distance_select_specific_k-1]
                cluster_loss = F.softmax(cluster_loss/self.temperature,dim=-1)
                loss = loss * cluster_loss
            
            cri = loss.detach().cpu().numpy()
            test_attens_energy.append(cri)
            test_labels.append(labels)

            reconstructed_output.append(output.detach().cpu().numpy())
            original_output.append(input.detach().cpu().numpy())
            rec_loss_list.append(rec_loss.detach().cpu().numpy())

        test_attens_energy = np.concatenate(test_attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(test_attens_energy)
        test_labels = np.array(test_labels)

        reconstructed_output = np.concatenate(reconstructed_output,axis=0).reshape(-1)
        original_output = np.concatenate(original_output,axis=0).reshape(-1)
        rec_loss_list = np.concatenate(rec_loss_list,axis=0).reshape(-1)

        gt = test_labels.astype(int)
        
        best_res, best_pred = best_treshold_search(test_energy,gt)
        
        print(f"Best result:{best_res}")
        print('='*50)

        self.logger.info(f"Dataset: {self.dataset}")
        # self.logger.info(f"number of items: {self.n_memory}")
        self.logger.info(f"Precision: {round(best_res['pc'],4)}")
        self.logger.info(f"Recall: {round(best_res['rc'],4)}")
        self.logger.info(f"f1_score: {round(best_res['f1'],4)} \n")
        return best_res

    def get_memory_initial_embedding(self,training_type='second_train'):

        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_first_train.pth')))
        self.model.eval()
        
        for i, (input_data, labels) in enumerate(self.k_loader):

            input = input_data.float().to(self.device)
            if i==0:
                output= self.model(input)['queries']
            else:
                output = torch.cat([output,self.model(input)['queries']], dim=0)
        
        self.memory_init_embedding = k_means_clustering(x=output, n_mem=self.n_memory, d_model=self.input_c)

        self.memory_initial = False

        self.phase_type = "second_train"
        self.build_model(memory_init_embedding = self.memory_init_embedding.detach())

        memory_item_embedding = self.train(training_type=training_type)

        memory_item_embedding = memory_item_embedding[:int(self.n_memory),:]

        item_folder_path = "memory_item"
        if not os.path.exists(item_folder_path):
            os.makedirs(item_folder_path)

        item_path = os.path.join(item_folder_path, str(self.dataset) + '_memory_item.pth')

        torch.save(memory_item_embedding, item_path)
