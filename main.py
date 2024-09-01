import os
import argparse
import torch
from torch.backends import cudnn
from utils.utils import *
from solver import Solver
import torch
import random
import json

# 固定随机种子
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
seed_everything()

def str2bool(v):
    return v.lower() in ('true')

def save(best_res,config):
    filename = f"res/{config.dataset}_epoch{config.num_epochs}_encodertype_{config.encoder_type}_encoderlayernum{config.encoder_layer_num}_layerclusternum{config.layer_cluster_num}_useclusterloss{config.use_cluster_loss}_trainuseclusterloss{config.train_use_cluster_loss}_latentdistanceselecttype{config.latent_distance_select_type}_lambda1{config.lambda1}_latentdistanceselecttopk{config.latent_distance_select_top_k}_latentdistanceselectspecifick{config.latent_distance_select_specific_k}_temperature{config.temperature}_runtimes{config.run_times}.txt"
    # filename = f"res/res.txt"
    with open(filename, "w") as fw:
        json.dump(best_res, fw)
        
def get_res(config):
    filename = f"res/{config.dataset}_epoch{config.num_epochs}_encodertype_{config.encoder_type}_encoderlayernum{config.encoder_layer_num}_layerclusternum{config.layer_cluster_num}_useclusterloss{config.use_cluster_loss}_trainuseclusterloss{config.train_use_cluster_loss}_latentdistanceselecttype{config.latent_distance_select_type}_lambda1{config.lambda1}_latentdistanceselecttopk{config.latent_distance_select_top_k}_latentdistanceselectspecifick{config.latent_distance_select_specific_k}_temperature{config.temperature}_runtimes{config.run_times}.txt"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            res = json.load(f)
    else:
        res = None
    return res
        

def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))
    best_res = {}
    if config.mode == 'train':
        best_res = solver.train(training_type='first_train')
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'memory_initial':
        solver.get_memory_initial_embedding(training_type='second_train')

    save(best_res,config)
    return solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lambd',type=float, default=0.1)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SMD')
    # parser.add_argument('--dataset', type=str, default='MSL')
    # parser.add_argument('--dataset', type=str, default='SMAP')
    # parser.add_argument('--dataset', type=str, default='PSM')
    # parser.add_argument('--dataset', type=str, default='SWAT')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'memory_initial'])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_heads',type=int, default=8)
    parser.add_argument('--data_path', type=str, default='dataset/SMD/')
    # parser.add_argument('--data_path', type=str, default='dataset/MSL/')
    # parser.add_argument('--data_path', type=str, default='dataset/SMAP/')
    # parser.add_argument('--data_path', type=str, default='dataset/PSM/')
    # parser.add_argument('--data_path', type=str, default='dataset/SWAT/A1_A2/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=0.5)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=128) # 未使用
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--run_times',type=int, default=0, help='')
    parser.add_argument('--gpu',type=str, default="2", help='') 
    # [False,True]
    # [None,second_train]

    # stacked transformer ad
    parser.add_argument('--encoder_type',type=str,default="Temporal",help="", choices=['Temporal', 'Spatio', 'SpatioTemporal'])
    parser.add_argument('--encoder_layer_num',type=int,default=1,help="")
    # parser.add_argument('--layer_cluster_num',type=int,default=2,help="encoder_layer_num + embedding layer") # 使用几个隐层距离进行异常检测效果最好？
    # todo: 探究几个隐层距离如何融合？目前是相加的方式。其他可能的方式，相乘？还有别的吗？相关性选择！
    parser.add_argument('--latent_distance_select_type',type=str,default="fusion",choices=['max','min','fusion','mean','specific','topk'])
    parser.add_argument('--latent_distance_select_top_k',type=int,default=1)
    parser.add_argument('--latent_distance_select_specific_k',type=int,default=1)
    # todo: 思考算相关性时，是否考虑不同维度的值，而非所有维度的均值
    # 可以用相关性融合或者选择，都可以
    # train
    parser.add_argument('--train_use_cluster_loss',type=str2bool, default=True, help='')
    parser.add_argument('--lambda1',type=float,default=0.1)
    # test
    # todo: 探究隐层距离和重构损失如何融合？目前是softmax相乘的方式。其他可能的方式，直接相乘、相加，还有别的吗？
    parser.add_argument('--use_cluster_loss',type=str2bool, default=True, help='')
    parser.add_argument('--use_recon_loss',type=str2bool, default=True, help='')
    
    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    config.layer_cluster_num = config.encoder_layer_num + 1
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    res = get_res(config)
    if res is None:
        main(config)
    else:
        print(f"exist result:{res}")
