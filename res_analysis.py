import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def str2bool(v):
    return v.lower() in ('true')

def get_res(config):
    temp = []
    datasets = ["SMD","SMAP","MSL","PSM","SWAT"]
    for d in datasets:
        config.dataset = d
        filename = f"res/{config.dataset}_epoch{config.num_epochs}_encodertype_{config.encoder_type}_encoderlayernum{config.encoder_layer_num}_layerclusternum{config.layer_cluster_num}_useclusterloss{config.use_cluster_loss}_trainuseclusterloss{config.train_use_cluster_loss}_latentdistanceselecttype{config.latent_distance_select_type}_lambda1{config.lambda1}_latentdistanceselecttopk{config.latent_distance_select_top_k}_latentdistanceselectspecifick{config.latent_distance_select_specific_k}_temperature{config.temperature}_runtimes{config.run_times}.txt"
        with open(filename, "r") as f:
            res = json.load(f)
            temp.append(res['f1'])
            temp.append(res['rc'])
            temp.append(res['pc'])
    return np.array(temp).reshape(len(datasets),3)

def get_all_data_for_encoder_type_layer_cluster(config):
    # types = ['Temporal', 'Spatio', 'SpatioTemporal']
    # types = ['Temporal']
    types = ['SpatioTemporal']
    layer_num = [1,2,3,4,5]
    # cluster_num = [1]
    temp = []
    for t in types:
        config.encoder_type = t
        for l in layer_num:
            config.encoder_layer_num = l
            config.layer_cluster_num = l + 1  # cluster数量与encoder layer数量相同
            res = get_res(config)
            temp.append(res)
            # for c in range(1,l+2,1): # cluster数量与encoder layer数量不相同
            #     config.layer_cluster_num = c
    return np.array(temp).reshape(len(types),len(layer_num),5,3)

def plot_for_encoder_type_layer_cluster(config):
    data = get_all_data_for_encoder_type_layer_cluster(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) # todo c
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) # todo c
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel("spatio-temporal_encoder_layer_num")
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/spatio-temporal_encoder_layer_num_l+1.pdf")

def get_all_data_for_encoder_type_fix_layer_diff_cluster(config):
    # types = ['Temporal', 'Spatio', 'SpatioTemporal']
    types = ['Temporal']
    # types = ['Spatio']
    layer_num = [4]
    # cluster_num = [1]
    temp = []
    for t in types:
        config.encoder_type = t
        for l in layer_num:
            config.encoder_layer_num = l
            # config.layer_cluster_num = l + 1  # cluster数量与encoder layer数量相同
            # res = get_res(config)
            # temp.append(res)
            for c in range(1,l+2,1): # cluster数量与encoder layer数量不相同
                config.layer_cluster_num = c
                res = get_res(config)
                temp.append(res)
    return np.array(temp).reshape(len(types),layer_num[0]+1,5,3)

def plot_for_encoder_type_fix_layer_diff_cluster(config):
    data = get_all_data_for_encoder_type_fix_layer_diff_cluster(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) # todo c
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) # todo c
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel("spatio-temporal_cluster_layer_num")
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/spatio-temporal_cluster_layer_num_5.pdf")

def get_all_data_for_encoder_type_diff_layer_fix_cluster(config):
    # types = ['Temporal', 'Spatio', 'SpatioTemporal']
    # types = ['Temporal']
    types = ['SpatioTemporal']
    layer_num = [1,2,3,4,5]
    cluster_num = [1]
    temp = []
    for t in types:
        config.encoder_type = t
        for l in layer_num:
            config.encoder_layer_num = l
            for c in cluster_num: # cluster数量与encoder layer数量不相同
                config.layer_cluster_num = c
                res = get_res(config)
                temp.append(res)
    return np.array(temp).reshape(len(types),len(layer_num),5,3)

def plot_for_encoder_type_diff_layer_fix_cluster(config):
    data = get_all_data_for_encoder_type_diff_layer_fix_cluster(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) # todo c
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) # todo c
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel("spatio-temporal_encoder_layer_num")
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/spatio-temporal_encoder_layer_num_1.pdf")

def get_all_data_for_diff_layer(config):
    layer_num = [1,2,3,4,5]
    temp = []
    config.latent_distance_select_type = "fusion"
    for l in layer_num:
        config.encoder_layer_num = l
        config.layer_cluster_num = l+1
        res = get_res(config)
        temp.append(res)
    return np.array(temp).reshape(1,len(layer_num),5,3)

def plot_for_diff_layer(config):
    data = get_all_data_for_diff_layer(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) # todo c
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) # todo c
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel("fusion_diff_layer")
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/fusion_diff_layer.pdf")

def get_all_data_for_diff_layer_base(config):
    layer_num = [1,2,3,4,5]
    temp = []
    for l in layer_num:
        config.encoder_layer_num = l
        config.layer_cluster_num = l+1
        res = get_res(config)
        temp.append(res)
    return np.array(temp).reshape(1,len(layer_num),5,3)

def plot_for_diff_layer_base(config):
    config.train_use_cluster_loss = False
    config.use_cluster_loss = False
    data = get_all_data_for_diff_layer_base(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) 
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) 
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel("diff_encoder_layer_base")
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/diff_encoder_layer_base.pdf")

def get_all_data_for_specific_latent_layer_distance(config):
    layer_num = [1]
    temp = []
    specific_layer = [1,2]
    for l in layer_num:
        config.encoder_layer_num = l
        config.layer_cluster_num = l+1
        for s in specific_layer:
            config.latent_distance_select_specific_k = s
            res = get_res(config)
            temp.append(res)
    return np.array(temp).reshape(1,len(specific_layer),5,3)

def plot_for_specific_latent_layer_distance(config):
    config.train_use_cluster_loss = False
    config.use_cluster_loss = True
    config.latent_distance_select_type = "specific"
    config.temperature = 0.05
    config.lambda1 = 1.0
    encoder_layer = 1
    data = get_all_data_for_specific_latent_layer_distance(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) 
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) 
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel("specific_latent_layer_distance")
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/specific_latent_layer_distance_{encoder_layer}_lambda1.0.pdf")

def get_all_data_for_latent_layer_select_type(config):
    layer_num = [2]
    temp = []
    fusion_types = ['min','max','mean','fusion','specific']
    # specific_layer = [1,2,3,4,5]
    for l in layer_num:
        config.encoder_layer_num = l
        config.layer_cluster_num = l+1
        for f in fusion_types:
            config.latent_distance_select_type = f
            res = get_res(config)
            temp.append(res)
    return np.array(temp).reshape(1,len(fusion_types),5,3)

def plot_for_latent_layer_select_type(config):
    config.train_use_cluster_loss = False
    config.use_cluster_loss = True
    data = get_all_data_for_latent_layer_select_type(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) 
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) 
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel("latent_layer_select_type")
    plt.xticks(x,['min','max','mean','fusion','specific'])
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/latent_layer_select_type_2_lambda1.0.pdf")

def get_all_data_for_diff_lambda1(config):
    layer_num = [1]
    temp = []
    # lambdas = [0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1.0,1.2,1.5,1.7,2.0]
    lambdas = [0.0,0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07,0.1,0.3,0.5]
    # lambdas = [0.01,0.03,0.05,0.07,0.1]
    # lambdas = [0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1.0]
    # specific_layer = [1,2,3,4,5]
    for l in layer_num:
        config.encoder_layer_num = l
        config.layer_cluster_num = l+1
        for f in lambdas:
            config.lambda1 = f
            res = get_res(config)
            temp.append(res)
    return np.array(temp).reshape(1,len(lambdas),5,3)

def plot_for_diff_lambda1(config):
    config.train_use_cluster_loss = True
    config.use_cluster_loss = True
    data = get_all_data_for_diff_lambda1(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    # x = [i for i in range(1,len(smd_res)+1,1)]
    # x = [0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1.0]
    # x = [0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1.0,1.2,1.5,1.7,2.0]
    x = [0.0,0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07,0.1,0.3,0.5]
    # x = [0.01,0.03,0.05,0.07,0.1]
    plt.plot(x,smd_res,"p-c",label="SMD",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="SMAP",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="MSL",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="PSM",alpha=0.7) 
    plt.plot(x,swat_res,"h:y",label="SWaT",alpha=0.7) 
    plt.plot(x,avg_res,"s-.r",label="Average",alpha=0.7)
    print(avg_res)
    plt.grid()
    plt.xlabel("lambda")
    # plt.xticks(x,['min','max','mean','fusion','specific'])
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.46, 1.05),ncol=6)
    
    plt.savefig(f"analysis/diff_lambda1_1.pdf")

def get_all_data_for_ablation(config):
    layer_num = [1]
    temp = []
    # lambdas = [0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1.0]
    # lambdas = [0.01,0.03,0.05,0.07,0.1]
    # specific_layer = [1,2,3,4,5]
    # config.train_use_cluster_loss = True
    # config.use_cluster_loss = True
    for l in layer_num:
        config.encoder_layer_num = l
        config.layer_cluster_num = l+1
        for t in [True,False]:
            for u in [True,False]:
                config.train_use_cluster_loss = t
                config.use_cluster_loss = u
                res = get_res(config)
                temp.append(res)
    return np.array(temp).reshape(1,-1,5,3)

def plot_for_ablation(config):
    data = get_all_data_for_ablation(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    # x = [0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1.0]
    # x = [0.01,0.03,0.05,0.07,0.1]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) 
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) 
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel("ablation")
    print(avg_res)
    # plt.xticks(x,['min','max','mean','fusion','specific'])
    plt.xticks(x,['tu','t-','-u','--'])
    # plt.xticks(x,['tu','-u'])
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/ablation_1_lambda1.0.pdf")

def get_all_data_for_specific_latent_layer_diff_encoder_layer(config):
    # types = ['Temporal', 'Spatio', 'SpatioTemporal']
    # types = ['Temporal']
    types = ['Temporal']
    # layer_num = [1,2,3]
    layer_num = [1,2,3,4,5]
    # cluster_num = [1]
    temp = []
    for t in types:
        config.encoder_type = t
        for l in layer_num:
            config.encoder_layer_num = l
            config.layer_cluster_num = l+1
            res = get_res(config)
            temp.append(res)
    return np.array(temp).reshape(len(types),len(layer_num),5,3)

def plot_for_specific_latent_layer_diff_encoder_layer(config):
    config.latent_distance_select_specific_k = 1
    config.latent_distance_select_type = "specific" 
    # config.latent_distance_select_type = "fusion" 
    data = get_all_data_for_specific_latent_layer_diff_encoder_layer(config)
    temporal_f1_data = data[0,:,:,0].T
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) # todo c
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) # todo c
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel(f"{config.latent_distance_select_type}_latent_layer_diff_encoder_layer") 
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/{config.latent_distance_select_type}_latent_layer_diff_encoder_layer.pdf")

def get_all_data_for_diff_temperature(config):
    layer_num = [1]
    temp = []
    # temperatures = [0.0001,0.0003,0.0005,0.0007,0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1.0]
    temperatures = [0.0001,0.0003,0.0005,0.0007,0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07,0.1]
    # lambdas = [0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1.0]
    # lambdas = [0.01,0.03,0.05,0.07,0.1]
    # specific_layer = [1,2,3,4,5]
    # config.train_use_cluster_loss = True
    # config.use_cluster_loss = True
    for l in layer_num:
        config.encoder_layer_num = l
        config.layer_cluster_num = l+1
        for t in [True,False]:
            for t2 in temperatures:
                config.train_use_cluster_loss = t
                config.use_cluster_loss = True
                config.temperature = t2
                res = get_res(config)
                temp.append(res)
    return np.array(temp).reshape(-1,2,5,3).transpose(1,0,2,3)

def plot_for_diff_temperature(config):
    data = get_all_data_for_diff_temperature(config)
    with_cluster_train_data = data[0,:,:,0].T
    without_cluster_train_data = data[1,:,:,0].T
    # temporal_f1_data = with_cluster_train_data
    temporal_f1_data = without_cluster_train_data
    smd_res = temporal_f1_data[0]
    smap_res = temporal_f1_data[1]
    msl_res = temporal_f1_data[2]
    psm_res = temporal_f1_data[3]
    swat_res = temporal_f1_data[4]
    avg_res = temporal_f1_data.mean(0)
    
    

    plt.cla()
    x = [i for i in range(1,len(smd_res)+1,1)]
    # x = [0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1.0]
    # x = [0.01,0.03,0.05,0.07,0.1]
    x = [0.0001,0.0003,0.0005,0.0007,0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07,0.1]
    plt.plot(x,smd_res,"p-c",label="smd",alpha=0.7)
    plt.plot(x,smap_res,"^:g",label="smap",alpha=0.7)
    plt.plot(x,msl_res,"P--b",label="msl",alpha=0.7)
    plt.plot(x,psm_res,"D-m",label="psm",alpha=0.7) 
    plt.plot(x,swat_res,"h:y",label="swat",alpha=0.7) 
    plt.plot(x,avg_res,"s-.r",label="avg",alpha=0.7)
    plt.grid()
    plt.xlabel("temperature")
    print(avg_res)
    # plt.xticks(x,['min','max','mean','fusion','specific'])
    # plt.xticks(x,['tu','t-','-u','--'])
    # plt.xticks(x,['tu','-u'])
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"analysis/temperature_trainuseclusterlossFalse_encoderlayer1.pdf")

def get_all_data_for_rq2(config):
    layer_num = [1,2,3,4,5]
    ans = []
    for l in layer_num:
        config.encoder_layer_num = l
        config.layer_cluster_num = l+1
        specific_layer = [i+1 for i in range(l+1)]
        temp = []
        for s in specific_layer:
            config.latent_distance_select_specific_k = s
            res = get_res(config)
            temp.append(res)
        temp = np.array(temp).reshape(len(specific_layer),5,3)[:,:,0]
        ans.append(temp)
    return ans

def plot_for_rq2(config):
    config.train_use_cluster_loss = False
    config.use_cluster_loss = True
    config.latent_distance_select_type = "specific"
    config.temperature = 0.05
    config.lambda1 = 0.1
    data = get_all_data_for_rq2(config) # encoder_layer_num, specific_layer_num, dataset_num
    encoder_layer_1 = data[0].mean(-1)
    encoder_layer_2 = data[1].mean(-1)
    encoder_layer_3 = data[2].mean(-1)
    encoder_layer_4 = data[3].mean(-1)
    encoder_layer_5 = data[4].mean(-1)

    plt.cla()
    x = [i for i in range(len(encoder_layer_5))]
    plt.plot(range(len(encoder_layer_1)),encoder_layer_1,"p-c",label="m=1",alpha=0.7)
    plt.plot(range(len(encoder_layer_2)),encoder_layer_2,"^:g",label="m=2",alpha=0.7)
    plt.plot(range(len(encoder_layer_3)),encoder_layer_3,"P--b",label="m=3",alpha=0.7)
    plt.plot(range(len(encoder_layer_4)),encoder_layer_4,"D-m",label="m=4",alpha=0.7) 
    plt.plot(range(len(encoder_layer_5)),encoder_layer_5,"h:y",label="m=5",alpha=0.7) 
    plt.grid()
    plt.xlabel("Layers")
    plt.ylabel("F1-score")
    X = ["EBL","ECL 1","ECL 2","ECL 3","ECL 4","ECL 5"]
    plt.xticks(x,X)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=5)
    
    plt.savefig(f"analysis/rq2.pdf")

def get_all_data_for_rq4_1(config):
    layer_num = [1,2,3,4,5]
    temp = []
    fusion_types = ['min','max','mean','fusion','specific']
    for f in fusion_types:
        config.latent_distance_select_type = f
        for l in layer_num:
            config.encoder_layer_num = l
            config.layer_cluster_num = l+1
            res = get_res(config)
            temp.append(res)
    return np.array(temp).reshape(len(fusion_types),len(layer_num),5,3)[:,:,:,0]

def get_all_data_for_rq4_0(config):
    layer_num = [1,2,3,4,5]
    temp = []
    fusion_types = ['fusion']
    for f in fusion_types:
        config.latent_distance_select_type = f
        for l in layer_num:
            config.encoder_layer_num = l
            config.layer_cluster_num = l+1
            res = get_res(config)
            temp.append(res)
    return np.array(temp).reshape(len(fusion_types),len(layer_num),5,3)[:,:,:,0]

def plot_for_rq4(config):
    config.train_use_cluster_loss = False
    config.use_cluster_loss = True
    data = get_all_data_for_rq4_1(config)
    config.train_use_cluster_loss = False
    config.use_cluster_loss = False
    data0 = get_all_data_for_rq4_0(config)
    min_data = data[0].mean(-1)
    max_data = data[1].mean(-1)
    mean_data = data[2].mean(-1)
    fusion_data = data[3].mean(-1)
    specific_data = data[4].mean(-1)
    base_data = data0[0].mean(-1)
    

    plt.cla()
    x = [i for i in range(1,len(min_data)+1,1)]
    plt.plot(x,base_data,"p-c",label="base",alpha=0.7)
    plt.plot(x,min_data,"^:g",label="min",alpha=0.7)
    plt.plot(x,max_data,"P--b",label="max",alpha=0.7)
    plt.plot(x,mean_data,"D-m",label="mean",alpha=0.7) 
    plt.plot(x,specific_data,"h:y",label="specific",alpha=0.7) 
    plt.plot(x,fusion_data,"s-.r",label="corr",alpha=0.7)
    plt.grid()
    plt.xlabel("Encoder_layer_num m")
    # plt.xticks(x,['min','max','mean','fusion','specific'])
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.46, 1.05),ncol=6)
    
    plt.savefig(f"analysis/rq4.pdf")


def plot_time_efficiency(config):
    
    X = ["Training","Inference"]
    anomalyTrans = [21.06,4.63]
    MEMTO = [91.09,6.38]
    Our = [7.91,4.31]
    X_axis = np.arange(len(X)) 
    
    plt.figure(figsize=(6,4))
    plt.bar(X_axis - 0.3, anomalyTrans, 0.3, label = 'AnomalyTrans',hatch = '/',color="b",alpha=0.7) 
    plt.bar(X_axis, MEMTO, 0.3, label = 'MEMTO',hatch = '+',color="g",alpha=0.7) 
    plt.bar(X_axis + 0.3, Our, 0.3, label = 'Our',hatch = 'x',color="m",alpha=0.7) 
    
    plt.xticks(X_axis, X) 
    # plt.xlabel("Groups") 
    plt.ylabel("Time (s)") 
    # plt.title("Number of Students in each group") 
    plt.legend() 
    plt.savefig(f"analysis/time_efficiency.pdf")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--input_c', type=int, default=50)
    parser.add_argument('--output_c', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lambd',type=float, default=0.1)
    parser.add_argument('--pretrained_model', type=str, default=None)
    # parser.add_argument('--dataset', type=str, default='SMD')
    # parser.add_argument('--dataset', type=str, default='MSL')
    # parser.add_argument('--dataset', type=str, default='SMAP')
    # parser.add_argument('--dataset', type=str, default='PSM')
    parser.add_argument('--dataset', type=str, default='SWAT')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'memory_initial'])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_heads',type=int, default=8)
    # parser.add_argument('--data_path', type=str, default='../MEMTO/data/SMD/')
    # parser.add_argument('--data_path', type=str, default='../MEMTO/data/MSL/')
    # parser.add_argument('--data_path', type=str, default='../MEMTO/data/SMAP/')
    # parser.add_argument('--data_path', type=str, default='../MEMTO/data/PSM/')
    parser.add_argument('--data_path', type=str, default='../MEMTO/data/SWAT/A1_A2/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=0.5)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=128) # 未使用
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--temperature', type=int, default=0.05)
    parser.add_argument('--run_times',type=int, default=0, help='')
    parser.add_argument('--gpu',type=str, default="1", help='') 
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
    parser.add_argument('--use_cluster_loss',type=str2bool, default=False, help='')
    parser.add_argument('--use_recon_loss',type=str2bool, default=True, help='')
    
    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    config.layer_cluster_num = config.encoder_layer_num + 1

    # plot_for_encoder_type_layer_cluster(config)
    # plot_for_encoder_type_fix_layer_diff_cluster(config)
    # plot_for_encoder_type_diff_layer_fix_cluster(config)
    # plot_for_diff_layer(config)
    # plot_for_diff_layer_base(config)
    # plot_for_specific_latent_layer_distance(config)
    # plot_for_specific_latent_layer_diff_encoder_layer(config)
    # plot_for_latent_layer_select_type(config)
    plot_for_diff_lambda1(config)
    # plot_for_ablation(config)
    # plot_for_diff_temperature(config)
    # plot_for_rq2(config)
    # plot_for_rq4(config)
    # plot_time_efficiency(config)

    # res = get_res(config)
    # print(res)
    # print(res[:,0].mean())
    

