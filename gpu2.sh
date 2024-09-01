# # 基线：展示encoder层数对于异常检测的影响
# for l in 1 2 3 4 5 # encoder_layer_num 
# do 
#     # echo $l $((l+1)) $t
#     python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $l --latent_distance_select_type fusion --train_use_cluster_loss False --use_cluster_loss False
#     python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $l  --latent_distance_select_type fusion --train_use_cluster_loss False --use_cluster_loss False
#     python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $l  --latent_distance_select_type fusion --train_use_cluster_loss False --use_cluster_loss False
#     python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $l  --latent_distance_select_type fusion --train_use_cluster_loss False --use_cluster_loss False
#     python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $l --latent_distance_select_type fusion --train_use_cluster_loss False --use_cluster_loss False
# done

# # # 基线：展示选择特定隐层距离用于异常检测，encoder层数对于异常检测的影响
# for e in 1 2 3 # 4 5 # encoder_layer_num 
# do 
#     # echo $l $((l+1)) $t
#     python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True --temperature 0.1 --lambda1 0.1 --latent_distance_select_specific_k 1
#     python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True --temperature 0.1 --lambda1 0.1 --latent_distance_select_specific_k 1
#     python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True --temperature 0.1 --lambda1 0.1 --latent_distance_select_specific_k 1
#     python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True --temperature 0.1 --lambda1 0.1 --latent_distance_select_specific_k 1
#     python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True --temperature 0.1 --lambda1 0.1 --latent_distance_select_specific_k 1
# done

# 基线：展示在特定（或不同）encoder层数（4层）下，选择任意一个隐层距离用于异常检测的效果
# for l in 1 2 3 4 5 # latent_distance_select_specific_k 
# do 
#     python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_specific_k $l --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
# done

# # # 基线：展示在特定（或不同）encoder层数（2层）下，选择任意一个隐层距离用于异常检测的效果
# for l in 1 2 3 # latent_distance_select_specific_k 
# do 
#     python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_specific_k $l --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
# done

# # 基线：展示在特定（或不同）encoder层数（1层）下，选择任意一个隐层距离用于异常检测的效果
# for l in 1 2 # latent_distance_select_specific_k 
# do 
#     python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_specific_k $l --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_specific_k $l  --latent_distance_select_type specific --train_use_cluster_loss False --use_cluster_loss True
# done

# # 基线：展示在特定（或不同）encoder层数（4层）下，不同的隐层距离使用方式（mean、min、max、attn）对于异常检测的影响
# for l in max min fusion mean specific # latent_distance_select_type
# do 
#     python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4  --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4  --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4  --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
# done

# # # 基线：展示在特定（或不同）encoder层数（2层）下，不同的隐层距离使用方式（mean、min、max、attn）对于异常检测的影响
# for l in max min fusion mean specific # latent_distance_select_type
# do 
#     python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2  --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2  --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2  --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
# done

# 基线：展示在特定（或不同）encoder层数（1层）下，不同的隐层距离使用方式（mean、min、max、attn）对于异常检测的影响
# for l in max min fusion mean specific # latent_distance_select_type
# do 
#     python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
#     python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_type $l --train_use_cluster_loss False --use_cluster_loss True
# done

# # 参数敏感性：在特定（或不同）encoder层数（4层）下，探究不同lambda1对于异常检测的影响，即聚类学习与重构学习之间的权衡
# for l in 0.01 0.03 0.05 0.07 0.1 0.3 0.5 0.7 1.0
# do
#         python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
# done

# # # # 参数敏感性：在特定（或不同）encoder层数（2层）下，探究不同lambda1对于异常检测的影响，即聚类学习与重构学习之间的权衡
# for l in 0.0 # 0.001 0.003 0.005 0.007 # 0.01 0.03 0.05 0.07 0.1 0.3 0.5 0.7 1.0 1.2 1.5 1.7 2.0
# do
#         python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
# done

# # 参数敏感性：在特定（或不同）encoder层数（1层）下，探究不同lambda1对于异常检测的影响，即聚类学习与重构学习之间的权衡
# for l in 0.0 # 0.001 0.003 0.005 0.007 # 0.01 0.03 0.05 0.07 0.1 0.3 0.5 0.7 1.0 1.2 1.5 1.7 2.0
# do
#         python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
#         python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss True --lambda1 $l
# done

# # 消融实验：在特定（或不同）encoder层数（4层）下，训练时是否使用聚类损失，即聚类学习是否有用
# # 消融实验：在特定（或不同）encoder层数（4层）下，推断时是否使用隐层距离，即隐层距离对于异常检测有帮助
# for l in True False # use_cluster_loss
# do 
#     for t in True False # train_use_cluster_loss
#     do
#         python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 4 --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#     done
# done

# 消融实验：在特定（或不同）encoder层数（1层）下，训练时是否使用聚类损失，即聚类学习是否有用
# 消融实验：在特定（或不同）encoder层数（1层）下，推断时是否使用隐层距离，即隐层距离对于异常检测有帮助
# for l in True False # use_cluster_loss
# do 
#     for t in True False # train_use_cluster_loss
#     do
#         python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#     done
# done


# # 消融实验：在特定（或不同）encoder层数（2层）下，训练时是否使用聚类损失，即聚类学习是否有用
# # 消融实验：在特定（或不同）encoder层数（2层）下，推断时是否使用隐层距离，即隐层距离对于异常检测有帮助
# for l in True False # use_cluster_loss
# do 
#     for t in True False # train_use_cluster_loss
#     do
#         python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#         python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 2 --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss $l
#     done
# done

# # 消融实验：在不同temperature下，在特定（或不同）encoder层数（2层）下，探讨聚类损失在训练的作用
# for l in 0.0001 0.0003 0.0005 0.0007 # 0.001 0.003 0.005 0.007 # 0.01 0.03 0.05 0.07 #0.1 0.3 0.5 0.7 1.0
# do 
#     for t in True False # train_use_cluster_loss
#     do
#         for e in 1
#         do
#                 python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss True --temperature $l --lambda1 0.1
#                 python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss True --temperature $l --lambda1 0.1
#                 python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e  --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss True --temperature $l --lambda1 0.1
#                 python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss True --temperature $l --lambda1 0.1
#                 python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num $e --latent_distance_select_type fusion --train_use_cluster_loss $t --use_cluster_loss True --temperature $l --lambda1 0.1
#         done
#     done
# done

# 可视化：隐层表示的分布，包括分布中心、正常异常->证明“隐层对于异常检测有帮助” done
# 可视化：结果融合时的注意力分布->探究“所提方法对于不同层隐层距离的融合权重”

# 补充实验
python main.py  --dataset SMD --data_path ../MEMTO/data/SMD/  --input_c 38 --output_c 38 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss False
python main.py  --dataset MSL --data_path ../MEMTO/data/MSL/   --input_c 55 --output_c 55 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss False
python main.py  --dataset SMAP --data_path ../MEMTO/data/SMAP/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss False
python main.py  --dataset PSM --data_path ../MEMTO/data/PSM/  --input_c 25 --output_c 25 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1  --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss False
python main.py  --dataset SWAT --data_path ../MEMTO/data/SWAT/A1_A2/  --input_c 50 --output_c 50 --num_epochs 10 --run_times 0 --gpu 2 --encoder_layer_num 1 --latent_distance_select_type fusion --train_use_cluster_loss True --use_cluster_loss False