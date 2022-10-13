Machine Leanrning Reduced-order Model

Training

python main.py --model_path ./saved_models/vaetmp --data_shape 1 2400 3600 1 -V train --data_path ../data/tccs/ocean/SST_modified --data_patch_size 150 --model_patch_size 128 --num_channels 96 --latent_dim 128 --epochs 1 --batch_size 8

python main.py --data_shape 1 2400 3600 1 -V --model_path ./saved_models/res_2d_attn_sst train --data_path ../data/tccs/ocean/SST_modified --data_patch_size 128 --model_patch_size 128 --pre_num_channels 32 --num_channels 96 --latent_dim 96 --num_embeddings 128 --num_residual_layers 3 --num_transformer_layers 2 --epochs 3 --batch_size 8 --train_verbose

Resume

python main.py --data_shape 1 2400 3600 1 -V --model_path ./saved_models/res_2d_attn_sst train --data_path ../data/tccs/ocean/SST_modified --data_patch_size 128 --model_patch_size 128 --pre_num_channels 32 --num_channels 96 --latent_dim 96 --num_embeddings 128 --num_residual_layers 3 --num_transformer_layers 2 --epochs 3 --batch_size 8 --resume --iter -1 --train_verbose

Compression

python main.py --data_shape 1 2400 3600 1 --model_path ./saved_models/res_2d_attn_sst-latent_dim_96-num_embeddings_128-batch_size_128-data_patch_size_150-model_patch_size_128/checkpoints/best_model/ -V compress --input_file ../data/tccs/ocean/SST_modified/SST.033801-033912.nc
