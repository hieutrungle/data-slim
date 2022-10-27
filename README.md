Machine Leanrning Reduced-order Model

Training

python main.py --command train --model_path ./saved_models/simple_model --data_dir ../data/tccs/ocean/SST_modified --patch_size 64 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 2 --epochs 3 --batch_size 8 --verbose True --train_verbose False --local_test True

Resume

python main.py --command train --model_path ./saved_models/res_2d_attn_sst --data_dir ../data/tccs/ocean/SST_modified --patch_size 64 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 2 --epochs 3 --batch_size 8 --resume True --iter -1 --train_verbose True --local_test True

python main.py --command train --model_path ./saved_models/simple_model_sst --data_dir ../data/tccs/ocean/SST_modified --patch_size 64 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 2 --epochs 3 --batch_size 8 --resume True --iter -1 --train_verbose True --local_test True

Compression

python main.py --command compress --verbose True --model_path ./saved_models/res_2d_attn_sst-patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch\=004-val_mse_loss\=0.05082-val_loss\=0.30779.pt --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --input_path ../data/tccs/ocean/SST_modified/SST.025001-025912.nc --output_path ./outputs/testing

python main.py --command compress --verbose True --model_path ./saved_models/simple_model_sst-patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch\=004-val_mse_loss\=0.05082-val_loss\=0.30779.pt --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --input_path ../data/tccs/ocean/SST_modified/SST.025001-025912.nc

Decompression

python main.py --command decompress --verbose True --model_path ./saved_models/res_2d_attn_sst-patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch\=004-val_mse_loss\=0.05082-val_loss\=0.30779.pt --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --input_path ./outputs/testing --output_path ./outputs/decompression_testing
