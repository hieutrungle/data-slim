Machine Leanrning Reduced-order Model

Training

python main.py --command train --model_path ./saved_models/testing_hier_mbconv --data_dir ../data/tccs/ocean/SST_modified --patch_size 64 --pre_num_channels 12 --num_channels 32 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --epochs 1 --batch_size 8 --model_type hier_mbconv_adaptive_2 --verbose True --train_verbose True --local_test True --lr 3e-4

Resume

python main.py --command train --model_path ./saved_models/res_2d_attn_sst --data_dir ../data/tccs/ocean/SST_modified --patch_size 64 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 2 --epochs 3 --batch_size 8 --resume True --iter -1 --train_verbose True --local_test True

python main.py --command train --model_path ./saved_models/simple_model_sst --data_dir ../data/tccs/ocean/SST_modified --patch_size 64 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 2 --epochs 3 --batch_size 8 --resume True --iter -1 --train_verbose True --local_test True

Compression

python main.py --command compress --verbose False --model_path ./saved_models/testing-res_1--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch\=000-val_mse_loss\=111.02807-val_loss\=555.14618.pt --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type res_1 --input_path ../data/tccs/ocean/SST_modified/SST.025001-025912.nc --output_path ./outputs/testing_res_1

python main.py --command compress --verbose False --model_path ./saved_models/testing-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch\=000-val_mse_loss\=22.63494-val_loss\=113.23413.pt --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical --input_path ../data/tccs/ocean/SST_modified/SST.025001-025912.nc --output_path ./outputs/testing_hierachical

Decompression

python main.py --command decompress --verbose False --model_path ./saved_models/testing-res_1--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch\=000-val_mse_loss\=111.02807-val_loss\=555.14618.pt --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type res_1 --input_path ./outputs/testing_res_1 --output_path ./outputs/decompression_testing_res_1

python main.py --command decompress --verbose True --model_path ./saved_models/testing-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch\=000-val_mse_loss\=1.44076-val_loss\=7.50242.pt --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical --input_path ./outputs/testing_hierachical --output_path ./outputs/decompression_testing_hierachical
