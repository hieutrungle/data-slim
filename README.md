# DataSlim

This project is a PyTorch implementation of the paper ["Hierarchical Autoencoder-based Lossy Compression for Large-scale High-resolution Scientific Data"](https://arxiv.org/).

# Usage

This section walks through how to train and sample from a model.

## Installation

Clone this repository and navigate to it in your terminal. Activate your environment. Then run:

```
pip install -r requirements.txt
```

This should install the all required python packages that the scripts depend on.

## Preparing Data

The training data is the Subsurface Sea Temperature (SST). The data have been modified to have the same dimension across all netCDF files.

Raw data can be found at [High-Resolution Earth System Prediction Project](https://datahub.geos.tamu.edu:8880/thredds/catalog/iHESPDataHUB/B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02/ocn/SST/catalog.html)

During training, all netCDF files are automatically processed to become suitable data types before being fed into the model. Simply pass `--data_path path/to/folder` to the training script, and it will take care of the rest.

## Training

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, data infomation, and training flags. Here are some reasonable defaults for a baseline:

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
TRAIN_FLAGS="--lr 3e-4 --batch_size 128 --epochs 100 --train_verbose True"
```

Once you have setup your hyper-parameters, you can run an experiment like so:

```
python main.py --command train --verbose True --data_path path/to/images --model_path /path/to/save/model $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS
```

The logs and saved models will be written to a logging directory determined by the `OPENAI_LOGDIR` environment variable. If it is not set, then a temporary directory will be created in `/tmp_logs`.

The above training script saves checkpoints to `.pt` files in the saved directory, which is defined in `--model_path`. These checkpoints will have names like `sst-epoch=008-val_mse_loss=0.01161-val_loss=0.07661.pt`.

## Resume training

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
TRAIN_FLAGS="--lr 3e-4 --batch_size 128 --epochs 100 --train_verbose True"
```

```
python main.py --command train --verbose True --data_path path/to/images --model_path /path/to/save/model $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS --resume True --iter -1
```

To be updated

## Compression

Once you have a path to your model, you can compress a similar file like so:

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
```

The patch size can be changed (should be a power of 2). The default value for the patch size is 256. With the hyperparameters defined, we can run the following command to compress the data:

```
python main.py --command compress --data_path path/to/images --model_path /path/to/save/model $MODEL_FLAGS $DATA_FLAGS --input_path /path/to/nc/file.nc --output_path /path/to/output/folder --batch_size 128
```

The batch size should be changed according to the GPU memory. This will output the compressed data and store them in the `output_path`. This command also handle meta data for the input file.

## Get data based on bounding box

Similar to the compression, we need to define sets of hyperparameters. The only difference is that we need to define the bounding box for the data we want to get.

Assuming the data is of size (120, 3600, 2400), which coresponds to (time, lat, lon), we want to get the data from (0, 524, 234) to (5, 2541, 2054)

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
GET_DATA_FLAGS="--start_time 0 --end_time 5 --start_pos_x 524 --start_pos_y 234 --end_pos_x 2541 --end_pos_y 2054"
```

With the hyperparameters defined, we can run the following command to get the data:

```
python main.py --command get_data --data_path path/to/images --model_path /path/to/save/model $MODEL_FLAGS $DATA_FLAGS $GET_DATA_FLAGS --input_path /path/to/compressed/data/folder/ --output_path /path/to/output/folder --batch_size 128
```

Again, the batch size can be adjusted. **The patch size argument should match the one used for compression**. The output will be a netCDF file with the same dimension as the original file.

## Examples

**Training**

For netcdf data, we can use the following command to train the model:

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600 --data_type netcdf"
TRAIN_FLAGS="--lr 3e-4 --batch_size 128 --epochs 100 --train_verbose True --local_test True"

python main.py --command train --data_path ../data/tccs/ocean/SST_modified --model_path ./saved_models/tmp --verbose True $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS
```

**Compression**

```
MODEL_FLAGS="--patch_size 256 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"

python main.py --command compress --verbose False --model_path ./examples/trained_hierarchical_models/checkpoints/sst-epoch\=008-val_mse_loss\=0.01161-val_loss\=0.07661.pt $MODEL_FLAGS $DATA_FLAGS --input_path ../data/tccs/ocean/SST_modified/SST.051001-051912.nc --output_path ./outputs/compressed_data  --batch_size 8
```

**Get data**

```
MODEL_FLAGS="--patch_size 256 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
GET_DATA_FLAGS="--start_time 0 --end_time 5 --start_pos_x 524 --start_pos_y 234 --end_pos_x 2541 --end_pos_y 2054"

python main.py --command get_data --model_path ./examples/trained_hierarchical_models/checkpoints/sst-epoch\=008-val_mse_loss\=0.01161-val_loss\=0.07661.pt $MODEL_FLAGS $DATA_FLAGS $GET_DATA_FLAGS --input_path ./outputs/compressed_data/ --output_path ./outputs/get_data_compressed_data --batch_size 10
```

**Decompression**
CLOUD

```
MODEL_FLAGS="--patch_size 256 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"

python main.py --command decompress --verbose False --model_path ./examples/trained_hierarchical_models/checkpoints/sst-epoch\=008-val_mse_loss\=0.01161-val_loss\=0.07661.pt $MODEL_FLAGS $DATA_FLAGS --input_path ./outputs/testing_sst_final_data  --output_path ./outputs/testing_sst_final_data_decompressed --batch_size 1
```

## Benchmark

**Training**

For binary CESM-CLOUD data (.f32), we can use the following command to train the model:

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
TRAIN_FLAGS="--lr 3e-4 --batch_size 128 --epochs 1 --train_verbose True --log_interval 10"
python main.py --command train --data_path ../data/SDRBENCH-CESM-ATM-26x1800x3600/CLOUD_1_26_1800_3600.f32 --model_path ./examples/tmp --verbose True $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS
```

For binary hurrican_isabel data (.f32), we can use the following command to train the model:

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 2 --model_type hierachical"
DATA_FLAGS="--data_height 500 --data_width 500  --data_depth 100 --data_type binary --ds_name hurrican_isabel --da_name QVAPORf48.log10.bin.f32"
TRAIN_FLAGS="--lr 3e-4 --batch_size 128 --epochs 1 --train_verbose True --log_interval 10"
python main.py --command train --data_path ../data/SDRBENCH-Hurricane-ISABEL-100x500x500/QVAPORf48.log10.bin.f32 --model_path ./examples/tmp --verbose True $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS
```

For binary nyx data (.f32), we can use the following command to train the model:

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 2 --model_type hierachical"
DATA_FLAGS="--data_height 512 --data_width 512  --data_depth 512 --data_type binary --ds_name nyx --da_name temperature"
TRAIN_FLAGS="--lr 3e-4 --batch_size 128 --epochs 1 --train_verbose True --log_interval 10"
python main.py --command train --data_path ../data/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32 --model_path ./examples/tmp --verbose True $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS
```

**Compression**

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
python main.py --command compress --verbose False --model_path ./examples/trained_hierarchical_models/checkpoints/sst-epoch\=008-val_mse_loss\=0.01161-val_loss\=0.07661.pt $MODEL_FLAGS $DATA_FLAGS --input_path ../data/tccs/ocean/SST_modified/SST.025001-025912.nc --output_path ./outputs/compressed_data  --batch_size 128
```

For binary CESM-CLOUD data (.f32), we can use the following command to benchmark the model:

CESM 26x1800x3600

CLOUD

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 16 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.1"

python main.py --command compress --model_path ./saved_model/CLOUD_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_16-num_channels_64-latent_dim_128-num_embeddings_128-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=098-val_mse_loss=0.00003-val_loss=0.00017.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ../data/SDRBENCH-CESM-ATM-26x1800x3600/CLOUD_1_26_1800_3600.f32 --output_path ./outputs/CLOUD_1_26_1800_3600.f32_tol_0.1_images  --batch_size 1  --benchmark True --verbose True
```

FICE

```
MODEL_FLAGS="--patch_size 256 --pre_num_channels 16 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.1"

python main.py --command compress --model_path ./saved_model/FICE_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_16-num_channels_64-latent_dim_128-num_embeddings_128-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=091-val_mse_loss=0.00005-val_loss=0.00019.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ../data/SDRBENCH-CESM-ATM-26x1800x3600/FICE_1_26_1800_3600.f32 --output_path ./outputs/FICE_1_26_1800_3600.f32.f32_tol_0.1_images  --batch_size 8  --benchmark True --verbose True
```

CESM 1800x3600

CLOUD

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 16 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.1"

python main.py --command compress --model_path ./saved_model/CLOUD_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_16-num_channels_64-latent_dim_128-num_embeddings_128-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=098-val_mse_loss=0.00003-val_loss=0.00017.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ../data/SDRBENCH-CESM-ATM-cleared-1800x3600/CLDHGH_1_1800_3600.dat --output_path ./outputs/CLDHGH_1_1800_3600.dat_tol_0.1_images --batch_size 1 --benchmark True --verbose True
```

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 16 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.1"

python main.py --command compress --model_path ./saved_model/CLOUD_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_16-num_channels_64-latent_dim_128-num_embeddings_128-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=098-val_mse_loss=0.00003-val_loss=0.00017.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ../data/SDRBENCH-CESM-ATM-cleared-1800x3600/CLDLOW_1_1800_3600.dat --output_path ./outputs/CLDLOW_1_1800_3600.dat_tol_0.1_images --batch_size 1 --benchmark True --verbose True
```

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 16 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.1"

python main.py --command compress --model_path ./saved_model/CLOUD_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_16-num_channels_64-latent_dim_128-num_embeddings_128-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=098-val_mse_loss=0.00003-val_loss=0.00017.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ../data/SDRBENCH-CESM-ATM-cleared-1800x3600/CLDMED_1_1800_3600.dat --output_path ./outputs/CLDMED_1_1800_3600.dat_tol_0.1_images --batch_size 1 --benchmark True --verbose True
```

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 16 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.1"

python main.py --command compress --model_path ./saved_model/CLOUD_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_16-num_channels_64-latent_dim_128-num_embeddings_128-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=098-val_mse_loss=0.00003-val_loss=0.00017.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ../data/SDRBENCH-CESM-ATM-cleared-1800x3600/CLDTOT_1_1800_3600.dat --output_path ./outputs/CLDTOT_1_1800_3600.dat_tol_0.1_images --batch_size 1 --benchmark True --verbose True
```

For binary nyx data (.f32), we can use the following command to benchmark the model:

Temperature

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 2 --model_type hierachical"
DATA_FLAGS="--data_height 512 --data_width 512  --data_depth 512 --data_type binary --ds_name nyx --da_name temperature.log10.f32"
COMPRESSION_FLAGS="--straight_through_weight 1"
python main.py --command compress --model_path ./saved_model/temperature.log10.f32-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=093-val_mse_loss=0.00176-val_loss=2.86588.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ../data/SDRBENCH-EXASKY-NYX-512x512x512/temperature.log10.f32 --output_path ./outputs/temperature.log10_tol_0.1  --batch_size 1  --benchmark True
```

Dark_matter_density

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 2 --model_type hierachical"
DATA_FLAGS="--data_height 512 --data_width 512  --data_depth 512 --data_type binary --ds_name nyx --da_name dark_matter_density.log10.f32"
python main.py --command compress --model_path ./saved_model/baryon_density.log10.f32-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=085-val_mse_loss=0.00467-val_loss=2.76204.pt $MODEL_FLAGS $DATA_FLAGS --input_path ../data/SDRBENCH-EXASKY-NYX-512x512x512/dark_matter_density.log10.f32 --output_path ./outputs/dark_matter_density.log10  --batch_size 128  --benchmark True
```

Baryon_density

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 2 --model_type hierachical"
DATA_FLAGS="--data_height 512 --data_width 512  --data_depth 512 --data_type binary --ds_name nyx --da_name baryon_density.log10.f32"
COMPRESSION_FLAGS="--straight_through_weight 3"

python main.py --command compress --model_path ./saved_model/baryon_density.log10.f32-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=096-val_mse_loss=0.00463-val_loss=2.76147.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ../data/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.log10.f32 --output_path ./outputs/baryon_density.log10_tol_0.1  --batch_size 1  --benchmark True
```

**Decompression**

CESM 1800x3600

CLOUD

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 16 --num_channels 64 --latent_dim 128 --num_embeddings 128 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.1"

python main.py --command decompress --model_path ./saved_model/CLOUD_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_16-num_channels_64-latent_dim_128-num_embeddings_128-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=098-val_mse_loss=0.00003-val_loss=0.00017.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ./outputs/CLDHGH_1_1800_3600.dat_tol_0.1  --output_path ./outputs/CLDHGH_1_1800_3600.dat_tol_0.1-decompress --batch_size 1  --benchmark True
```

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.045"

python main.py --command decompress --model_path ./saved_model/CLOUD_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=077-val_mse_loss=0.00113-val_loss=2.11832.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ./outputs/CLDLOW_1_1800_3600.dat_tol_0.1 --output_path ./outputs/CLDLOW_1_1800_3600.dat_tol_0.1_decompress  --batch_size 1  --benchmark True
```

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.045"

python main.py --command decompress --model_path ./saved_model/CLOUD_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=077-val_mse_loss=0.00113-val_loss=2.11832.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ./outputs/CLDMED_1_1800_3600.dat_tol_0.1 --output_path ./outputs/CLDMED_1_1800_3600.dat_tol_0.1_decompress  --batch_size 1  --benchmark True
```

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 1 --model_type hierachical"
DATA_FLAGS="--data_height 1800 --data_width 3600 --data_type binary --ds_name CESM --da_name CLOUD"
COMPRESSION_FLAGS="--straight_through_weight 1000  --tolerance 0.045"

python main.py --command decompress --model_path ./saved_model/CLOUD_1_26_1800_3600.f32-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=077-val_mse_loss=0.00113-val_loss=2.11832.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ./outputs/CLDTOT_1_1800_3600.dat_tol_0.1 --output_path ./outputs/CLDTOT_1_1800_3600.dat_tol_0.1_decompress  --batch_size 1  --benchmark True
```

Temperature

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 2 --model_type hierachical"
DATA_FLAGS="--data_height 512 --data_width 512  --data_depth 512 --data_type binary --ds_name nyx --da_name temperature.log10.f32"
COMPRESSION_FLAGS="--straight_through_weight 1"
python main.py --command decompress --model_path ./saved_model/temperature.log10.f32-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=093-val_mse_loss=0.00176-val_loss=2.86588.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ./outputs/temperature.log10_tol_0.1 --output_path ./outputs/temperature.log10_tol_0.1-decompress --batch_size 1  --benchmark True
```

Baryon_density

```
MODEL_FLAGS="--patch_size 512 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --patch_channels 2 --model_type hierachical"
DATA_FLAGS="--data_height 512 --data_width 512  --data_depth 512 --data_type binary --ds_name nyx --da_name baryon_density.log10.f32"
COMPRESSION_FLAGS="--straight_through_weight 3"
python main.py --command decompress --model_path ./saved_model/temperature.log10.f32-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0/checkpoints/sst-epoch=093-val_mse_loss=0.00176-val_loss=2.86588.pt $MODEL_FLAGS $DATA_FLAGS $COMPRESSION_FLAGS --input_path ./outputs/baryon_density.log10_tol_0.1 --output_path ./outputs/baryon_density.log10_tol_0.1-decompress --batch_size 1  --benchmark True
```
