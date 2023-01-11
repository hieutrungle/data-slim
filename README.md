# Compression Vector-quantized Variational Autoencoder

This is the codebase for ["paper name and link"](https://arxiv.org/).

# Usage

This section of the README walks through how to train and sample from a model.

## Installation

Clone this repository and navigate to it in your terminal. Activate your environment. Then run:

```
pip install -r requirements.txt
```

This should install the all required python packages that the scripts depend on.

## Preparing Data

The training data is the Subsurface Sea Temperature (SST). The data has been modified to have the same dimension across all netCDF files. The folder containing all files are at ["Geohub"]()

During training, all netCDF files are automatically processed to become suitable data types before being fed into the model. Simply pass `--data_dir path/to/folder` to the training script, and it will take care of the rest.

## Training

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, data infomation, and training flags. Here are some reasonable defaults for a baseline:

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
TRAIN_FLAGS="--lr 3e-4 --batch_size 128 --epochs 100 --train_verbose True"
```

Once you have setup your hyper-parameters, you can run an experiment like so:

```
python main.py --command train --verbose True --data_dir path/to/images --model_path /path/to/save/model $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS
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
python main.py --command train --verbose True --data_dir path/to/images --model_path /path/to/save/model $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS --resume True --iter -1
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
python main.py --command compress --data_dir path/to/images --model_path /path/to/save/model $MODEL_FLAGS $DATA_FLAGS --input_path /path/to/nc/file.nc --output_path /path/to/output/folder --batch_size 128
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
python main.py --command get_data --data_dir path/to/images --model_path /path/to/save/model $MODEL_FLAGS $DATA_FLAGS $GET_DATA_FLAGS --input_path /path/to/compressed/data/folder/ --output_path /path/to/output/folder --batch_size 128
```

Again, the batch size can be adjusted. **The patch size argument should match the one used for compression**. The output will be a netCDF file with the same dimension as the original file.

## Examples

**Training**

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
TRAIN_FLAGS="--lr 3e-4 --batch_size 128 --epochs 100 --train_verbose True"
```

```
python main.py --command train --data_dir ../data/tccs/ocean/SST_modified --model_path ./saved_models/hierarchical_model --verbose True $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS
```

**Compression**

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
```

```
python main.py --command compress --verbose False --model_path ./saved_models/hierachical-model/checkpoints/sst-epoch\=008-val_mse_loss\=0.01161-val_loss\=0.07661.pt $MODEL_FLAGS $DATA_FLAGS --input_path ../data/tccs/ocean/SST_modified/SST.025001-025912.nc --output_path ./outputs/compressed_data
```

**Get data**

```
MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
DATA_FLAGS="--data_height 2400 --data_width 3600"
GET_DATA_FLAGS="--start_time 0 --end_time 5 --start_pos_x 524 --start_pos_y 234 --end_pos_x 2541 --end_pos_y 2054"
```

```
python main.py --command get_data --verbose True --model_path ./saved_models/hierachical-model/checkpoints/sst-epoch\=008-val_mse_loss\=0.01161-val_loss\=0.07661.pt $MODEL_FLAGS $DATA_FLAGS $GET_DATA_FLAGS --input_path ./outputs/hier_SST.051001-051912/ --output_path ./outputs/get_data_hier_SST.051001-051912 --batch_size 128
```
