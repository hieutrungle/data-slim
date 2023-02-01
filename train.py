import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar,
)
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
import os
import sys
from utils import utils, scheduler, logger, custom_callbacks
import gc

DEVICE = torch.device(str(os.environ.get("DEVICE", "cpu")))
NUM_GPUS = int(os.environ.get("NUM_GPUS", 0))


class Compressor(pl.LightningModule):
    def __init__(
        self, model, lr, warmup, max_iters, weight_decay, resume_checkpoint, **kwargs
    ):
        """
        Inputs:
            model: model to be trained
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters(ignore=["model"])
        # Create model
        self.model = model
        # self.vq_weight = 0.5
        self.mse_weight = 3
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.randn(tuple(self.model.input_shape))

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        loss, x_hat, perplexity = self.model(x)
        return loss, x_hat, perplexity

    def _get_fft_mse_loss(self, x, x_hat):
        dim = tuple(range(1, len(x.shape), 1))
        x_fft = torch.fft.fftn(x, dim=dim, norm="ortho")
        x_fft_magnitude = torch.abs(x_fft)
        x_fft_angle = torch.angle(x_fft)

        x_hat_fft = torch.fft.fftn(x_hat, dim=dim, norm="ortho")
        x_hat_fft_magnitude = torch.abs(x_hat_fft)
        x_hat_fft_angle = torch.angle(x_hat_fft)

        fft_magnitude_mse = F.mse_loss(x_fft_magnitude, x_hat_fft_magnitude)
        fft_angle_mse = F.mse_loss(x_fft_angle, x_hat_fft_angle)
        fft_mse_loss = fft_magnitude_mse + fft_angle_mse
        return fft_mse_loss

    def _get_loss(self, batch):
        """
        Given a batch of data, this function returns the reconstruction loss (MSE in our case)
        """
        x, mask = batch
        x = x.type(torch.float32)
        mask = mask.type(torch.float32)
        quantized_loss, x_hat, _ = self.forward(x)
        x_hat = x_hat.type(torch.float32)
        x = x * mask
        x_hat = x_hat * mask
        mse_loss = F.mse_loss(x, x_hat)
        # frequency domain mse loss
        fft_mse_loss = self._get_fft_mse_loss(x, x_hat)

        return mse_loss, quantized_loss, fft_mse_loss

    def configure_optimizers(self):
        if len(self.hparams.resume_checkpoint) > 0:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.9999),
                eps=1e-08,
                weight_decay=1e-8,
                amsgrad=False,
            )
        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.95),
                eps=1e-08,
                weight_decay=self.hparams.weight_decay,
                amsgrad=False,
            )
        # Apply lr scheduler per step
        lr_scheduler = scheduler.CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        mse_loss, quantized_loss, fft_mse_loss = self._get_loss(batch)
        loss = mse_loss * self.mse_weight + quantized_loss + fft_mse_loss * 0.5
        # loss = mse_loss * self.mse_weight + quantized_loss
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log("mse_loss", mse_loss, prog_bar=True)
        self.log("quantized_loss", quantized_loss, prog_bar=True)
        self.log("fft_mse_loss", fft_mse_loss, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        self.log("hp/train_loss", loss, sync_dist=True)
        self.log("hp/train_mse", mse_loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mse_loss, quantized_loss, fft_mse_loss = self._get_loss(batch)
        loss = mse_loss * self.mse_weight + quantized_loss + fft_mse_loss
        self.log("val_mse_loss", mse_loss, sync_dist=True)
        self.log("val_quantized_loss", quantized_loss, sync_dist=True)
        self.log("val_fft_mse_loss", fft_mse_loss, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("hp/val_loss", loss, sync_dist=True)
        self.log("hp/val_mse", mse_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        mse_loss, quantized_loss, fft_mse_loss = self._get_loss(batch)
        loss = mse_loss * self.mse_weight + quantized_loss + fft_mse_loss
        self.log("test_mse_loss", mse_loss, sync_dist=True)
        self.log("test_quantized_loss", quantized_loss, sync_dist=True)
        self.log("test_fft_mse_loss", fft_mse_loss, sync_dist=True)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

    def compress(self, x):
        return self.model.compress(x)

    def decompress(self, x):
        return self.model.decompress(x)

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {"hp/train_loss": 0, "hp/train_mse": 0, "hp/val_loss": 0, "hp/val_mse": 0},
        )


def train(
    model,
    train_ds,
    model_path,
    epochs,
    lr,
    warm_up_portion,
    weight_decay,
    log_interval,
    save_interval,
    resume_checkpoint,
    test_ds=None,
    train_verbose=False,
    args=None,
    dataio=None,
):
    """train the model"""
    compressor_args = dict(
        lr=lr,
        warmup=epochs * len(train_ds) * warm_up_portion,
        max_iters=epochs * len(train_ds),
        weight_decay=weight_decay,
        resume_checkpoint=resume_checkpoint,
    )

    (image, mask) = get_single_slice_test_ds(test_ds, dataio)

    # Save training parameters if we need to resume training in the future
    start_epoch = 0
    weight_filename = "sst-{epoch:03d}-{val_mse_loss:.5f}-{val_loss:.5f}"
    if "resume_epoch" in resume_checkpoint:
        start_epoch = resume_checkpoint["resume_epoch"]
        weight_filename = f"resume_start_{start_epoch}_" + weight_filename
        version = "resume"
    else:
        version = "pretrain"
    summaries_dir, checkpoints_dir = utils.mkdir_storage(model_path, resume_checkpoint)
    _callbacks = get_callbacks(checkpoints_dir, weight_filename, train_verbose)
    _callbacks.append(custom_callbacks.GenerateCallback(image, mask, dataio))

    logger.log(f"\nStart Training...")
    start_total_time = time.perf_counter()
    tfboard_logger = pl.loggers.tensorboard.TensorBoardLogger(
        summaries_dir, name="", version=version, log_graph=True, default_hp_metric=False
    )
    limit_val_batches = 1.0
    limit_train_batches = 1.0
    limit_test_batches = 1.0
    # if args is not None:
    #     if args.local_test == True:
    #         limit_val_batches = 0.05
    #         limit_train_batches = 0.05
    #         limit_test_batches = 0.05

    trainer = pl.Trainer(
        fast_dev_run=False,
        default_root_dir=os.path.join(checkpoints_dir),
        accelerator="gpu",
        devices=NUM_GPUS,
        max_epochs=epochs,
        log_every_n_steps=log_interval,
        logger=tfboard_logger,
        callbacks=_callbacks,
        # limit_val_batches=limit_val_batches,
        # limit_train_batches=limit_train_batches,
        # limit_test_batches=limit_test_batches,
        gradient_clip_algorithm="norm",
        enable_progress_bar=train_verbose,
    )
    lightning_model = Compressor(
        model=model,
        **compressor_args,
        **utils.args_to_dict(args, utils.model_defaults().keys()),
    )
    trainer.fit(lightning_model, train_ds, test_ds)
    total_training_time = time.perf_counter() - start_total_time
    logger.log(f"Training time: {total_training_time:0.2f} seconds")

    # Test best model on validation and test set
    logger.log(f"Loading best model from {_callbacks[1].best_model_path}")
    lightning_model = Compressor.load_from_checkpoint(
        _callbacks[1].best_model_path, model=model
    )
    test_result = trainer.test(lightning_model, test_ds, verbose=train_verbose)
    logger.log(f"\n{test_result}\n")

    for i, (path, _) in enumerate(trainer.checkpoint_callback.best_k_models.items()):
        m = Compressor.load_from_checkpoint(path, model=model)
        torch.save(m.model.state_dict(), path.rpartition(".")[0] + ".pt")
    model = model.to(DEVICE)

    gc.collect()
    logger.info(f"\nTraining completed!\n")

    return model


def get_callbacks(
    checkpoints_dir, weight_filename="{epoch:03d}-{train_loss:.2f}", verbose=False
):
    callbacks = [
        EarlyStopping("val_loss", patience=25, mode="min"),
        ModelCheckpoint(
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            dirpath=checkpoints_dir,
            filename=weight_filename,
            save_weights_only=True,
        ),
        LearningRateMonitor("step"),
    ]
    if verbose:
        callbacks.append(TQDMProgressBar(refresh_rate=100))
    return callbacks


def get_single_slice_test_ds(test_ds, dataio):
    """Get a single slice from the test dataset"""
    num_batch_per_time_slice = dataio.params["test.num_batch_per_time_slice"]
    image, mask = [], []
    for i, (da, tile_mask) in enumerate(test_ds):
        image.append(da.type(torch.float))
        mask.append(tile_mask.type(torch.float))
        if i >= num_batch_per_time_slice:
            break
    return image, mask
