import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
import os
import sys
from utils import utils, scheduler, logger
import gc

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AVAILABLE_GPUS = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
NUM_GPUS = len(AVAILABLE_GPUS)


class Compressor(pl.LightningModule):
    def __init__(self, model, lr, warmup, max_iters, weight_decay):
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
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.randn(tuple(self.model.input_shape))

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        loss, x_hat, perplexity = self.model(x)
        return loss, x_hat, perplexity

    def _get_loss(self, batch):
        """
        Given a batch of data, this function returns the reconstruction loss (MSE in our case)
        """
        x, mask = batch
        x = x.type(torch.float32)
        mask = mask.type(torch.float32)
        quantized_loss, x_hat, perplexity = self.forward(x)
        x_hat = x_hat.type(torch.float32)
        x = x * mask
        x_hat = x_hat * mask
        mse_loss = F.mse_loss(x, x_hat, reduction="none")
        mse_loss = mse_loss.sum(dim=[1, 2, 3]).mean()
        return mse_loss, quantized_loss

    def configure_optimizers(self):
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
        mse_loss, quantized_loss = self._get_loss(batch)
        loss = mse_loss * 2 + quantized_loss
        self.log("mse_loss", mse_loss, prog_bar=True)
        self.log("quantized_loss", quantized_loss, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mse_loss, quantized_loss = self._get_loss(batch)
        loss = mse_loss * 2 + quantized_loss
        self.log("val_mse_loss", mse_loss)
        self.log("val_quantized_loss", quantized_loss)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        mse_loss, quantized_loss = self._get_loss(batch)
        loss = mse_loss * 2 + quantized_loss
        self.log("test_mse_loss", mse_loss)
        self.log("test_quantized_loss", quantized_loss)
        self.log("test_loss", loss, on_epoch=True)

    def compress(self, x):
        return self.model.compress(x)

    def decompress(self, x):
        return self.model.decompress(x)

    def get_model(self):
        return self.model


def train(
    model,
    train_ds,
    dataio,
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
):
    """train the model"""
    compressor_args = dict(
        lr=lr,
        warmup=epochs * len(train_ds) * warm_up_portion,
        max_iters=epochs * len(train_ds),
        weight_decay=weight_decay,
    )

    # Save training parameters if we need to resume training in the future
    start_epoch = 0
    if "resume_epoch" in resume_checkpoint:
        start_epoch = resume_checkpoint["resume_epoch"]
        filename = f"resume_start_{start_epoch}_" + "sst-{epoch:03d}-{train_loss:.2f}"
    else:
        filename = "sst-{epoch:03d}-{train_loss:.2f}"

    summaries_dir, checkpoints_dir = utils.mkdir_storage(model_path, resume_checkpoint)
    summaries_dir = os.path.join(
        summaries_dir, datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    logger.log(f"\nStart Training...")
    _callbacks = [
        EarlyStopping("train_loss", patience=15, mode="min"),
        ModelCheckpoint(
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            dirpath=os.path.join(checkpoints_dir),
            filename=filename,
            save_weights_only=True,
        ),
        LearningRateMonitor("step"),
    ]
    start_total_time = time.perf_counter()
    tfboard_logger = pl.loggers.tensorboard.TensorBoardLogger(
        summaries_dir, log_graph=True
    )
    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoints_dir),
        accelerator="gpu",
        devices=NUM_GPUS,
        max_epochs=epochs,
        logger=tfboard_logger,
        callbacks=_callbacks,
        limit_train_batches=0.01,
        limit_val_batches=0.01,
        limit_test_batches=0.01,
        gradient_clip_algorithm="norm",
        enable_progress_bar=train_verbose,
    )
    lightning_model = Compressor(
        model=model,
        **compressor_args,
    )
    trainer.fit(lightning_model, train_ds, test_ds)
    total_training_time = time.perf_counter() - start_total_time
    logger.info(f"Training time: {total_training_time:0.2f} seconds")
    # Test best model on validation and test set
    test_result = trainer.test(lightning_model, test_ds, verbose=train_verbose)

    for i, (path, _) in enumerate(trainer.checkpoint_callback.best_k_models.items()):
        m = Compressor.load_from_checkpoint(path, model=model)
        torch.save(m.model.state_dict(), path.rpartition(".")[0] + ".pt")
    model = model.to(DEVICE)

    print()
    gc.collect()
    logger.info(f"Training completed!\n")

    return model
