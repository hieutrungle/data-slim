import tensorflow as tf
import numpy as np
import logging.config
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EarlyStopCallback(tf.keras.callbacks.Callback):
    """A class that stops if val_loss < best_loss after 'patience' of epochs"""

    def __init__(self, patience=10, monitor="val_loss", min_delta=0):
        super().__init__()
        self.best_weights = None
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta  # min loss improvement to be counted
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        self.counter = 0
        self.best_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        loss = logs[self.monitor]

        # Early Stop
        if (loss < self.best_loss) and (self.best_loss - loss >= self.min_delta):
            self.best_loss = loss
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.counter = 0
        else:
            self.counter += 1
        if self.counter > self.patience:
            self._stop_training()
        if isinstance(
            self.model.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            lr = self.model.optimizer.lr(self.model.optimizer.iterations)
        else:
            lr = self.model.optimizer.lr.numpy()

        logs.update({"lr": lr})

        if tf.math.is_inf(loss) or tf.math.is_nan(loss):
            logger.info(f"Loss is {loss}.")
            self._stop_training()

    def on_train_end(self, logs=None):
        loss = logs[self.monitor]
        logger.info(
            f"End of training. {self.monitor}: {loss:0.4f}.\n"
            f"Restoring best weights from epoch {self.best_epoch} "
            f"with loss being {self.best_loss:0.4f}."
        )
        self.model.set_weights(self.best_weights)

    def _stop_training(self):
        logger.info(f"Stop Training!\n")
        self.model.stop_training = True


class CheckpointCallback(tf.keras.callbacks.Callback):
    """Save checkpoint every 'epochs_til_ckpt' epochs"""

    def __init__(self, checkpoints_dir, epochs_til_ckpt):
        super().__init__()
        self.epochs_til_ckpt = epochs_til_ckpt
        self.checkpoints_dir = checkpoints_dir

    def on_epoch_end(self, epoch, logs=None):

        # save model when epochs_til_ckpt requirement is met
        if (not (epoch + 1) % self.epochs_til_ckpt) and epoch:
            save_path = os.path.join(self.checkpoints_dir, f"model_{epoch+1:06d}")
            logger.info(f"Checkpoint saved to {save_path}")
            self.model.save_weights(save_path)


class AnnealingLossWeightCallback(tf.keras.callbacks.Callback):
    """Anneal loss weight from 0 to 1"""

    def __init__(
        self, start_epoch, vq_anneal_portion, total_epochs, min_vq_weight=1e-2
    ):
        super().__init__()

        self.total_epochs = total_epochs
        self.min_vq_weight = min_vq_weight
        self.vq_anneal_portion = vq_anneal_portion
        self.start_epoch = start_epoch
        self.vq_weight = tf.minimum(
            tf.maximum(
                (start_epoch - 1) / (self.vq_anneal_portion * self.total_epochs),
                self.min_vq_weight,
            ),
            1,
        )

    def on_epoch_begin(self, epoch, logs=None):
        tf.keras.backend.set_value(self.model.vq_weight, self.vq_weight)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({"vq_weight": self.model.get_vq_weight()})

        if epoch + 1 > self.start_epoch:
            self.vq_weight = tf.minimum(
                tf.maximum(
                    (epoch + 1) / (self.vq_anneal_portion * self.total_epochs),
                    self.min_vq_weight,
                ),
                1,
            )
