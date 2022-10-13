import sys
import os
import torch
from utils import utils, custom_callbacks, scheduler
import gc
import logging.config
import time
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datetime import datetime
import warnings
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

# Callbacks
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer:
    """Trainer"""

    def __init__(self, args):

        self.epochs = args.epochs
        if args.verbose or args.train_verbose:
            logger.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")

    def train(
        self, args, model, ds, dataio, epochs_til_ckpt, resume_checkpoint, test_ds=None
    ):
        """train the model"""
        epochs = args.epochs
        learning_rate = args.learning_rate
        learning_rate_min = args.learning_rate_min
        vq_anneal_portion = args.vq_anneal_portion
        weight_decay = args.weight_decay
        warm_up = args.warm_up

        training_params = dataio.get_training_parameters()

        num_time_slices = training_params["train_num_time_slices"]
        num_patches = training_params["train_num_patches"]
        steps_per_epoch = training_params["train_num_batches"]
        steps_per_execution = math.ceil(steps_per_epoch / 10)
        if test_ds is None:
            validation_steps = None
        else:
            validation_steps = training_params["test_num_batches"]
        if args.verbose or args.train_verbose:
            logger.debug(f"ds: {ds.element_spec}")
            logger.debug(f"test_ds: {test_ds}")
            logger.debug(f"data_patch_size: {args.data_patch_size}")
            logger.debug(f"model_patch_size: {args.model_patch_size}")
            logger.debug(f"num_time_slices: {num_time_slices}")
            logger.debug(f"num_patches: {num_patches}")
            logger.debug(f"steps_per_epoch: {steps_per_epoch}")
            logger.debug(f"steps_per_execution: {steps_per_execution}")
            logger.debug(f"validation_steps: {validation_steps}")
        hovsd_shape = [1, args.model_patch_size, args.model_patch_size]

        # Save training parameters if we need to resume training in the future
        start_epoch = 0
        if "resume_epoch" in resume_checkpoint:
            start_epoch = resume_checkpoint["resume_epoch"]
        start_epoch = 0 if start_epoch < 0 else start_epoch
        self.start_epoch = start_epoch
        self.vq_anneal_portion = vq_anneal_portion
        optimizer = self.build_optimizer(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate,
            learning_rate_min=learning_rate_min,
            weight_decay=weight_decay,
            warmup_portion=warm_up,
        )
        model.compile(
            optimizer=optimizer,
            steps_per_execution=steps_per_execution,
        )

        summaries_dir, checkpoints_dir = utils.mkdir_storage(
            args.model_path, resume_checkpoint
        )
        summaries_dir = os.path.join(
            summaries_dir, datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        print()
        logger.info(f"Start Training...")
        # Populate with typical keras callbacks
        _callbacks = self.get_callbacks(
            summaries_dir, checkpoints_dir, epochs_til_ckpt, monitor="loss", mode="min"
        )
        start_total_time = time.perf_counter()

        if args.train_verbose:
            model_verbose = 1
        elif args.verbose:
            model_verbose = 2
        else:
            model_verbose = 0
        history = model.fit(
            ds,
            validation_data=test_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=_callbacks,
            verbose=model_verbose,
            initial_epoch=start_epoch,
        )

        total_training_time = time.perf_counter() - start_total_time
        logger.info(f"Training time: {total_training_time:0.2f} seconds")

        # save model at end of training
        self.save_model_sgd(model, os.path.join(checkpoints_dir, "best_model"))

        print()
        gc.collect()
        logger.info(f"Training completed!\n")
        return

    def build_optimizer(
        self,
        epochs,
        steps_per_epoch,
        learning_rate,
        learning_rate_min,
        weight_decay=1e-4,
        warmup_portion=0.1,
    ):

        # Optimizer
        warmup_epochs = int(epochs * warmup_portion)
        warmup_steps = int(steps_per_epoch * warmup_epochs)
        decay_steps = int(steps_per_epoch * epochs * (1 - warmup_portion))
        decay_schedule_fn = tf.keras.optimizers.schedules.CosineDecay(
            learning_rate, decay_steps, alpha=learning_rate_min
        )
        lr_schedule = scheduler.WarmUpLRSchedule(
            initial_learning_rate=learning_rate,
            decay_schedule_fn=decay_schedule_fn,
            warmup_steps=warmup_steps,
            power=1.0,
            name="warmUpCosineDecay",
        )
        optimizer = tfa.optimizers.AdamW(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.95,
            weight_decay=weight_decay,
        )
        return optimizer

    def get_callbacks(
        self,
        summaries_dir,
        checkpoints_dir,
        epochs_til_ckpt=5,
        monitor="val_loss",
        mode="auto",
    ):
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=summaries_dir, histogram_freq=1, update_freq="epoch"
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoints_dir, f"only_weights", f"best"),
                save_weights_only=True,
                monitor=monitor,
                mode=mode,
                save_best_only=True,
            ),
            custom_callbacks.EarlyStopCallback(
                patience=self.epochs // 4, monitor=monitor
            ),
            custom_callbacks.CheckpointCallback(checkpoints_dir, epochs_til_ckpt),
            custom_callbacks.AnnealingLossWeightCallback(
                self.start_epoch,
                self.vq_anneal_portion,
                self.epochs,
                min_vq_weight=1e-4,
            ),
        ]
        return callbacks

    def save_model_sgd(self, model, path):
        optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4, momentum=0.9)
        model.compile(optimizer=optimizer)
        model.save(path)

    def train_manually(
        self,
        args,
        model,
        train_loader,
        loss_module,
        optim_func,
        start_epoch,
        epochs,
        steps_per_epoch,
        steps_per_execution,
        _callbacks,
        device,
        logging_dir="runs/our_experiment",
        test_loader=None,
    ):
        # TODO: optimizer
        optimizer = optim_func(model.parameters())

        # Create TensorBoard logger
        writer = SummaryWriter(logging_dir)
        model_plotted = False

        # Set model to train mode
        model.train()
        logs = {}
        # Training loop
        for epoch in range(start_epoch, epochs):
            if args.train_verbose:
                tqdm.write(f"Epoch {epoch+1} ")
                pbar = tqdm(total=steps_per_epoch, position=0, leave=True)
            elif args.verbose:
                logger.info(f"Epoch {epoch+1} ")

            epoch_loss = 0.0

            start_epoch_time = time.perf_counter()
            for x, mask in train_loader:

                # For the very first batch, we visualize the computation graph in TensorBoard
                if not model_plotted:
                    writer.add_graph(model, x)
                    model_plotted = True

                ## Step 1: Move input data to device (only strictly necessary if we use GPU)
                x = x.to(device)
                mask = mask.to(device)

                ## Step 2: Run the model on the input data
                x_hat = model(x)

                ## Step 3: Calculate the loss
                x_hat *= mask
                x *= mask
                loss = loss_module(x_hat, x.float())

                ## Step 4: Perform backpropagation
                # Before calculating the gradients, we need to ensure that they are all zero.
                # The gradients would not be overwritten, but actually added to the existing ones.
                optimizer.zero_grad()
                # Perform backpropagation
                loss.backward()

                ## Step 5: Update the parameters
                optimizer.step()

                ## Step 6: Take the running average of the loss
                epoch_loss += loss.item()

                logs.update({"time: ": time.perf_counter() - start_epoch_time})
                if args.train_verbose:
                    pbar.update(steps_per_execution)
                    pbar.set_description(
                        "; ".join([f"{k}: {v:0.4f}" for k, v in logs.items()])
                    )

            if args.train_verbose:
                pbar.close()
            else:
                logger.info("; ".join([f"{k}: {v:0.5f}" for k, v in logs.items()]))

            # Add average loss to TensorBoard
            epoch_loss /= len(train_loader)
            writer.add_scalar("training_loss", epoch_loss, global_step=epoch + 1)

            # Visualize prediction and add figure to TensorBoard
            # Since matplotlib figures can be slow in rendering, we only do it every 10th epoch
            # if test_loader is not None:
            #     if (epoch + 1) % 10 == 0:
            #         visual_data = next(iter(test_loader))
            #         x, mask = visual_data  # Get first batch
            #         fig = self.visualize_prediction(model, x, mask, device)
            #         writer.add_figure("predictions", fig, global_step=epoch + 1)
        torch.save(model.state_dict(), "our_model.tar")
        writer.close()

    @torch.no_grad()  # Decorator, same effect as "with torch.no_grad(): ..." over the whole function.
    def visualize_prediction(self, model, x, mask, device):

        fig = plt.figure(figsize=(4, 4), dpi=500)

        model.to(device)
        x_hat = model(x)
        x_hat *= mask
        x *= mask
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(x_hat, torch.Tensor):
            x_hat = x_hat.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        return fig

    # callbacks = tf.keras.callbacks.CallbackList(
    #     _callbacks, add_history=True, model=model
    # )
    # logs = {}
    # callbacks.on_train_begin(logs=logs)

    # # training loop
    # for epoch in range(start_epoch, epochs):
    #     callbacks.on_epoch_begin(epoch, logs=logs)
    #     if args.train_verbose:
    #         tqdm.write(f"Epoch {epoch+1} ")
    #         pbar = tqdm(total=steps_per_epoch, position=0, leave=True)
    #     elif args.verbose:
    #         logger.info(f"Epoch {epoch+1} ")

    #     start_epoch_time = time.perf_counter()
    #     tf_da_iter = iter(ds)
    #     for bulk_step in range(steps_per_epoch // steps_per_execution):
    #         if isinstance(
    #             model.optimizer.lr,
    #             tf.keras.optimizers.schedules.LearningRateSchedule,
    #         ):
    #             lr = model.optimizer.lr(model.optimizer.iterations)
    #         else:
    #             lr = model.optimizer.lr.numpy()

    #         callbacks.on_batch_begin(bulk_step, logs=logs)
    #         callbacks.on_train_batch_begin(bulk_step, logs=logs)

    #         model.reset_states()
    #         logs = self.train_step(
    #             model,
    #             tf_da_iter,
    #             tf.constant(steps_per_execution, dtype=tf.int32),
    #         )
    #         logs.update({"lr": lr, "vq_weight": model.get_vq_weight()})
    #         if args.train_verbose:
    #             pbar.update(steps_per_execution)
    #             pbar.set_description(
    #                 "; ".join([f"{k}: {v:0.4f}" for k, v in logs.items()])
    #             )

    #         callbacks.on_train_batch_end(bulk_step, logs=logs)
    #         callbacks.on_batch_end(bulk_step, logs=logs)

    #     if args.train_verbose:
    #         pbar.close()
    #     else:
    #         logs.update({"time: ": time.perf_counter() - start_epoch_time})
    #         logger.info("; ".join([f"{k}: {v:0.5f}" for k, v in logs.items()]))

    #     callbacks.on_epoch_end(epoch, logs=logs)
    #     if model.stop_training:
    #         break

    # callbacks.on_train_end(logs=logs)

    # # Fetch the history object we normally get from keras.fit
    # history_object = None
    # for cb in callbacks:
    #     if isinstance(cb, tf.keras.callbacks.History):
    #         history_object = cb
    # assert history_object is not None
    # return history_object

    def eval_model(model, data_loader, device):
        model.eval()  # Set model to eval mode
        true_preds, num_preds = 0.0, 0.0

        with torch.no_grad():  # Deactivate gradients for the following code
            for data_inputs, data_labels in data_loader:

                # Determine prediction of model on dev set
                data_inputs, data_labels = data_inputs.to(device), data_labels.to(
                    device
                )
                preds = model(data_inputs)
                preds = preds.squeeze(dim=1)
                preds = torch.sigmoid(
                    preds
                )  # Sigmoid to map predictions between 0 and 1
                pred_labels = (preds >= 0.5).long()  # Binarize predictions to 0 and 1

                # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
                true_preds += (pred_labels == data_labels).sum()
                num_preds += data_labels.shape[0]

        acc = true_preds / num_preds
        print(f"Accuracy of the model: {100.0*acc:4.2f}%")

    @torch.no_grad()  # Decorator, same effect as "with torch.no_grad(): ..." over the whole function.
    def visualize_classification(model, data, label, device):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        data_0 = data[label == 0]
        data_1 = data[label == 1]

        fig = plt.figure(figsize=(4, 4), dpi=500)
        plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
        plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
        plt.title("Dataset samples")
        plt.ylabel(r"$x_2$")
        plt.xlabel(r"$x_1$")
        plt.legend()

        # Let's make use of a lot of operations we have learned above
        model.to(device)
        c0 = torch.Tensor(to_rgba("C0")).to(device)
        c1 = torch.Tensor(to_rgba("C1")).to(device)
        x1 = torch.arange(-0.5, 1.5, step=0.01, device=device)
        x2 = torch.arange(-0.5, 1.5, step=0.01, device=device)
        xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
        model_inputs = torch.stack([xx1, xx2], dim=-1)
        preds = model(model_inputs)
        preds = torch.sigmoid(preds)
        output_image = (1 - preds) * c0[None, None] + preds * c1[
            None, None
        ]  # Specifying "None" in a dimension creates a new one
        output_image = (
            output_image.cpu().numpy()
        )  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
        plt.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
        plt.grid(False)
        return fig
