from pytorch_lightning.callbacks import Callback
import torch
import torchvision
import matplotlib.pyplot as plt


class GenerateCallback(Callback):
    def __init__(self, input_tiles, mask_tiles, dataio):
        super().__init__()
        self.input_tiles = input_tiles  # Images to reconstruct during training
        self.mask_tiles = mask_tiles
        self.dataio = dataio

    def on_test_end(self, trainer, pl_module):
        # Reconstruct images
        num_patch_per_time_slice = self.dataio.params["test.num_patch_per_time_slice"]
        x = self.input_tiles
        mask_tiles = self.mask_tiles

        with torch.no_grad():
            pl_module.eval()
            # Reconstruct images using batches of patches
            x_hat = []
            for x_tile in x:
                _, x_hat_tile, _ = pl_module(
                    x_tile.to(pl_module.device).type(torch.float)
                )
                x_hat.append(x_hat_tile)

            # From tiles to the original image
            x_hat = torch.cat(x_hat, dim=0).to(pl_module.device).type(torch.float)
            x = torch.cat(x, dim=0).to(pl_module.device).type(torch.float)
            mask_tiles = (
                torch.cat(mask_tiles, dim=0).to(pl_module.device).type(torch.float)
            )
            x = x[:num_patch_per_time_slice]
            x_min, x_max = torch.min(x).detach().cpu(), torch.max(x).detach().cpu()
            mask_tiles = mask_tiles[:num_patch_per_time_slice]
            x_hat = x_hat[:num_patch_per_time_slice]

            x = x * mask_tiles
            x = torch.permute(x, (0, 2, 3, 1)).detach().cpu().numpy()
            x = self.dataio.revert_partition(x)
            x = torch.tensor(x)
            x = torch.permute(x, (0, 3, 1, 2)).detach().cpu()

            x_hat = x_hat * mask_tiles
            x_hat = torch.permute(x_hat, (0, 2, 3, 1)).detach().cpu().numpy()
            x_hat = self.dataio.revert_partition(x_hat)
            x_hat = torch.tensor(x_hat)
            x_hat = torch.permute(x_hat, (0, 3, 1, 2)).detach().cpu()

            pl_module.train()

        # Plot and add to tensorboard
        imgs = torch.stack([x, x_hat], dim=1).flatten(0, 1)
        grid = torchvision.utils.make_grid(
            imgs, nrow=2, normalize=True, value_range=(x_min, x_max)
        )
        trainer.logger.experiment.add_image(
            "Original (left) vs Reconstruction (right)",
            grid,
            global_step=trainer.global_step,
        )

        x = torch.permute(x, (0, 2, 3, 1)).numpy()
        x_hat = torch.permute(x_hat, (0, 2, 3, 1)).numpy()
        # plot with plt
        fig_ = plt.figure(figsize=(36, 12))
        axes = fig_.subplots(nrows=1, ncols=2, sharey=True)
        for i, (image, name) in enumerate(
            zip([x, x_hat], ["original", "reconstruction"])
        ):
            im = axes[i].imshow(image[0], vmin=x_min, vmax=x_max, cmap="seismic")
            axes[i].set_adjustable("box")
            axes[i].axis("tight")
            axes[i].axis("off")
            axes[i].set_title(f"{name}")
            axes[i].autoscale(False)
        fig_.colorbar(
            im,
            ax=axes,
            location="bottom",
            fraction=0.08,
            pad=0.1,
            shrink=0.8,
            cmap="seismic",
        )
        fig_.suptitle(
            f"Visualization of the original (left) and reconstructed (right) data"
        )
        trainer.logger.experiment.add_figure(
            "Visualization of the original and reconstructed data",
            fig_,
            trainer.current_epoch,
        )
