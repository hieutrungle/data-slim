from typing import Callable
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (
            1
            + np.cos(np.pi * (epoch - self.warmup) / (self.max_num_iters - self.warmup))
        )
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


if __name__ == "__main__":

    # Needed for initializing the lr scheduler
    p = torch.nn.Parameter(torch.empty(4, 4))
    optimizer = torch.optim.Adam([p], lr=4e-3)
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=20, max_iters=100)
    for i in range(100):
        lr_scheduler.step()
        print(f"{i} lr: {lr_scheduler.get_lr()}")

    # Plotting
    epochs = list(range(100))
    sns.set()
    plt.figure(figsize=(8, 3))
    plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
    plt.ylabel("Learning rate factor")
    plt.xlabel("Iterations (in batches)")
    plt.title("Cosine Warm-up Learning Rate Scheduler")
    plt.show()
    sns.reset_orig()
