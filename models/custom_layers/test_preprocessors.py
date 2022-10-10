import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

try:
    import preprocessors
except:
    from . import preprocessors


def check_normalization(self, model, ds):
    for d, split in zip(ds, ["train", "validation"]):
        d = np.concatenate(list(d), axis=0)
        d_minmax = [np.min(d), np.max(d)]
        d = model.data_preprocessor(d)
        d_minmax.extend([np.min(d), np.max(d)])
        d = model.data_preprocessor(d, normalize=0)
        d_minmax.extend([np.min(d), np.max(d)])
        self.logger.info(
            f"{split} dataset - shape: {d.shape}\n"
            f"before normalization: min: {d_minmax[0]}, max: {d_minmax[1]};\n"
            f"after normalization: min: {d_minmax[2]}, max: {d_minmax[3]};\n"
            f"after denormalization: min: {d_minmax[4]}, max: {d_minmax[5]};\n"
        )
    del d


def plot_standardizer():
    mu = 10
    variance = 2
    sigma = np.sqrt(variance)
    x = np.random.normal(mu, sigma, 1000)
    data_preprocessor = preprocessors.Standardizer(mu, variance)
    y = data_preprocessor(torch.tensor(x), normalize=1)
    x_hat = data_preprocessor(y, normalize=0)
    y = y.numpy()
    x_hat = x_hat.numpy()
    sns.set_style("whitegrid")
    sns.kdeplot(x, linewidth=10, label="x")
    sns.kdeplot(y, linewidth=5, label="y")
    sns.kdeplot(x_hat, linewidth=3, label="x_hat")
    plt.show()


if __name__ == "__main__":
    plot_standardizer()
