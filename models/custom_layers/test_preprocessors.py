import numpy as np


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
