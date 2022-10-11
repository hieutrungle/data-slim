import unittest
import torch
import patcher as pch
from PIL import Image
import numpy as np
import requests
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class test_patcher_data3d(unittest.TestCase):
    def test_random(self):
        patch_size = 4
        patcher = pch.Patcher3d(patch_size)

        input_size = (2, 64, 8, 8, 8)
        x = torch.randn(input_size)
        original_shape = x.shape[1:]
        patches = patcher(x)
        inverse_patcher = pch.InversePatcher3d(patch_size, original_shape)
        inverse_patches = inverse_patcher(patches)
        self.assertEqual(
            torch.equal(inverse_patches, x),
            True,
            f"Inverse patches should be equal to original data. "
            f"Inverse patches shape: {inverse_patches.shape};"
            f"Original data shape: {x.shape}",
        )


class test_patcher_data2d(unittest.TestCase):
    def test_random(self):
        patch_size = 2
        patcher = pch.Patcher2d(patch_size)

        input_size = (2, 64, 16, 16)
        x = torch.randn(input_size)
        original_shape = x.shape[1:]
        patches = patcher(x)
        inverse_patcher = pch.InversePatcher2d(patch_size, original_shape)
        inverse_patches = inverse_patcher(patches)
        self.assertEqual(
            torch.equal(inverse_patches, x),
            True,
            f"Inverse patches should be equal to original data. "
            f"Inverse patches shape: {inverse_patches.shape};"
            f"Original data shape: {x.shape}",
        )


if __name__ == "__main__":
    # setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
        ]
    )

    img = Image.open(
        requests.get(
            "https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png",
            stream=True,
        ).raw
    )

    img = transform(img)[
        None,
    ]
    print("Image shape:", img.shape)
    img_channel_last = img.permute(0, 2, 3, 1).numpy()
    print("Image with channel last format shape:", img_channel_last.shape)

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(img_channel_last[0])

    patcher = pch.Patcher2d(64)
    patches = patcher(img)
    patches = patches.reshape(-1, 64, 64, 3)
    patches = patches.numpy()
    print(f"patches shape: {patches.shape}")

    min_value = np.min(img_channel_last)
    max_value = np.max(img_channel_last)
    # print(f"min_value: {min_value}; max_value: {max_value}")

    fig = plt.figure(figsize=(10, 12))
    axes = fig.subplots(nrows=4, ncols=4, sharey=True)
    for i, (ax, patch) in enumerate(zip(axes.flat, patches)):
        im = ax.imshow(patch, vmin=min_value, vmax=max_value, cmap="seismic")
        ax.set_adjustable("box")
        ax.axis("tight")
        ax.axis("off")
        ax.autoscale(False)
    # fig.colorbar(
    #     im,
    #     ax=axes,
    #     location="bottom",
    #     fraction=0.08,
    #     pad=0.1,
    #     shrink=0.8,
    #     cmap="seismic",
    # )
    fig.suptitle(f"Visualization of the original image and its patches")

    plt.show()

    unittest.main()
