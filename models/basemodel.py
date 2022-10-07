import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import torch
import torch.nn as nn
import numpy as np
import abc

# from .custom_layers import cus_layers


class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def summarize_model(self):
        total_params = sum(param.numel() for param in self.parameters())

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        totalParams = trainable_params + non_trainable_params
        print(
            f"\n================================================================="
            f"\n                     {self.name} Summary"
            f"\nTotal params: {totalParams:,}"
            f"\nTrainable params: {trainable_params:,}"
            f"\nNon-trainable params: {non_trainable_params:,}"
            f"\n_________________________________________________________________\n"
        )


class SimpleClassifier(BaseModel):
    def __init__(self, num_inputs, num_hidden, num_outputs, name="SimpleClassifier"):
        super().__init__(name=name)
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.bn = nn.BatchNorm1d(num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.bn(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    # Printing a module shows all its submodules
    print(model)
    model.summarize_model()

# class BaseModel(tf.keras.Model):
#     @abc.abstractmethod
#     def __init__(self, in_shape, name=None, **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.in_shape = in_shape

#     @abc.abstractmethod
#     def model(self):
#         x = tf.keras.Input(shape=self.in_shape)
#         return tf.keras.Model(inputs=x, outputs=self.__call__(x), name=self.name)

#     @abc.abstractmethod
#     def summarize_model(self):


#         trainableParams = int(
#             np.sum([np.prod(v.get_shape()) for v in self.trainable_weights])
#         )
#         nonTrainableParams = int(
#             np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights])
#         )
#         totalParams = trainableParams + nonTrainableParams
#         print(
#             f"\n================================================================="
#             f"\n                     {self.name} Summary"
#             f"\nTotal params: {totalParams:,}"
#             f"\nTrainable params: {trainableParams:,}"
#             f"\nNon-trainable params: {nonTrainableParams:,}"
#             f"\n_________________________________________________________________\n"
#         )

#     @abc.abstractmethod
#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "in_shape": self.in_shape,
#             }
#         )
#         return config


# class Encoder(BaseModel):
#     """Encoder"""

#     def __init__(
#         self, in_shape, latent_dim, num_channels, num_conv_layers, name=None, **kwargs
#     ):
#         super().__init__(in_shape=in_shape, name=name, **kwargs)
#         self.in_shape = in_shape
#         self.latent_dim = latent_dim
#         self.num_channels = num_channels
#         self.num_conv_layers = num_conv_layers
#         self.cells = []
#         intial_dim = num_channels // (2**2)

#         first_cell = tf.keras.layers.Conv2D(
#             intial_dim, kernel_size=1, strides=1, padding="same", name="enc_block0"
#         )

#         cells = []
#         for i in range(num_conv_layers):
#             cell = cus_layers.DownSamplingResBlock2D(
#                 num_channels=num_channels,
#                 kernel_size=3,
#                 strides=2,
#                 name=f"enc_block{i+1}",
#             )
#             cells.append(cell)

#         last_cell = tf.keras.layers.Conv2D(
#             latent_dim, kernel_size=1, strides=1, padding="same", name="enc_block_final"
#         )

#         self.cells = [first_cell, *cells, last_cell]

#     def __call__(self, x):
#         for cell in self.cells:
#             x = cell(x)
#         return x

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "latent_dim": self.latent_dim,
#                 "num_channels": self.num_channels,
#                 "num_conv_layers": self.num_conv_layers,
#             }
#         )
#         return config


# class Decoder(BaseModel):
#     """Decoder"""

#     def __init__(self, in_shape, num_channels, num_conv_layers, name=None, **kwargs):
#         super().__init__(in_shape=in_shape, name=name, **kwargs)
#         self.in_shape = in_shape
#         self.num_channels = num_channels
#         self.num_conv_layers = num_conv_layers
#         latent_dim = in_shape[-1]

#         first_cell = tf.keras.layers.Conv2DTranspose(
#             latent_dim, kernel_size=1, strides=1, padding="same", name="dec_block0"
#         )

#         cells = []
#         for i in range(num_conv_layers):
#             cell = cus_layers.UpSamplingResBlock2D(
#                 num_channels=num_channels,
#                 kernel_size=4,
#                 strides=2,
#                 name=f"dec_block{i+1}",
#             )
#             cells.append(cell)

#         last_cell = tf.keras.layers.Conv2DTranspose(
#             1, kernel_size=1, strides=1, padding="same", name="dec_block_final"
#         )

#         self.cells = [first_cell, *cells, last_cell]

#     def __call__(self, x):
#         for cell in self.cells:
#             x = cell(x)
#         return x

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "num_channels": self.num_channels,
#                 "num_conv_layers": self.num_conv_layers,
#             }
#         )
#         return config
