import torch
from torch import nn, Tensor
from utils import logger, utils
from models import basemodel
from models.custom_layers import (
    vector_quantizer,
    preprocessors,
    cus_blocks,
    cus_layers,
    gdn,
)
from torchinfo import summary
from functools import partial
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.models._meta import _IMAGENET_CATEGORIES
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math
from dataclasses import dataclass
from torchvision.models._utils import _make_divisible
import warnings
import copy
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import (
    _make_divisible,
    _ovewrite_named_param,
    handle_legacy_interface,
)
import sys


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(
        channels: int, width_mult: float, min_value: Optional[int] = None
    ) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class PreBlockConfig(_MBConvConfig):
    # Stores configuration of Pre Processing Block
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = PreBlock
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class PostBlockConfig(_MBConvConfig):
    # Stores configuration of Post Processing Block
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = PostBlock
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class MBConvConfig(_MBConvConfig):
    # Stores configuration of Encoder MBConv
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = EncMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores configuration of Encoder Fused MBConv
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = EncFusedMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class DecMBConvConfig(_MBConvConfig):
    # Stores configuration of Decoder MBConv
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = DecMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class DecFusedMBConvConfig(_MBConvConfig):
    # Stores configuration of Decoder Fused MBConv
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = DecFusedMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class PreBlock(nn.Module):
    """Preprocessing block for the encoder."""

    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module] = None,
        name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self._conv0 = cus_layers.Conv2dSame(
            cnf.input_channels, cnf.out_channels, kernel_size=1
        )
        self.act0 = nn.GELU()
        self.act1 = nn.GELU()
        self._conv1 = cus_layers.Conv2dSame(
            cnf.out_channels,
            cnf.out_channels,
            kernel_size=cnf.kernel,
            stride=1,
        )
        self._conv2 = cus_layers.Conv2dSame(
            cnf.out_channels,
            cnf.out_channels,
            kernel_size=cnf.kernel,
            stride=1,
        )

    def forward(self, x):
        x = self._conv0(x)
        x = self.act0(x)
        x = self._conv1(x)
        x = self.act1(x)
        x = self._conv2(x)
        return x


class PostBlock(nn.Module):
    """Postprocessing block for the decoder"""

    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module] = None,
        name="PostBlock",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.act0 = nn.GELU()
        self.act1 = nn.GELU()
        self._conv1 = nn.ConvTranspose2d(
            cnf.input_channels,
            cnf.input_channels,
            kernel_size=cnf.kernel,
            padding=max(1, cnf.kernel // 2),
            stride=1,
        )
        self._conv2 = nn.ConvTranspose2d(
            cnf.input_channels,
            cnf.out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        x = self.act0(x)
        x = self._conv1(x)
        x = self.act1(x)
        x = self._conv2(x)
        return x


class EncMBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module] = None,
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU
        self.norm_layer = gdn.GDN(in_channels=cnf.input_channels, inverse=False)

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=partial(nn.SiLU, inplace=True),
            )
        )

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        if self.norm_layer is not None:
            input = self.norm_layer(input)
        result = self.block(input)
        if self.use_res_connect:
            # result = self.stochastic_depth(result)
            result += input
        return result


class EncFusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module] = None,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU
        self.norm_layer = gdn.GDN(in_channels=cnf.input_channels, inverse=False)

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        if self.norm_layer is not None:
            input = self.norm_layer(input)
        result = self.block(input)
        if self.use_res_connect:
            # result = self.stochastic_depth(result)
            result += input
        return result


class DecMBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU
        self.norm_layer = gdn.GDN(in_channels=cnf.input_channels, inverse=True)

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                nn.ConvTranspose2d(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    stride=1,
                )
            )
            layers.append(activation_layer())

        # depthwise
        layers.append(
            nn.ConvTranspose2d(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                padding=cnf.kernel // 2,
                output_padding=cnf.stride - 1,
                groups=expanded_channels,
            )
        )
        layers.append(activation_layer(inplace=True))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=partial(nn.SiLU, inplace=True),
            )
        )

        # project
        layers.append(
            nn.ConvTranspose2d(
                expanded_channels, cnf.out_channels, kernel_size=1, stride=1
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        if self.norm_layer is not None:
            input = self.norm_layer(input)
        result = self.block(input)
        if self.use_res_connect:
            # result = self.stochastic_depth(result)
            result += input
        return result


class DecFusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU
        self.norm_layer = gdn.GDN(in_channels=cnf.input_channels, inverse=True)
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                nn.ConvTranspose2d(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    padding=cnf.kernel // 2,
                    output_padding=cnf.stride - 1,
                )
            )
            layers.append(activation_layer())

            # project
            layers.append(
                nn.ConvTranspose2d(
                    expanded_channels, cnf.out_channels, kernel_size=1, stride=1
                )
            )
        else:
            layers.append(
                nn.ConvTranspose2d(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    padding=cnf.kernel // 2,
                    output_padding=cnf.stride - 1,
                )
            )
            layers.append(activation_layer())

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        if self.norm_layer is not None:
            input = self.norm_layer(input)
        result = self.block(input)
        if self.use_res_connect:
            # result = self.stochastic_depth(result)
            result += input
        return result


class Block(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float = 0,
        stochastic_depth_prob: float = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class
        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        layers: List[nn.Module] = []
        # print(f"norm_layer: {norm_layer}")

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                # norm layer
                # if norm_layer is not None:
                #     norm_layer.in_channels = block_cnf.input_channels
                # if isinstance(norm_layer, gdn.GDN):
                #     norm_layer = norm_layer(block_cnf.input_channels)
                # elif isinstance(norm_layer, nn.BatchNorm2d):
                #     norm_layer = norm_layer()
                # elif isinstance(norm_layer, nn.GroupNorm):
                #     norm_layer = norm_layer(8, block_cnf.input_channels)

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _hierarchical_block(
    inverted_residual_settings: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any,
):
    layers = []

    for i, inverted_residual_setting in enumerate(inverted_residual_settings.items()):
        block_name, block_config = inverted_residual_setting
        layers.append(
            Block(
                block_config,
                norm_layer=norm_layer,
                **kwargs,
            )
        )

    # model = nn.Sequential(*layers)

    # if weights is not None:
    #     model.load_state_dict(weights.get_state_dict(progress=progress))

    return layers


def _model_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("hierarchical"):
        inverted_residual_setting = {
            "pre_block": [PreBlockConfig(1, 3, 1, 1, 24, 1)],
            "encoder_0": [
                FusedMBConvConfig(1, 3, 1, 24, 24, 2),
                FusedMBConvConfig(4, 3, 2, 24, 48, 3),
                FusedMBConvConfig(4, 3, 2, 48, 64, 3),
                MBConvConfig(4, 3, 2, 64, 128, 3),
            ],
            "encoder_1": [
                MBConvConfig(6, 3, 1, 128, 160, 2),
                MBConvConfig(6, 3, 2, 160, 256, 2),
            ],
            # "encoder_2": [
            #     MBConvConfig(6, 3, 1, 256, 320, 1),
            #     MBConvConfig(6, 3, 2, 320, 512, 1),
            # ],
            # "decoder_2": [
            #     DecMBConvConfig(6, 3, 1, 512, 320, 1),
            #     DecMBConvConfig(6, 3, 2, 320, 256, 1),
            # ],
            "decoder_1": [
                DecMBConvConfig(6, 3, 2, 256, 160, 2),
                DecMBConvConfig(6, 3, 1, 160, 128, 2),
            ],
            "decoder_0": [
                DecMBConvConfig(4, 3, 2, 128, 64, 3),
                DecFusedMBConvConfig(4, 3, 2, 64, 48, 3),
                DecFusedMBConvConfig(4, 3, 2, 48, 24, 3),
                DecFusedMBConvConfig(1, 3, 1, 24, 24, 2),
            ],
            "post_block": [PostBlockConfig(1, 3, 1, 24, 1, 1)],
        }
        last_channel = 10
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def hierarchical_model(
    *,
    weights: Optional[None] = None,
    progress: bool = True,
    **kwargs: Any,
):
    """
    Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_S_Weights
        :members:
    """
    # weights = EfficientNet_V2_S_Weights.verify(weights)

    inverted_residual_setting, last_channel = _model_conf("hierarchical")
    return _hierarchical_block(
        inverted_residual_setting,
        0.0,
        weights,
        progress,
        **kwargs,
    )


class VQCPVAE(basemodel.BaseModel):
    """Vector-quantized Compression Variational Autoencoder"""

    def __init__(
        self,
        patch_size,
        patch_depth,
        patch_channels,
        pre_num_channels,
        num_channels,
        latent_dim,
        num_embeddings,
        num_residual_blocks,
        num_transformer_blocks,
        num_heads,
        dropout,
        ema_decay,
        commitment_cost,
        model_type="hierarchical",
        name=None,
        **kwargs,
    ):
        super().__init__(model_type=model_type, name=name, **kwargs)
        if patch_depth <= 0:
            self.input_shape = [1, patch_channels, patch_size, patch_size]
        else:
            self.input_shape = [1, patch_channels, patch_size, patch_size, patch_size]
        inverted_residual_settings, last_channel = _model_conf(model_type)
        print(f"len(inverted_residual_settings) = {len(inverted_residual_settings)}")
        self.num_encoding_levels = (len(inverted_residual_settings) - 2) // 2

        data_channels = self.input_shape[1]  # in_shape = (B, C, H, W)
        embedding_dim = latent_dim
        self.pre_num_channels = pre_num_channels
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.num_residual_blocks = num_residual_blocks
        self.vq_weight = 1.0

        self.data_preprocessor = preprocessors.IdentityDataProcessor()

        # Main blocks
        encoding_blocks = _hierarchical_block(
            inverted_residual_settings,
            dropout=0.0,
            weights=None,
            progress=True,
        )
        self.pre_block = encoding_blocks[0]
        self.encoders = nn.ModuleList(encoding_blocks[1 : self.num_encoding_levels + 1])
        self.decoders = encoding_blocks[self.num_encoding_levels + 1 : -1]
        self.decoders.reverse()
        self.decoders = nn.ModuleList(self.decoders)
        self.post_block = encoding_blocks[-1]

        # Quantization blocks and connecting blocks
        self.vq_layers = nn.ModuleList()
        self.first_dec_combineds = nn.ModuleList()
        self.final_conv_encoders = nn.ModuleList()
        self.first_conv_decoders = []
        for i in range(self.num_encoding_levels):
            # Vector Quantizer layers

            if ema_decay > 0.0:
                self.vq_layers.append(
                    vector_quantizer.VectorQuantizerEMA(
                        num_embeddings,
                        embedding_dim,
                        commitment_cost,
                        ema_decay,
                        name=f"vq_{i}",
                    )
                )
            else:
                self.vq_layers.append(
                    vector_quantizer.VectorQuantizer(
                        num_embeddings,
                        embedding_dim,
                        commitment_cost,
                        ema_decay,
                        name=f"vq_{i}",
                    )
                )
            # Final Convolutional Encoder layers
            enc_conf = utils.get_nth_key(inverted_residual_settings, i + 1)
            last_channel_enc = inverted_residual_settings[enc_conf][-1].out_channels
            self.final_conv_encoders.append(
                cus_layers.Conv2dSame(
                    last_channel_enc,
                    latent_dim,
                    kernel_size=5,
                    stride=1,
                )
            )

            # First Convolutional Decoder layers
            dec_conf = utils.get_nth_key(
                inverted_residual_settings, i + self.num_encoding_levels + 1
            )
            first_channel_dec = inverted_residual_settings[dec_conf][0].input_channels
            self.first_conv_decoders.append(
                nn.ConvTranspose2d(
                    latent_dim,
                    first_channel_dec,
                    kernel_size=5,
                    padding=2,
                    stride=1,
                )
            )

            # Combined layers of upper quantization outputs and lower quantization inputs
            if i != self.num_encoding_levels - 1:
                self.first_dec_combineds.append(
                    nn.ConvTranspose2d(
                        latent_dim,
                        latent_dim,
                        kernel_size=5,
                        padding=2,
                        output_padding=1,
                        stride=2,
                    )
                )
        self.first_conv_decoders.reverse()
        self.first_conv_decoders = nn.ModuleList(self.first_conv_decoders)

        # if num_transformer_blocks > 0:
        #     self.forward_attention = basemodel.AttentionEncoder(
        #         channels=latent_dim,
        #         num_heads=num_heads,
        #         num_blocks=num_transformer_blocks,
        #         dropout=dropout,
        #         name=f"forward_attention",
        #     )
        #     self.backward_attention = basemodel.AttentionEncoder(
        #         channels=latent_dim,
        #         num_heads=num_heads,
        #         num_blocks=num_transformer_blocks,
        #         name=f"backward_attention",
        #     )

        # else:
        #     self.forward_attention = basemodel.Indentity(name=f"forward_attention")
        #     self.backward_attention = basemodel.Indentity(name=f"backward_attention")

        logger.log(f"Initialization of {self.name} completed!")

    def _encode(self, x):
        """Encodes data."""
        x = self.data_preprocessor(x, normalize=1)
        x = self.pre_block(x)
        enc_outputs = []
        y = x
        for i in range(self.num_encoding_levels):
            y = self.encoders[i](y)
            y_latent = self.final_conv_encoders[i](y)
            enc_outputs.append(y_latent)
        return enc_outputs

    def _forward_imp(self, x):
        #### loop version
        # encoding
        enc_outputs = self._encode(x)

        # quantization & decoding
        yz = 0
        z_dec = 0
        loss = 0
        for i in reversed(range(self.num_encoding_levels)):
            if i != self.num_encoding_levels - 1:
                yz = self.first_dec_combineds[i - 1](yz)
            y_loss, y_quantized, perplexity, _ = self.vq_layers[i](enc_outputs[i] + yz)
            y_hat = self.first_conv_decoders[i](y_quantized)
            y_hat = self.decoders[i](y_hat + z_dec)
            yz = y_quantized
            z_dec = y_hat
            loss += y_loss

        x_hat = self.post_block(y_hat)
        x_hat = self.data_preprocessor(x_hat, normalize=0)

        return x_hat, loss

    def forward(self, x):
        return self._forward_imp(x)

    def set_standardizer_layer(self, mean, variance, eta=1e-6):
        self.data_preprocessor = preprocessors.Standardizer(
            mean,
            variance,
            eta,
            name="data_processor",
        )

    def compress(self, x):
        """Compresses data."""
        enc_outputs = self._encode(x)

        yz = 0
        y_values = []
        y_shapes = []
        for i in reversed(range(self.num_encoding_levels)):
            if i != self.num_encoding_levels - 1:
                yz = self.first_dec_combineds[i - 1](yz)
            y = enc_outputs[i] + yz
            y = y.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            y_flattened = torch.reshape(y, [-1, self.latent_dim])
            # (BxHxW, 1)
            y_encoding_indices = self.vq_layers[i].get_code_indices(y_flattened)
            # Preserve spatial shapes of both image and latents.
            y_shape = y.shape[1:]

            yz = self.vq_layers[i].get_quantized_from_indices(
                y_encoding_indices, y_shape
            )
            y_values.append(y_encoding_indices)
            y_shapes.append(y_shape)

        y_values.reverse()
        y_shapes.reverse()
        outputs = y_values + y_shapes

        return outputs

    def decompress(self, outputs):
        """Decompresses an image."""
        y_values = outputs[: len(outputs) // 2]
        y_shapes = outputs[len(outputs) // 2 :]
        z_dec = 0
        for i in reversed(range(self.num_encoding_levels)):

            y_quantized = self.vq_layers[i].get_quantized_from_indices(
                y_values[i], y_shapes[i]
            )

            y_hat = self.first_conv_decoders[i](y_quantized)
            y_hat = self.decoders[i](y_hat + z_dec)
            z_dec = y_hat

        x_hat = self.post_block(y_hat)
        x_hat = self.data_preprocessor(x_hat, normalize=0)

        return x_hat


class TmpArgs:
    command = "train"
    data_dir = "../data/tccs/ocean/SST_modified"
    ds_name = "SST"
    model_path = "./saved_models/hierachical-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0"
    use_fp16 = False
    verbose = True
    resume = ""
    iter = -1
    local_test = False
    input_path = ""
    output_path = ""
    patch_size = 64
    patch_depth = -1
    patch_channels = 1
    pre_num_channels = 32
    num_channels = 64
    latent_dim = 128
    num_embeddings = 256
    num_residual_blocks = 3
    num_transformer_blocks = 0
    num_heads = 4
    dropout = 0.0
    ema_decay = 0.99
    commitment_cost = 0.25
    model_type = "hierachical"
    name = "hierachical-hierachical-hierachical--patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0"
    epochs = 50
    lr = 0.0004
    warm_up_portion = 0.15
    weight_decay = 0.0001
    log_interval = 2500
    save_interval = 10
    train_verbose = False
    data_height = 2400
    data_width = 3600
    data_depth = -1
    data_channels = 1
    batch_size = 128
    data_shape = (1, 2400, 3600, 1)
    prefix_folder = "-patch_size_64-pre_num_channels_32-num_channels_64-latent_dim_128-num_embeddings_256-num_residual_blocks_3-num_transformer_blocks_0"


if __name__ == "__main__":
    args = TmpArgs()
    pass
