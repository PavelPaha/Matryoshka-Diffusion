from dataclasses import dataclass, field


@dataclass
class ResNetConfig:
    in_channels: int = -1
    out_channels: int = -1
    num_groups_norm: int = 32
    dropout: float = 0.0
    use_attention_ffn: bool = False


@dataclass
class UNetConfig:
    resnet_config: ResNetConfig = field(default_factory=ResNetConfig)

    time_emb_dim: int = field(default=128)

    resolution_channels: list[int] = field(
        default_factory=lambda: [128, 256, 256, 512, 1024],
        metadata={"help": "Number of chanels on each block"},
    )

    num_resnets_per_resolution: list[int] = field(
        default_factory=lambda: [2, 2, 2, 2, 2],
        metadata={"help": "Number of residual used for each resolution"},
    )

    num_attention_layers: list[int] = field(
        default_factory=[0, 0, 1, 1, 0],
        metadata={"help": "Num of attention used for each block"},
    )

    conditioning_feature_dim: int = field(
        default=-1,
        metadata={"help": "Dimensions of conditioning vector for cross-attention"},
    )

    conditioning_feature_proj_dim: int = field(
        default=-1,
        metadata={"help": "If value > 0, lieanrly project the conditioning dimension"},
    )

    skip_mid_blocks: bool = field(default=False)

    nesting: bool = field(default=False)

    def __post_init__(self):
        assert len(self.num_attention_layers) == len(self.resolution_channels)

        assert len(self.num_resnets_per_resolution) == len(self.resolution_channels)


@dataclass
class NestedUNetConfig(UNetConfig):
    inner_config: UNetConfig = field(
        default_factory=lambda: UNetConfig(nesting=True),
        metadata={"help": "inner unet used as middle blocks"},
    )
    skip_mid_blocks: bool = field(default=True)
    skip_inner_unet_input: bool = field(
        default=False,
        metadata={
            "help": "If enabled, the inner unet only received the downsampled image, no features."
        },
    )
    skip_normalization: bool = field(
        default=False,
    )
    freeze_inner_unet: bool = field(default=False)
    interp_conditioning: bool = field(
        default=False,
    )
