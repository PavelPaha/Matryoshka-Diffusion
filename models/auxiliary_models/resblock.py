import torch
import torch.nn as nn

from configs.unet import ResNetConfig
from models.auxiliary_models.attention import PixelsAttention
from models.utils import zero_module
import torch.functional as F

class ResNet(nn.Module):
    def __init__(self, time_emb_dim, config: ResNetConfig):
        super().__init__()
        self.config = config
        self.norm1 = nn.GroupNorm(config.num_groups_norm, config.in_channels)
        self.conv1 = nn.Conv2d(
            config.in_channels,
            config.out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.time_layer = nn.Linear(time_emb_dim, config.out_channels * 2)
        self.norm2 = nn.GroupNorm(config.num_groups_norm, config.out_channels)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = zero_module(
            nn.Conv2d(
                config.out_channels,
                config.out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )
        if self.config.out_channels != self.config.in_channels:
            self.conv3 = nn.Conv2d(
                config.in_channels, config.out_channels, kernel_size=1, bias=True
            )

    def forward(self, x, temb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        ta, tb = (
            self.time_layer(F.silu(temb)).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        )

        # if h.size(0) > ta.size(0):  # HACK. repeat to match the shape.
        #     N = h.size(0) // ta.size(0)
        #     ta = einops.repeat(ta, "b c h w -> (b n) c h w", n=N)
        #     tb = einops.repeat(tb, "b c h w -> (b n) c h w", n=N)

        h = F.silu(self.norm2(h) * (1 + ta) + tb)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.config.out_channels != self.config.in_channels:
            x = self.conv3(x)
        return h + x

class ResNetBlock(nn.Module):
    def __init__(
        self,
        time_emb_dim: int,
        num_residual_blocks: int,
        num_attention_layers: int,
        downsample_output: bool,
        upsample_output: bool,
        resnet_configs: list[ResNetConfig],
        cond_dim: int = -1,
    ):
        super().__init__()
        resnets = []
        self.num_residual_blocks = num_residual_blocks
        self.num_attention_layers = num_attention_layers
        self.upsample_output = upsample_output
        self.downsample_output = downsample_output
        
        assert (downsample_output and upsample_output) == False

        for i in range(num_residual_blocks):
            cur_config = resnet_configs[i]
            resnets.append(ResNet(time_emb_dim, cur_config))
        self.resnets = nn.ModuleList(resnets)

        if self.num_attention_layers > 0:
            attn = []
            for i in range(num_residual_blocks):
                for j in range(self.num_attention_layers):
                    attn.append(
                        PixelsAttention(
                            resnet_configs[i].out_channels,
                            cond_dim=cond_dim,
                            use_attention_ffn=resnet_configs[i].use_attention_ffn,
                        )
                    )
            self.attn = nn.ModuleList(attn)

        if self.downsample_output:
            self.resample = nn.Conv2d(
                resnet_configs[-1].out_channels,
                resnet_configs[-1].out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )

        elif self.upsample_output:
            self.resample = nn.Conv2d(
                resnet_configs[-1].out_channels,
                resnet_configs[-1].out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )

    def forward(
        self,
        x,
        temb,
        skip_activations=None,
        return_activations=False,
        conditioning=None,
        cond_mask=None,
    ):
        activations = []

        for i in range(self.num_residual_blocks):
            if skip_activations is not None:
                skip_input = skip_activations.pop(0)
                x = torch.cat((x, skip_input), axis=1)

            x = self.resnets[i](x, temb)

            if self.num_attention_layers > 0:
                L = self.num_attention_layers
                for j in range(L):
                    x = self.attn[i * L + j](x, conditioning, cond_mask)
            
            activations.append(x)

        if self.downsample_output or self.upsample_output:
            if self.upsample_output:
                x = F.interpolate(x.type(torch.float32), scale_factor=2).type(x.dtype)
            x = self.resample(x)
            activations.append(x)

        if not return_activations:
            return x
        return x, activations
