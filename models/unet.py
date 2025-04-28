import copy
import math
import torch
import torch.nn as nn

from configs.unet import UNetConfig
from models.auxiliary_models.resblock import ResNetBlock
from models.utils import zero_module
import torch.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, config: UNetConfig):
        super().__init__()
        self.down_blocks = []
        self.config = config
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # we will overwrite config.conditioning_feature_dim if config.conditioning_feature_proj_dim is provided        
        self.input_conditioning_feature_dim = config.conditioning_feature_dim
        if (
            config.conditioning_feature_dim > 0
            and config.conditioning_feature_proj_dim > 0
        ):
            config.conditioning_feature_dim = config.conditioning_feature_proj_dim
            self.lm_proj = nn.Linear(
                self.input_conditioning_feature_dim, config.conditioning_feature_dim
            )

        self.time_emb_dim = (
            config.resolution_channels[0] * 4
            if config.time_emb_dim is None
            else config.time_emb_dim
        )

        half_dim = self.time_emb_dim // 8
        emb = math.log(10000) / half_dim  # make this consistent to the adm unet
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer("t_emb", emb.unsqueeze(0), persistent=False)

        self.temb_layer1 = nn.Linear(self.time_emb_dim // 4, self.time_emb_dim)
        self.temb_layer2 = nn.Linear(self.time_emb_dim, self.time_emb_dim)


        channels = config.resolution_channels[0]
        self.conv_in = nn.Conv2d(
            input_channels, channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        skip_channels = [channels]
        num_resolutions = len(config.resolution_channels)
        self.num_resolutions = num_resolutions

        for i in range(num_resolutions):
            down_resnet_configs = []
            num_resnets_per_resolution = config.num_resnets_per_resolution[i]
            for _ in range(num_resnets_per_resolution):
                resnet_config = copy.copy(config.resnet_config)
                resnet_config.in_channels = channels
                resnet_config.out_channels = config.resolution_channels[i]
                skip_channels.append(resnet_config.out_channels)
                down_resnet_configs.append(resnet_config)
                channels = resnet_config.out_channels

            if i != num_resolutions - 1:
                # no downsampling here, so no skip connections.
                skip_channels.append(resnet_config.out_channels)

            num_attention_layers = self.config.num_attention_layers[i]
            
            self.down_blocks.append(
                ResNetBlock(
                    self.time_emb_dim,
                    num_resnets_per_resolution,
                    num_attention_layers,
                    downsample_output=i != num_resolutions - 1,
                    upsample_output=False,
                    resnet_configs=down_resnet_configs,
                    cond_dim=(
                        config.conditioning_feature_dim
                        if self.config.num_attention_layers[i] > 0
                        else -1
                    ),
                )
            )
            channels = resnet_config.out_channels

        # middle resnets keep the resolution.
        resnet_config = copy.copy(resnet_config)
        resnet_config.in_channels = channels
        resnet_config.out_channels = channels

        if not config.skip_mid_blocks:
            self.mid_blocks = [
                ResNetBlock(
                    self.time_emb_dim,
                    1,
                    True,  # attn
                    False,  # downsample
                    False,  # upsample
                    resnet_configs=[resnet_config],
                    cond_dim=config.conditioning_feature_dim,
                ),
                ResNetBlock(
                    self.time_emb_dim,
                    1,
                    False,  # attn
                    False,  # downsample
                    False,  # upsample
                    resnet_configs=[copy.copy(resnet_config)],
                ),
            ]

        self.up_blocks = []
        for i in reversed(range(num_resolutions)):
            up_resnet_configs = []
            num_resnets_per_resolution = config.num_resnets_per_resolution[i]
            for j in range(num_resnets_per_resolution + 1):
                resnet_config = copy.copy(config.resnet_config)
                resnet_config.in_channels = channels + skip_channels.pop()
                resnet_config.out_channels = config.resolution_channels[i]
                up_resnet_configs.append(resnet_config)
                channels = resnet_config.out_channels

            num_attention_layers = config.num_attention_layers[i]
            
            self.up_blocks.append(
                ResNetBlock(
                    self.time_emb_dim,
                    num_resnets_per_resolution + 1,
                    num_attention_layers,
                    downsample_output=False,
                    upsample_output=i != 0,
                    resnet_configs=up_resnet_configs,
                    cond_dim=(
                        config.conditioning_feature_dim
                        if self.config.num_attention_layers[i] > 0
                        else -1
                    ),
                )
            )
            channels = resnet_config.out_channels

        self.norm_out = nn.GroupNorm(config.resnet_config.num_groups_norm, channels)
        self.conv_out = zero_module(
            nn.Conv2d(channels, output_channels, kernel_size=3, padding=1)
        )

        self.down_blocks = nn.ModuleList(self.down_blocks)
        if not config.skip_mid_blocks:
            self.mid_blocks = nn.ModuleList(self.mid_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)





    def create_time_emb(self, times, ff_layers=None):
        temb = times.view(-1, 1) * self.t_emb
        temb = torch.cat([torch.sin(temb), torch.cos(temb)], dim=1)
        if temb.shape[1] % 2 == 1:
            # zero pad
            temb = torch.cat([temb, torch.zeros(times.shape[0], 1)], dim=1)
        if ff_layers is None:
            layer1, layer2 = self.temb_layer1, self.temb_layer2
        else:
            layer1, layer2 = ff_layers
        temb = layer2(F.silu(layer1(temb)))
        return temb


    def forward_input_layer(self, x_t, normalize=False):
        if isinstance(x_t, list) and len(x_t) == 1:
            x_t = x_t[0]
        if normalize:
            x_t = x_t / x_t.std((1, 2, 3), keepdims=True)
        x = self.conv_in(x_t)
        return x

    def forward_output_layer(self, x):
        x_out = F.silu(self.norm_out(x))
        x_out = self.conv_out(x_out)
        return x_out

    def forward_downsample(self, x, temb, conditioning, cond_mask):
        skip_activations = [x]
        for i, block in enumerate(self.down_blocks):
            if self.config.num_attention_layers[i] > 0:
                x, activations = block(
                    x,
                    temb,
                    return_activations=True,
                    conditioning=conditioning,
                    cond_mask=cond_mask,
                )
            else:
                x, activations = block(x, temb, return_activations=True)
            skip_activations.extend(activations)
        return x, skip_activations

    def forward_upsample(self, x, temb, conditioning, cond_mask, skip_activations):
        num_resolutions = len(self.config.resolution_channels)
        for i, block in enumerate(self.up_blocks):
            ri = num_resolutions - 1 - i
            num_skip = self.config.num_resnets_per_resolution[ri] + 1
            skip_connections = skip_activations[-num_skip:]
            skip_connections.reverse()
            if self.config.num_attention_layers[ri] > 0:
                x = block(
                    x,
                    temb,
                    skip_activations=skip_connections,
                    conditioning=conditioning,
                    cond_mask=cond_mask,
                )
            else:
                x = block(x, temb, skip_activations=skip_connections)
            del skip_activations[-num_skip:]
        return x

    def forward_denoising(
        self, x_t, times, conditioning=None, cond_mask=None
    ):
        # 1. time embedding
        temb = self.create_time_emb(times)

        # 2. input layer
        if self.config.nesting:
            x_t, x_feat = x_t
        x = self.forward_input_layer(x_t)
        
        if self.config.nesting:
            x = x + x_feat

        # 3. downsample blocks
        x, skip_activations = self.forward_downsample(x, temb, conditioning, cond_mask)

        # 4. middle blocks
        if not self.config.skip_mid_blocks:
            x = self.mid_blocks[0](
                x, temb, conditioning=conditioning, cond_mask=cond_mask
            )
            x = self.mid_blocks[1](x, temb)

        # 5. upsample blocks
        x = self.forward_upsample(x, temb, conditioning, cond_mask, skip_activations)

        # 6. output layer
        x_out = self.forward_output_layer(x)
        if self.config.nesting:
            return x_out, x
        return x_out
    
    def forward_conditioning(self, conditioning):
        if self.config.conditioning_feature_proj_dim > 0:
            conditioning = self.lm_proj(conditioning)
        
        return conditioning


    def forward(
        self,
        x_t: torch.Tensor,
        times: torch.Tensor,
        conditioning: torch.Tensor = None,
        cond_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        conditioning = self.forward_conditioning(conditioning)

        return self.forward_denoising(
            x_t, times, conditioning, cond_mask
        )

    @property
    def model_type(self):
        return "unet"
