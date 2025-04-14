import einops
import torch
import torch.nn as nn

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = nn.BatchNorm2d(out_channels) if norm_cfg else None
        self.act = nn.GELU() if act_cfg else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x


class SAMAggregatorNeck(nn.Module):
    def __init__(
            self,
            in_channels=[768]*16,
            inner_channels=128,
            selected_channels: list=None,
            out_channels=256,
            kernel_size=3,
            stride=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='GELU', inplace=True),
            up_sample_scale=4,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_channels = out_channels
        self.stride = stride
        self.selected_channels = selected_channels
        self.up_sample_scale = up_sample_scale

        self.down_sample_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.down_sample_layers.append(
                nn.Sequential(
                    ConvModule(
                        in_channels[idx],
                        inner_channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    ConvModule(
                        inner_channels,
                        inner_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                )
            )
        self.fusion_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.fusion_layers.append(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        self.up_layers = nn.ModuleList()
        self.up_layers.append(
            nn.Sequential(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )
        self.up_layers.append(
            ConvModule(
                inner_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None
            )
        )

        self.up_sample_layers = nn.ModuleList()
        assert up_sample_scale == 4
        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, inputs):
        inner_states = inputs
        inner_states = [einops.rearrange(inner_states[idx], 'b h w c -> b c h w') for idx in range(len(self.selected_channels))]
        inner_states = [layer(x) for layer, x in zip(self.down_sample_layers, inner_states)]

        x = None
        for inner_state, layer in zip(inner_states, self.fusion_layers):
            if x is not None:
                inner_state = x + inner_state
            x = inner_state + layer(inner_state)
        x = self.up_layers[0](x) + x
        img_feats_0 = self.up_layers[1](x)

        img_feats_1 = self.up_sample_layers[0](img_feats_0) + self.up_sample_layers[1](img_feats_0)

        img_feats_2 = self.up_sample_layers[2](img_feats_1) + self.up_sample_layers[3](img_feats_1)

        return img_feats_2, img_feats_1, img_feats_0

if __name__ == "__main__":
    x = [torch.randn((2, 64, 64, 768)).cuda()] * 3
    model = SAMAggregatorNeck(selected_channels=[6-1, 9-1, 12-1]).cuda()
    out = model(x)
    
    for i in range(len(out)):
        print(out[i].shape)