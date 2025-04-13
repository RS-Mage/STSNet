import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_NORM_MOMENTUM = 0.1



class EnhancedSqueezeExcitationLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(EnhancedSqueezeExcitationLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        y1 = self.max_pool(x).view(batch, channels)
        y2 = self.fc(y1).view(batch, channels, 1, 1)
        weights = y + y2
        return x * weights.expand_as(x)


class BottleneckBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion,
                                  momentum=BATCH_NORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out)
        return out


class BottleneckBlock1(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion,
                                  momentum=BATCH_NORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)
        return out


class BottleneckBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion,
                                  momentum=BATCH_NORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)
        return out


class BottleneckBlock3(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion,
                                  momentum=BATCH_NORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)
        return out


class AtrousConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(AtrousConv, self).__init__(*modules)


class AtrousPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(AtrousPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(AtrousSpatialPyramidPooling, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(AtrousConv(in_channels, out_channels, rate))

        modules.append(AtrousPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ChannelAdapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


from timm.models.layers import trunc_normal_
import math


class ChannelAttentionWeights(nn.Module):
    def __init__(self, channels, reduction=1):
        super(ChannelAttentionWeights, self).__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels * 4, self.channels * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels * 4 // reduction, self.channels * 2),
            nn.Sigmoid())
        self.conv1x1 = nn.Conv2d(self.channels, self.channels * 2, kernel_size=1, padding=0, bias=False)

    def forward(self, x1, x2):
        batch, _, height, width = x1.shape
        avg = self.avg_pool(x1)
        max_pool = self.max_pool(x1)
        avg1 = self.avg_pool(x2)
        max1 = self.max_pool(x2)
        shared1 = self.conv1x1(avg) + self.conv1x1(max_pool)
        shared2 = self.conv1x1(avg1) + self.conv1x1(max1)
        avg = shared1.view(batch, self.channels * 2)
        max_pool = shared2.view(batch, self.channels * 2)

        y = torch.cat((avg, max_pool), dim=1)
        y = self.mlp(y).view(batch, self.channels * 2, 1)
        channel_weights = y.reshape(batch, 2, self.channels, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class SpatialAttentionWeights(nn.Module):
    def __init__(self, channels, reduction=1):
        super(SpatialAttentionWeights, self).__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        batch, _, height, width = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(batch, 2, 1, height, width).permute(1, 0, 2, 3, 4)
        return spatial_weights


class FeatureFusionModule(nn.Module):
    def __init__(self, channels, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super(FeatureFusionModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelAttentionWeights(channels=channels, reduction=reduction)
        self.spatial_weights = SpatialAttentionWeights(channels=channels, reduction=reduction)
        self.conv1 = nn.Conv2d(channels * 2, channels, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        t1 = channel_weights[1] * x2
        t2 = channel_weights[1] * x1
        t1 = torch.sigmoid(t1)
        t2 = torch.sigmoid(t2)
        out_x1 = torch.cat((t2, t1), 1)
        out_x1 = self.conv1(out_x1)
        out_x1 = x1 + 0.01 * out_x1
        return out_x1


from Reviewer_3_4.fusion.Time4 import LSTM_UNET


class FusionNet(nn.Module):
    def __init__(self, num_classes=8, backbone='hrnetv2_w32', pretrained=False):
        super(FusionNet, self).__init__()
        num_filters = {
            'hrnetv2_w18': [18, 36, 72, 144],
            'hrnetv2_w32': [32, 64, 128, 256],
            'hrnetv2_w48': [48, 96, 192, 384],
        }[backbone]
        self.depthwise_conv = DepthwiseSeparableConv(249, 249)
        self.conv1 = nn.Conv2d(249, 249, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(249, momentum=BATCH_NORM_MOMENTUM)
        self.conv2 = nn.Conv2d(249, 249, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(249, momentum=BATCH_NORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.se1 = EnhancedSqueezeExcitationLayer(249, 16)

        self.stage1 = BottleneckBlock(249, 249)
        self.stage1_1 = BottleneckBlock(249, 249)

        self.se2 = EnhancedSqueezeExcitationLayer(249, 16)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=249, out_channels=300, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(300, momentum=BATCH_NORM_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stage2 = BottleneckBlock1(300, 300)
        self.stage2_2 = BottleneckBlock1(300, 300)

        self.se3 = EnhancedSqueezeExcitationLayer(300, 16)

        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=300, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, momentum=BATCH_NORM_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        self.stage3 = BottleneckBlock2(512, 512)
        self.stage3_3 = BottleneckBlock2(512, 512)

        self.se4 = EnhancedSqueezeExcitationLayer(512, 16)

        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, momentum=BATCH_NORM_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        self.stage4 = BottleneckBlock3(512, 512)
        self.stage4_4 = BottleneckBlock3(512, 512)
        self.se5 = EnhancedSqueezeExcitationLayer(512, 16)


        self.stage1_1_1 = BottleneckBlock3(249, 512)
        self.stage2_2_2 = BottleneckBlock3(300, 256)
        self.stage3_3_3 = BottleneckBlock3(512, 128)
        self.stage4_4_4 = BottleneckBlock3(512, 128)
        self.adapter1 = ChannelAdapter(300, 249)
        self.adapter2 = ChannelAdapter(512, 300)
        self.adapter3 = ChannelAdapter(512, 256)
        self.adapter4 = ChannelAdapter(1024, 8)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=4, padding=1, dilation=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=8, padding=2, dilation=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, stride=16, padding=2, dilation=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=8, stride=32, padding=2, dilation=1)
        self.channel_transform1 = nn.Conv2d(249, 300, kernel_size=1)
        self.channel_transform2 = nn.Conv2d(249, 512, kernel_size=1)
        self.channel_transform3 = nn.Conv2d(249, 512, kernel_size=1)

        self.depthwise_conv1 = DepthwiseSeparableConv(300, 300)
        self.stage1_depthwise = BottleneckBlock(300, 300)
        self.depthwise_conv2 = DepthwiseSeparableConv(512, 512)
        self.stage2_depthwise = BottleneckBlock(512, 512)
        self.depthwise_conv3 = DepthwiseSeparableConv(512, 512)
        self.stage3_depthwise = BottleneckBlock(512, 512)

        self.temporal_net = LSTM_UNET()
        self.fusion_module = FeatureFusionModule(channels=512, reduction=1)
        self.fusion_module1 = FeatureFusionModule(channels=512, reduction=1)

    def forward(self, x, y):
        xs = self.depthwise_conv(x)

        x = self.conv1(xs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.se1(x)
        x1 = self.stage1(x)
        x1_1 = self.stage1_1(x1)
        x2 = self.se2(x1)

        x2_s_1 = self.maxpool2(xs)
        x2_s_1 = self.channel_transform1(x2_s_1)

        x3_s_1 = self.maxpool3(xs)
        x3_s_1 = self.channel_transform2(x3_s_1)

        x4_s_1 = self.maxpool4(xs)
        x4_s_1 = self.channel_transform3(x4_s_1)

        x3 = self.downsample1(x2)
        x3 = x2_s_1 + x3
        x4 = self.depthwise_conv1(x3)

        x5 = self.se3(x4)
        x5_2 = self.stage2_2(x5)

        x6 = self.downsample2(x5)
        x6 = x3_s_1 + x6
        x7 = self.depthwise_conv2(x6)
        x8 = self.se4(x7)
        x8_2 = self.stage3_3(x8)

        x9 = self.downsample3(x8)
        x9 = x9 + x4_s_1
        x10 = self.depthwise_conv3(x9)
        x11 = self.se5(x10)
        x11_2 = self.stage4_4(x11)

        _, v3 = self.temporal_net(y)
        x11_2 = self.fusion_module1(x11_2, v3)

        x1_r1 = F.interpolate(x5_2, size=(75, 75), mode='bilinear', align_corners=True)
        x1_r1 = self.adapter1(x1_r1)
        x1_r2 = x1_1 + x1_r1

        x2_r1 = F.interpolate(x8_2, size=(38, 38), mode='bilinear', align_corners=True)
        x2_r1 = self.adapter2(x2_r1)
        x2_r2 = x5_2 + x2_r1

        x3_r1 = F.interpolate(x11_2, size=(19, 19), mode='bilinear', align_corners=True)
        x3_r2 = x8_2 + x3_r1

        x1_1_1 = self.stage1_1_1(x1_r2)
        x2_2_2 = self.stage2_2_2(x2_r2)
        x3_3_3 = self.stage3_3_3(x3_r2)
        x4_4_4 = self.stage4_4_4(x11_2)

        x11_a = F.interpolate(x4_4_4, size=(19, 19), mode='bilinear', align_corners=True)
        x11_a = torch.cat((x3_3_3, x11_a), dim=1)
        x11_b = F.interpolate(x11_a, size=(38, 38), mode='bilinear', align_corners=True)
        x11_b = torch.cat((x2_2_2, x11_b), dim=1)
        x11_c = F.interpolate(x11_b, size=(75, 75), mode='bilinear', align_corners=True)
        x11_c = torch.cat((x1_1_1, x11_c), dim=1)
        x1234 = F.interpolate(x11_c, size=(300, 300), mode='bilinear', align_corners=True)
        x1234 = self.adapter4(x1234)

        return x1234


