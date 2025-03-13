import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3, 4), keepdim=True)
        out = self.fc1(avg_pool) + self.fc1(max_pool)
        out = F.relu(out)
        out = self.fc2(out)
        return x * torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        return x * attention


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        out = self.sa(out)
        return out + residual


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factors):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factors = upscale_factors
        self.expand_conv = None

    def forward(self, x):
        b, c, h, w, d = x.size()
        a_factor, b_factor, c_factor = self.upscale_factors
        if self.expand_conv is None:
            self.expand_conv = nn.Conv3d(c, c * (a_factor * b_factor * c_factor), kernel_size=1, bias=False).to(x.device)

        x = self.expand_conv(x)
        c_new = c
        x = x.view(b, c_new, a_factor, b_factor, c_factor, h, w, d)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(b, c_new, h * a_factor, w * b_factor, d * c_factor)
        return x


class GaussianSplattingNet(nn.Module):
    def __init__(self, input_channels=1, base_channels=64, num_resblocks=4, upscale_factors=(2, 2, 2)):
        super(GaussianSplattingNet, self).__init__()
        self.initial_conv = nn.Conv3d(input_channels, base_channels, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_resblocks)])
        self.pixel_shuffle = PixelShuffle3D(upscale_factors)
        self.final_conv = nn.Conv3d(base_channels, 11, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.resblocks(x)
        x = self.pixel_shuffle(x)
        x = self.final_conv(x)
        # Ensure constraints on the last dimension
        x[:, 0] = torch.sigmoid(x[:, 0])  # Range (0, 1)
        x[:, 1:5] = F.normalize(x[:, 1:5], dim=1)  # Quaternion normalization
        x[:, 5:8] = F.relu(x[:, 5:8])  # Positive variance
        x[:, 8:11] = torch.sigmoid(x[:, 8:11]) * 0.5  # Range (0, 0.5)
        return x


# Testing the network
if __name__ == "__main__":
    # Create a random input tensor
    input_tensor = torch.randn(2, 1, 16, 16, 16)  # (batch_size, channels, depth, height, width)
    model = GaussianSplattingNet(input_channels=1, base_channels=64, num_resblocks=4, upscale_factors=(2, 3, 4))

    # Forward pass
    output = model(input_tensor)

    # Print output shape
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    print("Output sample:", output[0, :, 0, 0, 0])
