import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
        instance_norm: bool = True,
    ):
        super(DoubleConv, self).__init__()

        norm3d = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d

        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            norm3d(out_channels),
            self.get_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            norm3d(out_channels),
            self.get_activation(activation),
        )

    def get_activation(self, name):
        if name == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(
                f"Unknown activation '{name}'. Supported activations are: 'leaky_relu', 'relu', 'sigmoid'."
            )

    def forward(self, x):
        return self.DoubleConv(x)
    
class EncoderClassifier2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_class: int = 2,
        kernel_size: int = 3,
        activation: str = "leaky_relu",
        current_filters: int = 32,
        instance_norm: bool = True,
        num_pool: int = 3,
    ):
        super(EncoderClassifier2D, self).__init__()

        self.num_pool = num_pool
        self.base_filter = current_filters

        self.encoder_blocks = nn.ModuleList()
        self.maxpool = nn.ModuleList()

        for i in range(num_pool + 1):
            self.encoder_blocks.append(
                DoubleConv(
                    in_channels, current_filters, kernel_size, activation, instance_norm
                )
            )

            in_channels = current_filters

            if i < num_pool:
                self.maxpool.append(nn.MaxPool2d(2))
                current_filters *= 2

        # Output
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.max_channel = self.base_filter* (2**num_pool)
        # print(self.max_channel)
        self.classifier = nn.Linear(self.max_channel, num_class)
        # self.activation = nn.Sigmoid()

    def forward(self, x):

        for i in range(self.num_pool + 1):
            x = self.encoder_blocks[i](x)

            if i < self.num_pool:
                x = self.maxpool[i](
                    x
                )  # module = self.maxpool[i] / x = module.forward(x)

        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x

