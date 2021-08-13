import torch
import torch.nn as nn


class ResNet34(nn.Module):
    """
    Resnet34 implementation based on https://arxiv.org/pdf/1512.03385.pdf. Other than the MaxPooling layer at the start and the AvgPooling in the fully connected part, no Pooling layers are used between the resnet blocks. Downscaling is performed by using stride withing the convolutional layers greater than AT THE START of every block. At the same moment (meaning the exact same layer) the output filters quadruples and both the image shape and number of channels (filers) remains constant within every block. As a result, skip connections taking place at the start of each block need the respective identities to be modified accordingly.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.in_channels = in_channels
        self.conv1_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        fix_identity_shape2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self.block(64, 64, fix_identity_shape2, 2),
            self.block(64, 64),
            self.block(64, 64)
        )
        fix_identity_shape3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )
        self.conv3_x = nn.Sequential(
            self.block(64, 128, fix_identity_shape3, 2),
            self.block(128, 128),
            self.block(128, 128),
            self.block(128, 128)
        )
        fix_identity_shape4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )
        self.conv4_x = nn.Sequential(
            self.block(128, 256, fix_identity_shape4, 2),
            self.block(256, 256),
            self.block(256, 256),
            self.block(256, 256),
            self.block(256, 256),
            self.block(256, 256)
        )
        fix_identity_shape5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )
        self.conv5_x = nn.Sequential(
            self.block(256, 512, fix_identity_shape5, 2),
            self.block(512, 512),
            self.block(512, 512)
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # block 1 - (simple conv layer): 3 x 224 x 224 --> 64 x 112 x 112
        x = self.conv1_block(x)

        # block 2 - (resnet block): 64 x 112 x 112 --> 64 x 56 x 56
        x = self.conv2_x(x)

        # block 3 - (resnet block): 64 x 56 x 56 --> 128 x 28 x 28
        x = self.conv3_x(x)

        # block 4 - (resnet block): 128 x 28 x28 --> 256 x 14 x 14
        x = self.conv4_x(x)

        # block 5 - (resnet block): 256 x 14 x 14 --> 512 x 7 x 7
        x = self.conv5_x(x)

        # bloxck 6 - (fully connected): flattens 512 x 7 x 7 --> 25088
        x = self.fc(x)

        return x

    class block(nn.Module):
        def __init__(self, input_channels, output_channels, fix_identity_shape=None, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(output_channels)
            self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(output_channels)
            self.relu = nn.ReLU()
            self.fix_identity_shape = fix_identity_shape

        def forward(self, x):
            identity = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)

            if self.fix_identity_shape is not None:
                identity = self.fix_identity_shape(identity)

            x = x + identity
            x = self.relu(x)

            return x


if __name__ == '__main__':
    model = ResNet34(3, 1000)
    img = torch.randn(3, 224, 224).reshape(1, 3, 224, 224)
    output = model(img)
