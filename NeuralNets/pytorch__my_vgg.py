import torch
import torch.nn as nn
import torchvision


class Vgg(nn.Module):
    """
        An implementation of the VGG, without batch normalization. Reminder: expects a 224x224 RGB image.
    Outputs the raw values for the num of classes. A softmax needs to be added on top of this or a
    loss that contains a softmax needs to be used.
    Paper url: https://arxiv.org/pdf/1409.1556.pdf

    """

    def __init__(self, num_classes, vgg_type='vgg16', custom_vgg=False):
        """
        Parameters
        ----------
        num_classes : Final layer's output (in other words, the number of classes of the problem)
        vgg_type : One of [vgg11, vgg13, vgg16, vgg19]
        custom_vgg : By default it's set to False; assigning a list of custom vgg architecture results in the parameter vgg_type being ignored

        """
        super(MyVgg, self).__init__()
        self.num_classes = num_classes
        if custom_vgg == False:
            vgg_dict = {
                'vgg11': [3, 64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
                'vgg13': [3, 64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
                'vgg16': [3, 64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
                'vgg19': [3, 64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512,
                          512, 512, 'P']
            }
            if vgg_type not in vgg_dict.keys():
                raise ValueError(
                    f"Invalid vgg_type '{vgg_type}'. Expected one of: {[temp for temp in vgg_dict.keys()]}")
            self.architecture = vgg_dict[vgg_type]
        else:
            self.architecture = custom_vgg
        # print(f'Using: {self.architecture}')
        # Converting nn.ModuleList to nn.Sequential, as it generates a 'forward'-related error
        ModuleList = nn.ModuleList(self.create_features_block())
        self.features = nn.Sequential(*ModuleList)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = self.create_classifier_block()

    def forward(self, x):
        out = self.features(x)
        # print(f"Tensor shape output from features' block:\n {out.shape}\n")
        out = self.avgpool(out)
        # print(f"Tensor shape output from avg pool block:\n {out.shape}\n")
        out = nn.Flatten()(out)
        # print(f"Tensor shape after it's been flattened:\n {out.shape}\n")
        out = self.classifier(out)
        # print(f"Tensor shape output from classifier block:\n {out.shape}\n")
        return out

    def create__conv_relu_pool(self, conv_in, conv_out, add_a_maxpool_at_the_start=False, conv_k=(3, 3), conv_s=(1, 1), conv_p=(1, 1)):
        """
        Class function that returns a tuple of (Conv2d, ReLU, optional:MaxPool)

        Returns
        -------
        Tuple of (Conv2d, ReLU, optional:MaxPool)
        """
        conv = nn.Conv2d(in_channels=conv_in, out_channels=conv_out, kernel_size=conv_k, stride=conv_s, padding=conv_p)
        relu = nn.ReLU(inplace=True)

        if add_a_maxpool_at_the_start:
            pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            return pool, conv, relu

        return conv, relu

    def create_features_block(self):
        tuple_of_layers = ()
        for index, num_of_convs in enumerate(self.architecture):
            #             print(f'pos_0 index: {index}')
            #             print(f'pos_1 index: {index+1}')
            #             print(f'pos_2 index: {index+2}\n')
            pos_0 = num_of_convs
            pos_1 = self.architecture[index + 1]
            try:
                pos_2 = self.architecture[index + 2]
            except:
                pos_2 = 'NaN'

            if pos_0 != 'P':
                if pos_1 != 'P':
                    conv = self.create__conv_relu_pool(pos_0, pos_1)
                    tuple_of_layers += conv

                elif pos_1 == 'P':
                    conv = self.create__conv_relu_pool(pos_0, pos_2, add_a_maxpool_at_the_start=True)
                    tuple_of_layers += conv

            # The last part of the feature's block is also a MaxPool layer that needs to be added
            if index == len(self.architecture) - 3 and pos_2 == 'P':
                pool = tuple([nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)])
                tuple_of_layers += pool
                break
        return tuple_of_layers

    def create_classifier_block(self):
        """
        Returns a nn.Sequential fully connected layer
        """
        # The first linear layer needs to align with the last convolution layers output.
        if self.architecture[-1] == 'P':
            conv_layer_output = self.architecture[-2]
        else:
            conv_layer_output = self.architecture[-1]
        seq = nn.Sequential(
            nn.Linear(in_features=conv_layer_output * 7 * 7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
        return seq


if __name__ == "__main__":
    pass