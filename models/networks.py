import torch.nn as nn

from models.encoders import VGG11Encoder, AlexNetEncoder
from models.resnets import resnet50


class LPLNet(nn.Module):
    def __init__(self, encoder='vgg', pretrained=False, train_end_to_end=False, projector_mlp=False, projection_size=256, mlp_hidden_size=2048, base_image_size=32, no_biases=False):
        """
        :param resnet_encoder (bool): Use Resnet-50 instead of small VGG arch
        :param pretrained (bool): Use a pretrained resnet
        :param train_end_to_end (bool): Enable backprop between conv blocks
        :param projector_mlp (bool): Whether to project representations through an MLP before calculating loss
        :param projection_size (int): Only used when projection mlp is enabled
        :param hidden_layer_size (int): Only used when projection mlp is enabled
        :param base_image_size (int): input image size (eg. 32 for cifar, 96 for stl10)
        """
        super(LPLNet, self).__init__()

        self.resnet_encoder = encoder == 'resnet'
        # Encoder
        if encoder == 'resnet':
            self.encoder = resnet50(pretrained=pretrained)
            self.pooler = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_size = 2048
        elif encoder == 'alexnet':
            self.encoder = AlexNetEncoder(train_end_to_end=train_end_to_end,
                                    projector_mlp=projector_mlp,
                                    hidden_layer_size=mlp_hidden_size,
                                    projection_size=projection_size,
                                    base_image_size=base_image_size,
                                    no_biases=no_biases)
            self.feature_size = self.encoder.channel_sizes[-1]
        elif encoder == 'vgg':
            self.encoder = VGG11Encoder(train_end_to_end=train_end_to_end,
                                    projector_mlp=projector_mlp,
                                    hidden_layer_size=mlp_hidden_size,
                                    projection_size=projection_size,
                                    base_image_size=base_image_size,
                                    no_biases=no_biases)

            self.feature_size = self.encoder.channel_sizes[-1]

    def forward(self, x):
        encoder_output = self.encoder(x)

        if self.resnet_encoder:
            encoder_output = [self.pooler(z).squeeze() for z in encoder_output]
            return None, None, encoder_output
        else:
            return encoder_output
