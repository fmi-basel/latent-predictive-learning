import torch.nn as nn

from models.encoders import VGG11Encoder


class LPLNet(nn.Module):
    def __init__(self, train_end_to_end=False, projector_mlp=False, projection_size=256, mlp_hidden_size=2048, base_image_size=32, no_biases=False):
        """
        :param train_end_to_end (bool): Enable backprop between conv blocks
        :param projector_mlp (bool): Whether to project representations through an MLP before calculating loss
        :param projection_size (int): Only used when projection mlp is enabled
        :param hidden_layer_size (int): Only used when projection mlp is enabled
        :param base_image_size (int): input image size (eg. 32 for cifar, 96 for stl10)
        """
        super(LPLNet, self).__init__()

        # Encoder
        # Only one encoder arch provided here, so this class is sort of redundant at the moment
        self.encoder = VGG11Encoder(train_end_to_end=train_end_to_end,
                                    projector_mlp=projector_mlp,
                                    hidden_layer_size=mlp_hidden_size,
                                    projection_size=projection_size,
                                    base_image_size=base_image_size,
                                    no_biases=no_biases)

        self.feature_size = self.encoder.channel_sizes[-1]

    def forward(self, x):
        encoder_output = self.encoder(x)
        return encoder_output
