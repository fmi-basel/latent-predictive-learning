import torch.nn as nn


class MLP(nn.Module):
    """
    Simple module for projection MLPs
    """

    def __init__(self, input_dim=256, hidden_dim=2048, output_dim=256, no_biases=False):
        """
        :param input_dim: number of input units
        :param hidden_dim: number of hidden units
        :param output_dim: number of output units
        """
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=not no_biases),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, output_dim, bias=not no_biases),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    """
    Simple convolutional block with 3x3 conv filters used for VGG-like architectures
    """

    def __init__(self, in_channels, out_channels, pooling=True, kernel_size=3, padding=1, stride=1, groups=1, no_biases=False):
        """
        :param in_channels (int):
        :param out_channels (int):
        :param pooling (bool):
        """

        super(ConvBlock, self).__init__()

        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               bias=not no_biases, groups=groups)

        if pooling:
            pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            pool_layer = nn.Identity()

        self.module = nn.Sequential(conv_layer, nn.ReLU(inplace=True), pool_layer)

    def forward(self, x):
        return self.module(x)


class VGG11Encoder(nn.Module):
    """
    Custom implementation of VGG11 encoder with added support for greedy training
    """

    def __init__(self, train_end_to_end=False, projector_mlp=False, projection_size=256, hidden_layer_size=2048, base_image_size=32, no_biases=False):
        """
        :param train_end_to_end (bool): Enable backprop between conv blocks
        :param projector_mlp (bool): Whether to project representations through an MLP before calculating loss
        :param projection_size (int): Only used when projection mlp is enabled
        :param hidden_layer_size (int): Only used when projection mlp is enabled
        :param base_image_size (int): input image size (eg. 32 for cifar, 96 for stl10)
        """
        super(VGG11Encoder, self).__init__()

        # VGG11 conv layers configuration
        self.channel_sizes = [3, 64, 128, 256, 256, 512, 512, 512, 512]
        pooling = [True, True, False, True, False, True, False, True]

        # Configure end-to-end/layer-local architecture with or without projection MLP(s)
        self.layer_local = not train_end_to_end
        self.num_trainable_hooks = 1 if train_end_to_end else 8
        self.projection_sizes = [projection_size]*self.num_trainable_hooks if projector_mlp else self.channel_sizes[-self.num_trainable_hooks:]

        # Conv Blocks
        self.blocks = nn.ModuleList([])

        # Projector(s) - identity modules by default
        self.projectors = nn.ModuleList([])
        self.flattened_feature_dims = []

        # Pooler
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

        feature_map_size = base_image_size
        for i in range(8):
            if pooling[i]:
                feature_map_size /= 2
            self.blocks.append(ConvBlock(self.channel_sizes[i], self.channel_sizes[i + 1], pooling=pooling[i], no_biases=no_biases))
            input_dim = self.channel_sizes[i + 1]
            # Attach a projector MLP if specified either at every layer for layer-local training or just at the end
            if projector_mlp and (self.layer_local or i==7):
                projector = MLP(input_dim=int(input_dim), hidden_dim=hidden_layer_size, output_dim=projection_size, no_biases=no_biases)
                self.flattened_feature_dims.append(projection_size)
            else:
                projector = nn.Identity()
                self.flattened_feature_dims.append(input_dim*feature_map_size*feature_map_size)
            self.projectors.append(projector)

    def forward(self, x):
        z = []
        feature_maps = []
        for i, block in enumerate(self.blocks):
            x = block(x)

            # For layer-local training, record intermediate feature maps and pooled layer activities z (after projection if specified)
            # Also make sure to detach layer outputs so that gradients are not backproped
            if self.layer_local:
                x_pooled = self.pooler(x).view(x.size(0), -1)
                z.append(self.projectors[i](x_pooled))
                feature_maps.append(x)
                x = x.detach()
                
        x_pooled = self.pooler(x).view(x.size(0), -1)
        
        # Get outputs for end-to-end training
        if not self.layer_local:
            z.append(self.projectors[-1](x_pooled))
            feature_maps.append(x)
        
        return x_pooled, feature_maps, z


class AlexNetEncoder(nn.Module):
    def __init__(self, train_end_to_end=False, projector_mlp=False, projection_size=256, hidden_layer_size=2048, base_image_size=32, no_biases=False, extra_layer=False, dropout=0.5):
        super().__init__()

        self.layer_local = not train_end_to_end

        # Conv bloks
        self.blocks = nn.ModuleList([])
        # Projector(s) - identity modules by default
        self.projectors = nn.ModuleList([])
        self.flattened_feature_dims = []
        # Pooler
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

        self.blocks.append(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ))
        projector = nn.Identity()
        if projector_mlp and self.layer_local:
            projector = MLP(input_dim=64, hidden_dim=hidden_layer_size, output_dim=projection_size, no_biases=no_biases)
        self.projectors.append(projector)

        self.blocks.append(nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ))
        projector = nn.Identity()
        if projector_mlp and self.layer_local:
            projector = MLP(input_dim=192, hidden_dim=hidden_layer_size, output_dim=projection_size, no_biases=no_biases)
        self.projectors.append(projector)

        self.blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ))
        projector = nn.Identity()
        if projector_mlp and self.layer_local:
            projector = MLP(input_dim=384, hidden_dim=hidden_layer_size, output_dim=projection_size, no_biases=no_biases)
        self.projectors.append(projector)

        self.blocks.append(nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ))
        projector = nn.Identity()
        if projector_mlp and self.layer_local:
            projector = MLP(input_dim=256, hidden_dim=hidden_layer_size, output_dim=projection_size, no_biases=no_biases)
        self.projectors.append(projector)

        self.blocks.append(nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ))
        projector = nn.Identity()
        if projector_mlp:
            projector = MLP(input_dim=256, hidden_dim=hidden_layer_size, output_dim=projection_size, no_biases=no_biases)
        self.projectors.append(projector)

        self.blocks.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Dropout(dropout),
            nn.Conv2d(256, 4096, kernel_size=6, padding=0),
            nn.ReLU(inplace=True),
        ))
        projector = nn.Identity()
        if projector_mlp and self.layer_local:
            projector = MLP(input_dim=4096, hidden_dim=hidden_layer_size, output_dim=projection_size, no_biases=no_biases)
        self.projectors.append(projector)

        self.blocks.append(nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        ))
        projector = nn.Identity()
        if projector_mlp:
            projector = MLP(input_dim=4096, hidden_dim=hidden_layer_size, output_dim=projection_size, no_biases=no_biases)
        self.projectors.append(projector)
        
        self.channel_sizes = [64, 192, 384, 256, 256, 4096, 4096]
        # self.channel_sizes = [64, 192, 384, 256, 256]

    def forward(self, x):
        z = []
        feature_maps = []
        for i, block in enumerate(self.blocks):
            x = block(x)

            # For layer-local training, record intermediate feature maps and pooled layer activities z (after projection if specified)
            # Also make sure to detach layer outputs so that gradients are not backproped
            if self.layer_local:
                x_pooled = self.pooler(x).view(x.size(0), -1)
                z.append(self.projectors[i](x_pooled))
                feature_maps.append(x)
                x = x.detach()
                
        x_pooled = self.pooler(x).view(x.size(0), -1)
        
        # Get outputs for end-to-end training
        if not self.layer_local:
            z.append(self.projectors[-1](x_pooled))
            feature_maps.append(x)
        
        return x_pooled, feature_maps, z