import torch.nn as nn
import torch.nn.functional as F
from model.modules import Encoder, Decoder, DoubleConv

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

class AbstractUNet(nn.Module):
    def __init__(self, in_channels, out_channels, basic_module, f_maps=64, layer_order='icr',
                 num_levels=4, testing=False, en_kernel_type='2d', de_kernel_type='3d',
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, features_out=None, drop_rate=0.1, **kwargs):
        super(AbstractUNet, self).__init__()

        self.testing = testing
        self.features_out = features_out
        self.drop_rate = drop_rate

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num,
                                  apply_pooling=False,  # skip pooling in the first encoder
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  conv_kernel_type = en_kernel_type,
                                  padding=conv_padding)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  conv_kernel_type=en_kernel_type,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding)

            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        reversed_f_maps = list(reversed(f_maps))
        decoders_mean = []
        for i in range(len(reversed_f_maps) - 1):
            # if basic_module == DoubleConv:
            #     in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            # else:
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            decoder_mean = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              conv_kernel_type=de_kernel_type,
                              padding=conv_padding)
            decoders_mean.append(decoder_mean)
        self.decoders_mean = nn.ModuleList(decoders_mean)
        
        decoders_var = []
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            decoder_var = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              conv_kernel_type=de_kernel_type,
                              padding=conv_padding)
            decoders_var.append(decoder_var)
        self.decoders_var = nn.ModuleList(decoders_var)

        self.classifier_mean = nn.Conv2d(f_maps[0], out_channels, 3, 1, 1)
        self.classifier_var = nn.Conv2d(f_maps[0], out_channels, 3, 1, 1)
        # regression problem
        self.final_activation = None
    
    def forward(self, x):
        encoders_features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i == len(self.encoders) - 1:
                x = F.dropout(x, p=self.drop_rate, training=True)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        x_mean = x
        x_var = x

        for i, (decoder_mean, decoder_var, encoder_features) in enumerate(zip(self.decoders_mean, self.decoders_var, encoders_features)):
            x_mean = decoder_mean(encoder_features, x_mean)
            x_var = decoder_var(encoder_features, x_var)
            if i == 0:
                x_mean = F.dropout(x_mean, p=self.drop_rate, training=True)
                x_var = F.dropout(x_var, p=self.drop_rate, training=True)
        
        output_mean = self.classifier_mean(x_mean)
        output_var = self.classifier_var(x_var)

        return {'mean': output_mean, 'var': output_var}


class Unet2D(AbstractUNet):
    def __init__(self, in_channels, out_channels, f_maps=16, layer_order='icr',
                 num_levels=4, conv_padding=1, features_out=False, drop_rate=0.1, **kwargs):
        super(Unet2D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     en_kernel_type='2d', de_kernel_type='2d',
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     num_levels=num_levels, conv_padding=conv_padding, features_out=features_out,
                                     drop_rate=drop_rate, **kwargs)