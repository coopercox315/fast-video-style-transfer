import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decorator import CompoundDecorator
from models.decoder import CompoundDecoder

class CompoundTransformerNet(nn.Module):
    """
    A feed-forward style transfer network that:
    1. Encodes content and style at multiple layers.
    2. Applies a compound decorator to style features.
    3. Merges the styled features using a compound decoder.
    """

    def __init__(self, 
                 low_layer=5, #conv2_1 in VGG19
                 high_layer=19, #conv4_1 in VGG19
                 c_channels_low=128, c_channels_high=512, s_channels_low=128, s_channels_high=512, mid_channels=64):
        super().__init__()

        #we'll collect features from these two layers only
        self.low_layer = low_layer
        self.high_layer = high_layer

        #1. encoder to get multi-scale content and style features
        self.encoder = Encoder(layers=(low_layer, high_layer))

        #2. compound decorator to predict filters at each scale
        self.decorator = CompoundDecorator(c_channels_1=c_channels_low, c_channels_2=c_channels_high,
                                           s_channels_1=s_channels_low, s_channels_2=s_channels_high)
        
        #3. compound decoder to merge styled features
        self.decoder = CompoundDecoder(low_in_ch=c_channels_low, high_in_ch=c_channels_high, mid_ch=mid_channels)

    def forward(self, content_img, style_img):
        #1. encode content image
        c_features = self.encoder(content_img) # -> {low_layer: cF_low, high_layer: cF_high}

        #2. encode style image
        s_features = self.encoder(style_img) # -> {low_layer: sF_low, high_layer: sF_high}

        #3. retrieve features at the chosen scales
        cF_low = c_features[self.low_layer] # -> (B, c_channels_low, H_low, W_low)
        cF_high = c_features[self.high_layer] # -> (B, c_channels_high, H_high, W_high)
        sF_low = s_features[self.low_layer] # -> (B, s_channels_low, H_low, W_low)
        sF_high = s_features[self.high_layer] # -> (B, s_channels_high, H_high, W_high)

        #4. apply compound decorator to style features
        decorated_low, decorated_high = self.decorator(cF_low, sF_low, cF_high, sF_high)

        #5. merge styled features using compound decoder to get final image
        out_img = self.decoder(decorated_low, decorated_high)
        return out_img