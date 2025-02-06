import torch
import torch.nn as nn
import torch.nn.functional as F

class FilterPredictor(nn.Module):
    """
    Filter Predictor module which dynamically predicts a linear combination of different channels.
    Given style features of shape (batch_size, num_channels, height, width), the module outputs
    a set of small convolutional filters to be applied to the content features.
    """
    def __init__(self, style_channels, out_channels=64, kernel_size=3):
        super().__init__()
        #reads in style features and outputs a set of filters, then produces weights for a conv layer.

        hidden_dim = 128
        #flatten style features and produce 'out_channels' * 'kernel_size^2' filters
        self.fc = nn.Sequential(
            nn.Linear(style_channels, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_channels * kernel_size * kernel_size)
        )
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, style_features):
        """
        Forward Pass:
        We use adaptive average pooling to flatten style_features to (batch_size, num_channels).
        Then we feed it into the MLP to produce filter params of size (batch_size, out_channels * kernel_size^2).
        These filters reflect an average style signature across the entire image.

        - style_features: style features of shape (batch_size, num_channels, height, width)
        """
        B, C, H, W = style_features.size()

        #1. adaptive pooling to flatten style features to (B, C, 1, 1), then flatten to (B, C)
        style_pooled = F.adaptive_avg_pool2d(style_features, (1, 1)) # -> (B, C, 1, 1)
        style_pooled = style_pooled.view(B, -1) # -> (B, C).

        #2. feed into MLP
        filter_params = self.fc(style_pooled) # -> (B, out_channels * kernel_size^2)

        #3. reshape filter_params to (B, out_channels, kernel_size, kernel_size)
        filter_params =  filter_params.view(B, self.out_channels, self.kernel_size, self.kernel_size)
        return filter_params
    
class CompoundDecorator(nn.Module):
    """
    Compound Decorator module which applies filter predictors at multiple scales (low and high).
    This produces partial styled features at each scale, returned as decorated_low and decorated_high for further merging or decoding.
    """
    def __init__(self, c_channels_1 = 128, c_channels_2 = 512, s_channels_1 = 128, s_channels_2 = 512):
        super().__init__()
        #filter predictor for lower-scale features (e.g. conv2_1 in VGG19)
        self.filter_pred_low = FilterPredictor(s_channels_1, out_channels=c_channels_1, kernel_size=3)
        #filter predictor for higher-scale features (e.g. conv4_1 in VGG19)
        self.filter_pred_high = FilterPredictor(s_channels_2, out_channels=c_channels_2, kernel_size=3)

    def apply_filters(self, content_features, filters):
        """
        Applies the predicted filters to the content features.
        We do a naive per-sample dynamic convolution.
        
        - content_features: content features of shape (batch_size, num_channels, height, width) or (B, C, H, W)
        - filters: filters of shape (batch_size, out_channels, kernel_size, kernel_size) or (B, C_out, kH, kW) predicted from style features
        """
        B, C, H, W = content_features.size()
        B2, C_out, kH, kW = filters.size()

        assert B == B2, "Mismatch in batch sizes of content and style filters."

        styled_features = []
        for i in range(B):
            c_i = content_features[i].unsqueeze(0) # -> (1, C, H, W)
            f_i = filters[i].unsqueeze(1) # -> (C_out, 1, kH, kW)

            #apply filter to content features using group convolution if C_out != C, else depthwise convolution.
            #below 'groups=C_out' means each output channel is treated as a separate group. 
            styled_i = F.conv2d(c_i, f_i, padding=1, groups=C_out) # -> (1, C_out, H, W)
            styled_features.append(styled_i)
        
        #concatenate styled features back to (B, C_out, H, W)
        styled_features = torch.cat(styled_features, dim=0) # -> (B, C_out, H, W)
        return styled_features
    
    def forward(self, cF_low, sF_low, cF_high, sF_high):
        """
        Forward Pass:
        Given content and style features at two scales, apply filter predictors to produce styled features.
        
        - cF_low: content features at lower scale (e.g. conv2_1 in VGG19)
        - sF_low: style features at lower scale
        - cF_high: content features at higher scale (e.g. conv4_1 in VGG19)
        - sF_high: style features at higher scale
        """

        #1. predict filters from style features at each scale
        low_filters = self.filter_pred_low(sF_low) # -> (B, c_channels_1, 3, 3)
        high_filters = self.filter_pred_high(sF_high) # -> (B, c_channels_2, 3, 3)

        #2. apply them to content features at each scale
        decorated_low = self.apply_filters(cF_low, low_filters) # -> (B, c_channels_1, H_low, W_low)
        decorated_high = self.apply_filters(cF_high, high_filters) # -> (B, c_channels_2, H_high, W_high)

        #3. return the styled features at each scale for further merging or decoding
        return decorated_low, decorated_high