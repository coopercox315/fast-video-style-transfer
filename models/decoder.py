import torch
import torch.nn as nn
import torch.nn.functional as F

class CompoundDecoder(nn.Module):
    """
    Decoder that merges two feature maps (e.g from low and high scales) into a single final image.
    Each scale is partially decoded into a more common channel dimension (mid_ch), then combined. 
    """
    def __init__(self, low_in_ch=128, high_in_ch=512, mid_ch=64):
        super().__init__()

        #small decoder for low-scale features
        self.low_decoder = nn.Sequential(
            nn.Conv2d(low_in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        #small decoder for high-scale features
        self.high_decoder = nn.Sequential(
            nn.Conv2d(high_in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            #no upsampling here as we'll manually upsample to match the low-scale size.
        )

        #final decoder to merge low and high scale features, then upsample
        self.merge_decoder = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(mid_ch, 3, kernel_size=3, stride=1, padding=1),
            #output is (batch_size, 3, height, width) (final RGB image)
        )
    
    def forward(self, decorated_low, decorated_high):
        '''
        Forward pass:

        - decorated_low: (batch_size, low_in_ch, H_low, W_low)
        - decorated_high: (batch_size, high_in_ch, H_high, W_high)
        Returns:
        (batch_size, 3, H_final, W_final) (final stylized image)
        '''
        
        #1. partially decode each scale
        decoded_low = self.low_decoder(decorated_low) # -> (B, mid_ch, H2, W2)
        decoded_high = self.high_decoder(decorated_high) # -> (B, mid_ch, H3, W3)

        #2. upsample decoded_high so its spatial size matches decoded_low
        decoded_high_up = F.interpolate(
            decoded_high,
            size=(decoded_low.size(2), decoded_low.size(3)),
            mode='nearest'
        )
        #now decoded_low.shape == decoded_high_up.shape == (B, mid_ch, H2, W2)

        #3. concatenate the two decoded features => (B, mid_ch*2, H2, W2)
        merged = torch.cat((decoded_low, decoded_high_up), dim=1)

        #4. merge the two features and upsample to final size => (B, 3, ~H, ~W) 
        out = self.merge_decoder(merged)
        return out