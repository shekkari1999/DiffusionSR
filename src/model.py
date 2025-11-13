
import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
from config import (ds1_in_channels, ds1_out_channels, ds2_in_channels, ds2_out_channels, 
                    ds3_in_channels, ds3_out_channels, ds4_in_channels, ds4_out_channels, 
                    es1_in_channels, es1_out_channels, es2_in_channels, 
                    es2_out_channels, es3_in_channels, es3_out_channels, 
                    es4_in_channels, es4_out_channels, n_groupnorm_groups, shift_size,
                    timestep_embed_dim, initial_conv_out_channels, num_heads, window_size)

def sinusoidal_embedding(timesteps, dim=timestep_embed_dim):
    """
    timesteps: (B,) int64 tensor
    dim: embedding dimension
    returns: (B, dim) tensor

    Just like how positional encodings are there in Transformers
    """
    device = timesteps.device
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
    args = timesteps[:, None] * freq[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

class TimeEmbeddingMLP(nn.Module):
    def __init__(self, emb_dim, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, t_emb):
        return self.mlp(t_emb)   # (B, out_channels)

class InitialConv(nn.Module):
    '''
    Input :  We get LR image from human
    Output:  We send it to Encoder stage 1
    '''
    def __init__(self):
        '''
        Input Shape --> [256 x 256 x 3]
        Ouput Shape --> [256 x 256 x 64]
        '''
        super().__init__()
        self.net = nn.Conv2d(in_channels = 3, out_channels = initial_conv_out_channels, kernel_size = 3, padding = 1) # padding = 1 for keeping the spatial dimension same

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    '''
    Inside the Residual block, channels remain same
    Input  : From previous Encoder stage / Initial Conv
    Output : Downsampling block and we save skip connection for correspoding decoder stage
    '''
    def __init__(self, in_channels, out_channels, sin_embed_dim = timestep_embed_dim):
        '''
        This ResBlock will be used by following inchannels [64, 128, 256, 512]
        This ResBlock will be used by following outchannels [64, 128, 256, 512]
        '''
        super().__init__()

        ## 1st res block
        self.norm1 = nn.GroupNorm(num_groups = n_groupnorm_groups, num_channels = in_channels) ## num_groups 8 are standard it seems
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride=1, padding=1)

        ## timestamp embedding(just creating an MLP layer for conditioning to match channels)
        self.MLP_embed = TimeEmbeddingMLP(sin_embed_dim, out_channels=out_channels)


        ## 2nd res block
        self.norm2 = nn.GroupNorm(num_groups = n_groupnorm_groups, num_channels = out_channels) ## num_groups 8 are standard it seems
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=3, stride=1, padding=1)

        ## skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t): ## t is for timestep embedding
        sin_embed = sinusoidal_embedding(t)
        time_MLP_sin_embed = self.MLP_embed(sin_embed)[:, :, None, None] # along channels dim
        out = self.conv1(self.act1(self.norm1(x))) + time_MLP_sin_embed
        out = self.conv2(self.act2(self.norm2(out)))
        return out + self.skip(x)

    
class Downsample(nn.Module):
    '''
    Meaning: Reducing resolution and increasing channels. This helps in compute and increase the
    spatial region features.
    For example going from 256 x 256 --> 128 x 128 is reducing resolution. At the same time
    increasing channels(feature maps) from 64 --> 128 allows us to zoom out and take a broader look

    Input: From each encoder stage
    Output: To next encoder stage
    '''
    def __init__(self, in_channels, out_channels):
        '''
        Our target is to half the resolution and double the channels
        '''
        super().__init__()
        self.net = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.net(x)

class EncoderStage(nn.Module):
    '''
    Combine ResBlock and downsample here
    x --> resolution
    y --> channels
    Input:  [y, x, x]
    Output: [2y, x/2, x/2]
    '''
    def __init__(self, in_channels, out_channels, downsample = True):
        super().__init__()
        self.res1 = ResidualBlock(in_channels = in_channels, out_channels = out_channels)
        self.res2 = ResidualBlock(in_channels = out_channels, out_channels = out_channels)
        # handling this for the last part of the encoder stage 4
        self.do_downsample = Downsample(out_channels, out_channels * 2) if downsample else nn.Identity()
        self.downsample = self.do_downsample

    def forward(self, x, t):
        out = self.res1(x, t)   ## here out is h + skip(x)
        out_skipconnection = self.res2(out, t)
       # print(f'The shape after Encoder Stage before downsampling is {out.squeeze(dim = 0).shape}')
        out_downsampled = self.downsample(out_skipconnection)
       # print(f'The shape after Encoder Stage after downsampling is {out.squeeze(dim = 0).shape}')
        return out_downsampled, out_skipconnection

class FullEncoderModule(nn.Module):
    '''
    connect all 4 encoder stages(for now)

    '''
    def __init__(self):
        '''
        Passing through Encoder stages 1 by 1
        '''
        super().__init__()
        self.initial_conv = InitialConv()
        self.encoderstage_1 = EncoderStage(es1_in_channels, es1_out_channels)
        self.encoderstage_2 = EncoderStage(es2_in_channels, es2_out_channels)
        self.encoderstage_3 = EncoderStage(es3_in_channels,es3_out_channels)
        self.encoderstage_4 = EncoderStage(es4_in_channels, es4_out_channels, downsample = False)

    def forward(self, x, t):
        out = self.initial_conv(x)
        out_1, skip_1 = self.encoderstage_1(out, t)
        #print(f'The shape after Encoder Stage 1 after downsampling is {out_1.shape}')
        out_2, skip_2 = self.encoderstage_2(out_1, t)
        #print(f'The shape after Encoder Stage 2 after downsampling is {out_2.shape}')
        out_3, skip_3 = self.encoderstage_3(out_2, t)
        #print(f'The shape after Encoder Stage 3 after downsampling is {out_3.shape}')
        out_4, skip_4 = self.encoderstage_4(out_3, t)
        #print(f'The shape after Encoder Stage 4  is {out_4.shape}')
        # i think we should return these for correspoding decoder stages
        return (out_1, skip_1), (out_2, skip_2), (out_3, skip_3), (out_4, skip_4)


class SwinTransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads = num_heads, shift_size=0):
        '''

        As soon as the input image comes (512 x 32 x 32), we divide this into
        16 patches of 512 x 7 x 7

        Each patch is then flattented and it becomes (49 x 512)
        Now think of this as 49 tokens having 512 embedding dim vector. Usually a feature map is representation of pixel in embedding.
        If we say 3 x 4 x 4, that means each pixel is represented in 3 dim vector. Here, 49 pixels/tokens are represented in 512 dim.
        we will have an embedding layer for this.

        '''
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads # Store num_heads for mask generation
        self.attn = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.GELU(),
            nn.Linear(4 * in_channels, in_channels)
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.register_buffer("attn_mask", None, persistent=False)

    def get_windowed_tokens(self, x):
        '''
        In a window, how many pixels/tokens are there and what is its representation in terms of vec
        '''
        B, C, H, W = x.size()
        ws = self.window_size
        # move channel to last dim to make reshaping intuitive
        x = x.permute(0, 2, 3, 1).contiguous()        # (B, H, W, C)

        # reshape into blocks: (B, H//ws, ws, W//ws, ws, C)
        x = x.view(B, H // ws, ws, W // ws, ws, C)

         # reorder to (B, num_h, num_w, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, Nh, Nw, ws, ws, C)

        # merge windows: (B * Nh * Nw, ws * ws, C)
        windows_tokens = x.view(-1, ws * ws, C)

        return windows_tokens

    def window_reverse(self, windows, H, W, B):
        """Merge windows back to feature map."""
        ws = self.window_size
        num_windows_h = H // ws
        num_windows_w = W // ws
        x = windows.view(B, num_windows_h, num_windows_w, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

    def get_attn_mask(self, B, H, W, device):
        ws = self.window_size
        ss = self.shift_size

        if ss == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)

        cnt = 0
        h_slices = [slice(0, -ws),
                    slice(-ws, -ss),
                    slice(-ss, None)]

        w_slices = [slice(0, -ws),
                    slice(-ws, -ss),
                    slice(-ss, None)]

        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.get_windowed_tokens(img_mask.permute(0,3,1,2))
        mask_windows = mask_windows.squeeze(-1)      # [num_windows, ws*ws]

                # mask_windows → [num_windows, 49]

        # Compute base mask (no heads, no batch)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0))
        # shape: [num_windows, 49, 49]

        # Repeat for heads
        attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        # shape: [num_heads * num_windows, 49, 49]

        # Repeat for batch
        attn_mask = attn_mask.repeat(B, 1, 1)
        # shape: [B * num_heads * num_windows, 49, 49]

        return attn_mask

    def forward(self, x):
        # pad the input first(since we are using 7x7 window, we gotta make our image from 32x32 to 35x35)
        '''
        Here there are two types of swin blocks.
        1. Windowed swin block
        2. shifted windowed swin block

        In our code we use both these blocks one after the other. The difference is the first computes local attention, without shifting.
        The second, shifts first, them computes local attention, then shifts it back.
        '''
        shift_size = self.shift_size
        B, C, H, W = x.size()
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_r, 0, pad_b)) # (left, right, top, bottom)
        H_pad, W_pad = x.shape[2], x.shape[3]

        ## check the condition here
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))

        # divide the padded image into windowed tokens (that means [B, C, H, W] --> [B * num of windows, ws * ws, C])
        x_windowed_tokens = self.get_windowed_tokens(x)

        normed = self.norm1(x_windowed_tokens)

          # attention mask
        if self.shift_size > 0:
            if self.attn_mask is None or self.attn_mask.shape[0] != x_windowed_tokens.shape[0] * self.num_heads:
                self.attn_mask = self.get_attn_mask(B, H_pad, W_pad, x.device)

            attn_out, _ = self.attn(normed, normed, normed, attn_mask=self.attn_mask)
        else:
            attn_out, _ = self.attn(normed, normed, normed)
        x_windowed_tokens = x_windowed_tokens + attn_out ## residual connection
        x_mlp = self.mlp(self.norm2(x_windowed_tokens))

        x_windowed_tokens = x_windowed_tokens + x_mlp ## residula connection

        ### Now time to convert our localized attn tokens to spatialized feature map
        x_windowed_tokens = self.window_reverse(x_windowed_tokens, H_pad, W_pad, B)

        # Reverse shift
        if shift_size > 0:
            x = torch.roll(x_windowed_tokens, shifts=(shift_size, shift_size), dims=(2, 3))

        # Crop padding if added
        x = x[:, :, :H, :W]

        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels = es4_out_channels, out_channels = ds1_in_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels = in_channels, out_channels = out_channels)
        self.swintransformer1 = SwinTransformerBlock(in_channels = es4_out_channels, num_heads = num_heads, shift_size=0)
        self.swintransformer2 = SwinTransformerBlock(in_channels = es4_out_channels, num_heads = num_heads, shift_size=shift_size)
        self.res2 = ResidualBlock(in_channels = out_channels, out_channels = out_channels)

    def forward(self, x, t):
        res_out = self.res1(x, t)
        swin_out_1 = self.swintransformer1(res_out)
       # print(f'swin_out_1 shape is {swin_out_1.shape}')
        swin_out_2 = self.swintransformer2(swin_out_1)
       # print(f'swin_out_2 shape is {swin_out_2.shape}')
        res_out_2 = self.res2(swin_out_2, t)
        return res_out_2


class Upsample(nn.Module):
    '''
    Just increases resolution
    Input: From each decoder stage
    Output: To next decoder stage
    '''
    def __init__(self, in_channels, out_channels):
        '''
        Our target is to half the resolution and double the channels
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        )
    def forward(self, x):
        return self.net(x)


class DecoderStage(nn.Module):
    """
    Decoder block:
      - Optional upsample
      - Concatenate skip (channel dimension doubles)
      - Two residual blocks
    """
    def __init__(self, in_channels,skip_channels, out_channels, upsample=True):
        super().__init__()

        # Upsample first, but keep same number of channels
        self.upsample = Upsample(in_channels, in_channels) if upsample else nn.Identity()

        #
        merged_channels = in_channels + skip_channels


        # First ResBlock processes merged tensor
        self.res1 = ResidualBlock(in_channels = merged_channels, out_channels=out_channels)

        # Second ResBlock keeps output channels the same
        self.res2 = ResidualBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x, skip, t):
        """
        x    : (B, C, H, W) decoder input
        skip : (B, C_skip, H, W) encoder skip feature
        """
        x = self.upsample(x)                 # optional upsample
        x = torch.cat([x, skip], dim=1)      # concat along channels
        x = self.res1(x, t)
        x = self.res2(x, t)
        return x

class FullDecoderModule(nn.Module):
    '''
    connect all 4 encoder stages(for now)

    '''
    def __init__(self):
        '''
        Passing through Encoder stages 1 by 1
        '''
        super().__init__()
        self.decoderstage_1 = DecoderStage(in_channels = ds1_in_channels, skip_channels=es4_out_channels, out_channels= ds1_out_channels, upsample=False)
        self.decoderstage_2 = DecoderStage(in_channels = ds2_in_channels, skip_channels=es3_out_channels, out_channels=ds2_out_channels) # Adjusted input channels to include skip connection
        self.decoderstage_3 = DecoderStage(in_channels = ds3_in_channels, skip_channels=es2_out_channels, out_channels=ds3_out_channels) # Adjusted input channels
        self.decoderstage_4 = DecoderStage(in_channels = ds4_in_channels, skip_channels=es1_out_channels, out_channels=ds4_out_channels) # Adjusted input channels
        self.finalconv = nn.Conv2d(in_channels = ds4_out_channels, out_channels = 3, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, bottleneck_output, encoder_outputs, t):#
        # Unpack encoder outputs
        (out_1_enc, skip_1), (out_2_enc, skip_2), (out_3_enc, skip_3), (out_4_enc, skip_4) = encoder_outputs

        # Decoder stages, passing skip connections
        out_1_dec  = self.decoderstage_1(bottleneck_output, skip_4, t) # First decoder stage uses the bottleneck output last encoder output
        #print(f'The shape after Decoder Stage 1 is {out_1_dec.shape}')
        out_2_dec  = self.decoderstage_2(out_1_dec, skip_3, t) # Subsequent stages use previous decoder output and corresponding encoder skip
        #print(f'The shape after Decoder Stage 2 after upsampling is {out_2_dec.shape}')
        out_3_dec  = self.decoderstage_3(out_2_dec, skip_2, t)
        #print(f'The shape after Decoder Stage 3 after upsampling is {out_3_dec.shape}')
        out_4_dec  = self.decoderstage_4(out_3_dec, skip_1, t)
        #print(f'The shape after Encoder Stage 4 after upsampling is {out_4_dec.shape}')
        final_out = self.finalconv(out_4_dec)
        #print(f'The shape after final conv is {final_out.shape}')

        return final_out

class FullUNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = FullEncoderModule()
        self.bottleneck = Bottleneck()
        self.dec = FullDecoderModule()
    def forward(self, x, t):
        encoder_outputs = self.enc(x, t) ## with time stamp
        (out_1_enc, skip_1), (out_2_enc, skip_2), (out_3_enc, skip_3), (out_4_enc, skip_4) = encoder_outputs
        bottle_neck_output = self.bottleneck(out_4_enc, t)
        out = self.dec(bottle_neck_output, encoder_outputs, t)
        #out = self.dec(encoder_outputs, t)
        return out

