"""
@authors: A Bhattacharya, et. al
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the models that were used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from ViTsubmodules import *

def refine_inputs(X):

    # fill quaternion rotation if not given
    # make it [1, 0, 0, 0] repeated with numrows = X[0].shape[0]
    if X[2] is None:
        # X[2] = torch.Tensor([1, 0, 0, 0]).float()
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1

    # if input depth images are not of right shape, resize
    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

    return X

class LSTMNetVIT(nn.Module):
    """
    ViT+LSTM Network 
    Num Params: 3,563,663   
    """
    def __init__(self, input_channels=1):
        super().__init__()
        if input_channels not in (1, 2):
            raise ValueError('LSTMNetVIT input_channels must be 1 or 2')
        self.input_channels = input_channels
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(input_channels, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = spectral_norm(nn.Linear(4608, 512))
        self.lstm = (nn.LSTM(input_size=517, hidden_size=128,
                         num_layers=3, dropout=0.1))
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))

        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f'{self.__class__.__name__} expects image shape [T, {self.input_channels}, H, W], '
                f'got {tuple(x.shape)}'
            )
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        if len(X)>3:
            out,h = self.lstm(out, X[3])
        else:
            out,h = self.lstm(out)
        out = self.nn_fc2(out)
        return out, h

class CurrentFrameViTLSTM(LSTMNetVIT):
    """ViT+LSTM policy using only the current depth frame."""

    frame_offset = 0

    def __init__(self):
        super().__init__(input_channels=1)


class PreviousFrameViTLSTM(LSTMNetVIT):
    """ViT+LSTM policy using ``[D_{t-1}, D_t]``."""

    frame_offset = 1

    def __init__(self):
        super().__init__(input_channels=2)


class SecondPreviousFrameViTLSTM(LSTMNetVIT):
    """ViT+LSTM policy using ``[D_{t-2}, D_t]``."""

    frame_offset = 2

    def __init__(self):
        super().__init__(input_channels=2)


# Public mode selection is intentionally keyed by frame offset.  The model
# name and checkpoint prefix remain internal metadata, so training and
# inference cannot drift apart by maintaining separate mappings.
FRAME_MODE_SPECS = {
    0: ('CurrentFrameViTLSTM', 1, 'current_frame_vitlstm'),
    1: ('PreviousFrameViTLSTM', 2, 'previous_frame_vitlstm'),
    2: ('SecondPreviousFrameViTLSTM', 2, 'second_previous_frame_vitlstm'),
}


def frame_mode_spec(offset):
    try:
        return FRAME_MODE_SPECS[int(offset)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f'unsupported offset={offset}; expected one of {sorted(FRAME_MODE_SPECS)}'
        ) from exc

if __name__ == '__main__':
    print("MODEL NUM PARAMS ARE")
    model = CurrentFrameViTLSTM().float()
    print("CurrentFrameViTLSTM: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = PreviousFrameViTLSTM().float()
    print("PreviousFrameViTLSTM: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = SecondPreviousFrameViTLSTM().float()
    print("SecondPreviousFrameViTLSTM: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
