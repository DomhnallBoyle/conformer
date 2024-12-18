import numpy as np
import torch
import torchaudio

import config

from utils import plot_mels


class SpecAug:
    # https://pytorch.org/audio/master/tutorials/audio_feature_augmentation_tutorial.html#specaugment

    def __init__(self, debug=False):
        self.debug = debug
        self.warp = torchaudio.transforms.TimeStretch(fixed_rate=None, n_freq=config.num_mels)  # stretch in the timestep dimension
        self.time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=None, p=config.max_time_mask_ratio)] * config.num_time_masks  # add blocks of masks to time dimension
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=config.frequency_mask)  # add blocks of masks to freq dimension

    def run(self, x):
        num_timesteps = x.shape[1]
        x = x.permute(0, 2, 1)  # requires [B, C, T]

        if self.debug:
            mels = [x[0]]

        # TODO: not sure if casting to complex and back is correct, but complex dtype is required for TimeStretch
        # use a random warping rate for each iteration
        x = x.type(torch.complex64)
        x = self.warp(x, overriding_rate=np.random.uniform(config.min_warp_rate, config.max_warp_rate))
        x = x.type(torch.float32)

        # update max possible length of time mask to utterance length
        for time_mask in self.time_masks:
            time_mask.mask_param = num_timesteps
            x = time_mask(x)

        x = self.freq_mask(x)

        if self.debug:
            mels += [x[0]]
            plot_mels(mels, ['Before SpecAug', 'After SpecAug'])

        return x.permute(0, 2, 1)


class CustomCollator:
    
    def __init__(self, spec_aug=False, debug=False):
        self.spec_aug = SpecAug(debug=debug) if spec_aug else None

    def __call__(self, batch):
        audios, mels, transcripts, transcript_targets = zip(*batch)

        mels = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=config.pad_value)  # redundant because mels same length anyway

        # run Spec Augment if applicable
        if self.spec_aug:
            mels = self.spec_aug.run(mels)

        # get lengths before padding
        input_lengths = torch.Tensor([mel.shape[0] for mel in mels]).int()
        target_lengths = torch.Tensor([transcript_target.shape[0] for transcript_target in transcript_targets]).int()

        audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=config.pad_value)
        transcript_targets = torch.nn.utils.rnn.pad_sequence(transcript_targets, batch_first=True, padding_value=config.blank).int()

        return audios, mels, transcripts, transcript_targets, input_lengths, target_lengths
