import torch

import config


class CustomCollator:
    
    def __init__(self):
        pass

    def __call__(self, batch):
        mels, transcript_targets, transcripts = zip(*batch)

        # get lengths before padding
        input_lengths = torch.Tensor([mel.shape[0] for mel in mels]).int()
        target_lengths = torch.Tensor([transcript_target.shape[0] for transcript_target in transcript_targets]).int()

        mels = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=config.pad_value)  # redundant because mels same length anyway
        transcript_targets = torch.nn.utils.rnn.pad_sequence(transcript_targets, batch_first=True, padding_value=config.blank).int()

        return mels, transcript_targets, transcripts, input_lengths, target_lengths
