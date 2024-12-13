import argparse
import math

import numpy as np
import torch
import torchaudio

import config
from dataset import LibriSpeechDataset
from utils import list_type, plot_mels

# TODO: use model definitions from aiayn model.py instead of redoing

class PermuteLayer(torch.nn.Module):
    
    def __init__(self, dims):
        super().__init__()

        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class ConvolutionModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.num_features = config.params['d_encoder']
        self.expansion_factor = 2
        self.expansion = self.num_features * self.expansion_factor

        self.nn = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=self.num_features),
            PermuteLayer(dims=(0, 2, 1)),
            torch.nn.Conv1d(in_channels=self.num_features, out_channels=self.expansion, kernel_size=1, stride=1, padding=0, bias=True),  # point-wise conv
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=config.params['conv_kernel_size'], stride=1, padding=(config.params['conv_kernel_size']) // 2, bias=False),  # depth-wise conv
            torch.nn.BatchNorm1d(num_features=self.num_features),
            torch.nn.SiLU(),  # swish
            torch.nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=1, stride=1, padding=0, bias=True),  # point-wise conv
            torch.nn.Dropout(p=config.p_drop),
            PermuteLayer(dims=(0, 2, 1))
        )

    def forward(self, x):
        x_init = x

        x = self.nn(x)

        return x_init + x  # skip connection


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.scale = 1 / math.sqrt(config.params['d_attn'])

    def forward(self, Q, K, V, attn_mask=None):
        x = Q @ K.transpose(-2, -1)  # matrix dot product, swap dims of value
        x *= self.scale  # prevents small gradients from softmax
        if attn_mask is not None:
            x += attn_mask
        x = torch.nn.functional.softmax(x, dim=-1)
        x = x @ V

        return x


class MultiHeadAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.num_features = config.params['d_encoder']
        self.linear_projs_in = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Linear(in_features=self.num_features, out_features=config.params['d_attn']),  # Q
                torch.nn.Linear(in_features=self.num_features, out_features=config.params['d_attn']),  # K
                torch.nn.Linear(in_features=self.num_features, out_features=config.params['d_attn']),  # V
            ])
            for _ in range(config.params['num_attn_heads'])
        ])
        self.sdpa = ScaledDotProductAttention()
        self.linear_out = torch.nn.Linear(in_features=self.num_features, out_features=self.num_features)

    def forward(self, Q, K, V):
        batch_size, num_timesteps = Q.shape[:2]

        # temp matrices
        Q_all = torch.zeros((batch_size, config.params['num_attn_heads'], num_timesteps, config.params['d_attn']))
        K_all = torch.zeros_like(Q_all)
        V_all = torch.zeros_like(Q_all)

        for i in range(config.params['num_attn_heads']):
            Q_all[:, i, ...] = self.linear_projs_in[i][0](Q)
            K_all[:, i, ...] = self.linear_projs_in[i][1](K)
            V_all[:, i, ...] = self.linear_projs_in[i][2](V)

        x = self.sdpa(Q_all, K_all, V_all)  # parallel
        x = x.view(batch_size, num_timesteps, config.params['d_attn'] * config.params['num_attn_heads'])  # concat from attn heads
        x = self.linear_out(x)
        x = torch.nn.functional.dropout(x, p=config.p_drop, training=True)

        return x
        

class PositionalEncoding(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.pe = torch.zeros((config.max_len, config.params['d_encoder']))
        
        for i in range(config.max_len):
            for j in range(0, config.params['d_encoder'], 2):
                div_term = 1 / (10_000 ** (j / config.params['d_encoder']))
                self.pe[i][j] = math.sin(i * div_term)
                self.pe[i][j + 1] = math.cos(i * div_term)

    def forward(self, x):
        x += self.pe[:x.shape[1]]  # inject positional information

        return x


class MultiHeadSelfAttentionModule(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=config.params['d_encoder'])
        self.pe_layer = PositionalEncoding()
        self.mha = MultiHeadAttention()

    def forward(self, x):
        x_init = x    

        x = self.layer_norm(x)
        x = self.pe_layer(x)  # relative sinusoidal positional encoding
        x = self.mha(x, x, x)

        return x_init + x  # skip connection


class FeedForwardModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.num_features = config.params['d_encoder']
        self.expansion_factor = 4
        self.expansion = self.num_features * self.expansion_factor

        self.nn = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=self.num_features),
            torch.nn.Linear(in_features=self.num_features, out_features=self.expansion),
            torch.nn.SiLU(),  # swish
            torch.nn.Dropout(p=config.p_drop),
            torch.nn.Linear(in_features=self.expansion, out_features=self.num_features),
            torch.nn.Dropout(p=config.p_drop)
        )

    def forward(self, x):
        x_init = x

        x = self.nn(x)

        return x_init + x


class ConformerBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.ff_1 = FeedForwardModule()
        self.mhsa = MultiHeadSelfAttentionModule()
        self.conv = ConvolutionModule()
        self.ff_2 = FeedForwardModule()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=config.params['d_encoder'])

    def forward(self, x):
        x_init = x

        x = x_init + (0.5 * self.ff_1(x))  # half-step ff + skip connection
        x_init = x

        x = x_init + self.mhsa(x)
        x_init = x

        x = x_init + self.conv(x)
        x_init = x

        x = x_init + (0.5 * self.ff_2(x))

        x = self.layer_norm(x)

        return x
        

class SpecAug(torch.nn.Module):
    # https://pytorch.org/audio/master/tutorials/audio_feature_augmentation_tutorial.html#specaugment

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug
        self.warp = torchaudio.transforms.TimeStretch(fixed_rate=None, n_freq=config.num_mels)  # stretch in the timestep dimension
        self.time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=None, p=config.max_time_mask_ratio)] * config.num_time_masks  # add blocks of masks to time dimension
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=config.frequency_mask)  # add blocks of masks to freq dimension

    def forward(self, x):
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


class Encoder(torch.nn.Module):

    def __init__(self, debug=False):
        super().__init__()

        self.spec_aug = SpecAug(debug=debug)
        self.linear = torch.nn.Linear(in_features=config.d_features, out_features=config.params['d_encoder'])
        self.dropout = torch.nn.Dropout(p=config.p_drop) 
        self.blocks = torch.nn.Sequential(*[ConformerBlock() for _ in range(config.params['num_encoder_layers'])])

    def forward(self, x, training=False):
        if training:
            x = self.spec_aug(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.blocks(x)

        return x


class Decoder(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.lstm = torch.nn.LSTM(
            input_size=config.params['d_encoder'],
            hidden_size=config.params['d_decoder'],
            num_layers=config.params['num_decoder_layers'],
            bias=True,
            bidirectional=False
        )
        self.linear_out = torch.nn.Linear(in_features=config.params['d_decoder'], out_features=num_classes)
        self.activation_1 = torch.nn.LogSoftmax(dim=2)  # training requires log softmax, inf requires softmax
        self.activation_2 = torch.nn.Softmax(dim=2)

    def forward(self, x):
        x, (hidden_state, cell_state) = self.lstm(x)
        x = self.linear_out(x)

        return self.activation_1(x), self.activation_2(x)


class E2E(torch.nn.Module):
    
    def __init__(self, num_classes, debug=False):
        super().__init__()

        print(f'Creating model ({config.model_size})...', end='')

        self.encoder = Encoder(debug=debug)
        self.decoder = Decoder(num_classes=num_classes)

        print(f'{self.num_params} million total params')

    @property
    def num_encoder_params(self) -> int:
        return num_params(self.encoder)

    @property
    def num_decoder_params(self) -> int:
        return num_params(self.decoder)

    @property
    def num_params(self) -> int:
        return self.num_encoder_params + self.num_decoder_params

    def forward(self, x, training=False):
        encoder_out = self.encoder(x, training=training)
        
        return self.decoder(encoder_out)
    

def num_params(model: torch.nn.Module) -> float:
    num_params = sum(p.numel() for p in model.parameters())

    return round(num_params / 1_000_000, 1)


def main(args) -> None:
    if args.dataset_path:
        dataset = LibriSpeechDataset(path=args.dataset_path, sets=args.sets)
        x, _ = dataset[0]
        x = x.unsqueeze(0)
    else:
        batch_size, num_timesteps = 4, 100
        x = torch.rand((batch_size, num_timesteps, config.d_features))

    model = E2E(debug=args.debug)
    print(f'Num Encoder params: {model.num_encoder_params} million')
    print(f'Num Decoder params: {model.num_decoder_params} million')
    
    print(f'Input: {x.shape}')
    output = model(x)
    print(f'Output: {output.shape}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path')
    parser.add_argument('--sets', type=list_type)
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
