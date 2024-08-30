import math

import torch

import config
from utils import spec_augment


class ConvolutionModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.expansion_factor = 2
        self.expansion = config.d_features * self.expansion_factor

        self.nn = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=config.d_features),
            torch.nn.Conv1d(in_channels=config.d_features, out_channels=self.expansion, kernel_size=1),  # point-wise conv
            torch.nn.GLU(),
            torch.nn.Conv1d(in_channels=self.expansion, out_channels=self.expansion, kernel_size=config.params['conv_kernel_size']),  # depth-wise conv
            torch.nn.BatchNorm1d(num_features=self.expansion),
            torch.nn.SiLU(),  # swish
            torch.nn.Conv1d(in_channels=self.expansion, out_channels=config.d_features, kernel_size=1),  # point-wise conv
            torch.nn.Dropout()
        )

    def forward(self, x):
        x_init = x

        x = self.nn(x)

        return x_init + x  # skip connection


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.scale = 1 / math.sqrt(config.d_attn)

    def forward(self, Q, K, V):
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

        self.linear_projs_in = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Linear(in_features=config.params['d_encoder'], out_features=config.d_attn),  # Q
                torch.nn.Linear(in_features=config.params['d_encoder'], out_features=config.d_attn),  # K
                torch.nn.Linear(in_features=config.params['d_encoder'], out_features=config.d_attn),  # V
            ])
            for _ in range(config.params['num_attn_heads'])
        ])
        self.sdpa = ScaledDotProductAttention()
        self.linear_out = torch.nn.Linear(in_features=config.d_features, out_features=config.d_features)

    def forward(self, Q, K, V):
        batch_size, num_timesteps = Q.shape[:2]

        # temp matrices
        Q_all = torch.zeros((batch_size, config.params['num_attn_heads'], num_timesteps, config.d_attn))
        K_all = torch.zeros_like(Q_all)
        V_all = torch.zeros_like(Q_all)

        for i in range(config.params['num_attn_heads']):
            Q_all[:, i, ...] = self.linear_projs_in[i][0](Q)
            K_all[:, i, ...] = self.linear_projs_in[i][1](K)
            V_all[:, i, ...] = self.linear_projs_in[i][2](V)

        x = self.sdpa(Q_all, K_all, V_all)  # parallel
        x = x.view(batch_size, num_timesteps, config.d_attn * config.params['num_attn_heads'])  # concat from attn heads
        x = self.linear_out(x)
        x = torch.functional.dropout(x, p=None, training=True)

        return x
        

class PositionalEncoding(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.pe = torch.zeros((config.max_len, config.params['d_encoder']))
        
        for i in range(config.max_len):
            for j in range(0, config.params['d_encoder'], 2):
                div_term = 1 / (10000 ** (j / config.params['d_encoder']))
                self.pe[i][j] = math.sin(i * div_term)
                self.pe[i][j + 1] = math.cos(i * div_term)

    def forward(self, x):
        x += self.pe[:x.shape[1]]  # inject positional information

        return x


class MultiHeadSelfAttentionModule(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=config.d_features)
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

        self.expansion_factor = 4
        self.expansion = config.d_features * self.expansion_factor

        self.nn = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=config.d_features),
            torch.nn.Linear(in_features=config.d_features, out_features=self.expansion),
            torch.nn.SiLU(),  # swish
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=self.expansion, out_features=config.d_features),
            torch.nn.Dropout()
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
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=config.d_features)

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

    # spec-aug modifies the spectrogram by warping in the time direction,
    # masking blocks of consecutive frequency channels
    # and masking blocks of utterances in time
    # helps network be more robust to time deformations and partial loss of frequency and segments of speech

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.permute(0, 2, 1)

        # TODO: use this implementation instead - https://github.com/IMLHF/SpecAugmentPyTorch
        #  supports batches

        x = spec_augment(
            mel_spectrogram=x,
            time_warping_para=5,
            frequency_masking_para=config.frequency_mask,
            time_mask_num=config.num_time_masks
        )

        return x.permute(0, 2, 1)


class Conformer(torch.nn.Module):

    # TODO: how do we get d_encoder from d_features?

    def __init__(self, num_blocks):
        super().__init__()

        self.spec_aug = SpecAug()
        self.linear = torch.nn.Linear(in_features=config.d_features, out_features=config.params['d_encoder'])
        self.dropout = torch.nn.Dropout()
        self.blocks = [ConformerBlock()] * num_blocks

    def forward(self, x):
        x = self.spec_aug(x)
        x = self.linear(x)
        x = self.dropout(x)

        for conformer_block in self.blocks:
            x = conformer_block(x)

        return x


def main():
    batch_size, num_timesteps = 4, 100
    x = torch.rand((batch_size, num_timesteps, config.d_features))

    model = Conformer(num_blocks=2)
    output = model(x)
    print(output.shape)


if __name__ == '__main__':
    main()

