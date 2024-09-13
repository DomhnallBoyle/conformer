import math

import torch

import config
from spec_aug import SpecAugmentTorch


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
                div_term = 1 / (10000 ** (j / config.params['d_encoder']))
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
        

class SpecAug(SpecAugmentTorch):

    # spec-aug modifies the spectrogram by warping in the time direction,
    # masking blocks of consecutive frequency channels
    # and masking blocks of utterances in time
    # helps network be more robust to time deformations and partial loss of frequency and segments of speech

    # TODO: find the correct params for Spec Augment

    def __init__(self):
        super().__init__(**{
            'W': 5,  # time warping param
            'F': config.frequency_mask,  # frequency masking param
            'T': 0.05,  # time masking param
            'mF': 1,  # frequency mask num
            'mT': config.num_time_masks,  # time mask num
            'batch': True
        })

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)

        x = super().forward(spec_batch=x)

        return x.squeeze(1).permute(0, 2, 1)


class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.spec_aug = SpecAug()
        self.linear = torch.nn.Linear(in_features=config.d_features, out_features=config.params['d_encoder'])
        self.dropout = torch.nn.Dropout(p=config.p_drop) 
        self.blocks = torch.nn.Sequential(*[ConformerBlock() for _ in range(config.params['num_encoder_layers'])])

    def forward(self, x):
        x = self.spec_aug(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.blocks(x)

        return x


class Decoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=config.params['d_encoder'],
            hidden_size=config.params['d_decoder'],
            num_layers=config.params['num_decoder_layers'],
            bias=True,
            bidirectional=False,
            dropout=config.p_drop
        )

    def forward(self, x):
        output, (hidden_state, cell_state) = self.lstm(x)

        return output


class E2E(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder_out = self.encoder(x)
        
        return self.decoder(encoder_out)


def main():
    batch_size, num_timesteps = 4, 100
    x = torch.rand((batch_size, num_timesteps, config.d_features))

    model = E2E()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model {config.model_size}: {num_params} total params')
    
    print(f'Input: {x.shape}')
    output = model(x)
    print(f'Output: {output.shape}')
    

if __name__ == '__main__':
    main()

