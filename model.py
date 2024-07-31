import torch

import config


class ConvolutionModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.modules = torch.nn.Sequential(
            torch.nn.LayerNorm(),
            torch.nn.Conv1d(kernel_size=1),  # point-wise conv
            torch.nn.GLU(),
            torch.nn.Conv1d(),  # depth-wise conv
            torch.nn.BatchNorm1d(),
            torch.nn.SiLU(),  # swish
            torch.nn.Conv1d(kernel_size=1),  # point-wise conv
            torch.nn.Dropout()
        )

    def forward(self, x):
        x_init = x

        x = self.modules(x)

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
        )]
        self.sdpa = ScaledDotProductAttention()
        self.linear_out = torch.nn.Linear(in_features=None, out_features=None)

    def forward(self, Q, K, V):
        batch_size, num_timesteps = Q.shape[:2]

        # temp matrices
        Q_all = torch.zeros((batch_size, config.params['num_attn_heads'], num_timesteps, config.d_attn))
        K_all = torch.zeros_like(Q_all)
        V_all = torch.zeros_like(Q_all)

        for i in range(config.params['num_attn_heads'):
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

    def forward(self):
        x += self.pe[:x.shape[1]]  # inject positional information

        return x


class MultiHeadSelfAttentionModule(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm()
        self.pe_layer = PositionalEncoding()
        self.mha = MultiHeadAttention()

    def forward(self, x):
        x_init = x    

        x = self.layer_norm(x)
        x = self.pe_layer(x_norm)  # relative sinusoidal positional encoding
        x = self.mha(x, x, x)

        return x_init + x  # skip connection


class FeedForwardModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.expansion_factor = 4

        self.modules = torch.nn.Sequential(
            torch.nn.LayerNorm(),
            torch.nn.Linear(),
            torch.nn.SiLU(),  # swish
            torch.nn.Dropout(),
            torch.nn.Linear(),
            torch.nn.Dropout()
        )

    def forward(self, x):
        x_init = x

        x = self.modules(x)

        return x_init + x


class ConformerBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.ff_1 = FeedForwardModule()
        self.mhsa = MultiHeadSelfAttentionModule()
        self.conv = ConvolutionModule()
        self.ff_2 = FeedForwardModule()
        self.layer_norm = torch.nn.LayerNorm()

    def forward(self):
        pass


class SpecAug(torch.nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class Conformer(torch.nn.Module):
    
    def __init__(self, num_blocks):
        super().__init__()

        self.blocks = [ConformerBlock()] * num_blocks

    def forward(self):
        pass


def main():
    batch_size, num_timesteps, num_features = 4, 100, 80
    x = torch.rand((batch_size, num_timesteps, num_features))

    self.model = Conformer()
    output = self.model(x)
    print(output.shape)


if __name__ == '__main__':
    main()

