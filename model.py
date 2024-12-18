import argparse

import torch

import config
from aiayn.model import MultiHeadAttention, PositionalEncoding
from dataset import LibriSpeechDataset
from utils import get_num_params, list_type


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


class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(in_features=config.d_features, out_features=config.params['d_encoder'])
        self.dropout = torch.nn.Dropout(p=config.p_drop) 
        self.blocks = torch.nn.Sequential(*[ConformerBlock() for _ in range(config.params['num_encoder_layers'])])

    def forward(self, x):
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
    
    def __init__(self, num_classes, group_norm=False):
        super().__init__()

        print(f'Creating model ({config.model_size})...', end='')

        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=num_classes)

        if group_norm:
            convert_bn_layer(self)

        print(f'{self.num_params} million total params')

    @property
    def num_encoder_params(self) -> int:
        return get_num_params(self.encoder)

    @property
    def num_decoder_params(self) -> int:
        return get_num_params(self.decoder)

    @property
    def num_params(self) -> int:
        return self.num_encoder_params + self.num_decoder_params

    def forward(self, x):
        encoder_out = self.encoder(x)
        
        return self.decoder(encoder_out)


def convert_bn_layer(module: torch.nn.Module) -> None:
    # TODO: num_groups should be selected automatically -> error when num_channels (features) can't be divided evenly by num_groups
    # recursive function to convert batch-norm layers to group-norm for gradient accumulation training
    for name, l in module.named_children():
        if any([isinstance(l, x) for x in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]]):
            setattr(module, name, torch.nn.GroupNorm(16, num_channels=l.num_features))
        if len(list(l.children())) > 0:
            convert_bn_layer(l)


def main(args) -> None:
    dataset = LibriSpeechDataset(path=args.dataset_path, sets=args.sets)
    mel = dataset[0][0]
    mel = mel.unsqueeze(0)  # add batch dimension

    model = E2E(num_classes=dataset.num_classes + 1)
    print(f'Num Encoder params: {model.num_encoder_params} million')
    print(f'Num Decoder params: {model.num_decoder_params} million')
    
    print(f'Input: {mel.shape}')
    output = model(mel)[0]
    print(f'Output: {output.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('sets', type=list_type)

    main(parser.parse_args())
