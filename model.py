import torch


class ConvolutionModule(torch.nn.Module):

    def __init__(self):
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

    def forward(self):
        pass


class MultiHeadSelfAttentionModule(torch.nn.Module):
    
    def __init__(self):
        pass

    def forward(self):
        pass


class FeedForwardModule(torch.nn.Module):

    def __init__(self):
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
        pass


class ConformerBlock(torch.nn.Module):

    def __init__(self):
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

