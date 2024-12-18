import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

import aiayn.utils


def get_num_params(model: torch.nn.Module) -> float:
    num_params = aiayn.utils.get_num_params(model)

    return round(num_params / 1_000_000, 1)


def mel_db(mel: np.array) -> np.array:
    return librosa.amplitude_to_db(mel)


def plot_mels(mels: list[torch.Tensor], titles: list[str], show=True) -> Figure:
    assert len(mels) == len(titles)

    fig, axes = plt.subplots(len(mels), 1, sharex=True, sharey=True)

    if len(mels) == 1:
        axes = [axes]

    for i, (mel, title) in enumerate(zip(mels, titles)):
        axes[i].set_title(title)
        axes[i].imshow(mel_db(mel), origin='lower', aspect='auto')

    fig.tight_layout()
    if show:
        plt.show()

    return fig


def plot_mel_tensorboard(mel: np.array) -> Figure:
    return plot_mels([mel.permute(1, 0)], [''], show=False)


def plot_graph(plot, x, y, title, x_label, y_label, save_path=None) -> None:
    plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def list_type(s: str) -> list[str]:
    return s.split(',')


def decode(sentence_logits: torch.Tensor, decoder: dict) -> str:
    # sample = [T, C] (softmax)
    # blank = 0, which isn't in decoder - don't include it
    sentence_logits_argmax = torch.argmax(sentence_logits, dim=1).tolist()  # [T]
    sentence_decoded = [decoder[i] for i in sentence_logits_argmax if i in decoder]

    return ' '.join(sentence_decoded)
