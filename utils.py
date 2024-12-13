import librosa
import matplotlib.pyplot as plt
import torch


def plot_mels(mels: list[torch.Tensor], titles: list[str]) -> None:
    assert len(mels) == len(titles)

    fig, axes = plt.subplots(len(mels), 1, sharex=True, sharey=True)
    for i, (mel, title) in enumerate(zip(mels, titles)):
        axes[i].set_title(title)
        axes[i].imshow(librosa.amplitude_to_db(mel), origin='lower', aspect='auto')

    fig.tight_layout()
    plt.show()


def list_type(s: str) -> list[str]:
    return s.split(',')


def decode(sentence_logits: torch.Tensor, decoder: dict) -> str:
    # sample = [T, C] (softmax)
    # blank = 0, which isn't in decoder - don't include it
    sentence_logits_argmax = torch.argmax(sentence_logits, dim=1).tolist()  # [T]
    sentence_decoded = [decoder[i] for i in sentence_logits_argmax if i in decoder]

    return ' '.join(sentence_decoded)
