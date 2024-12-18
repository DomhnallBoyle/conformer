import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from tqdm import tqdm

import config
from utils import list_type, plot_graph


class LibriSpeechDataset(torch.utils.data.Dataset):
    
    def __init__(self, path: str, sets: list[str], name: str = 'train') -> None:
        self.path = Path(path)
        self.sets = [self.path.joinpath(_set) for _set in sets]
        self.name = name
        self.lengths_path = Path(f'{name}_lengths.pkl')
        self.lengths_freq_path = Path(f'{name}_lengths_freq.png')

        for _set in self.sets:
            if not _set.exists():
                print(f'Set "{_set}" does not exist')
                exit()

        self.data = self.get_data()
        self.lengths = self.get_lengths()
        self.vocab = self.get_vocab()  # vocab and decoder lookups
        self.decoder = {v: k for k, v in self.vocab.items()}
        self.num_classes = len(self.vocab)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> None:
        _set, audio_name, transcript = self.data[index]
        audio_path = self.get_audio_path(_set, audio_name)

        # load mel-spec filterbank features
        audio, mel = self.load_audio_and_mel(audio_path)

        # convert transcript to token indices
        transcript_targets = torch.Tensor([self.vocab[token] for token in transcript.split(' ')])

        return torch.from_numpy(audio), torch.from_numpy(mel), transcript, transcript_targets

    def get_audio_path(self, _set: str, audio_name: str) -> str:
        speaker_id, chapter_id = audio_name.split('-')[:2]
        audio_path = self.path / _set / speaker_id / chapter_id / (audio_name + '.flac')

        return audio_path

    def load_audio_and_mel(self, audio_path: str) -> np.array:
        wav, sample_rate = librosa.load(audio_path, sr=None)  # use sample rate of files
        assert sample_rate == config.sample_rate, f'Audio sample rate ({sample_rate}) != {config.sample_rate}'
        
        return wav, librosa.feature.melspectrogram(
            y=wav,
            sr=sample_rate,
            win_length=config.window_size,
            hop_length=config.stride,
            n_mels=config.num_mels,
        ).T

    def read_transcript_file(self, path: Path) -> list[list[str]]:
        data = []
        with path.open('r') as f:
            for line in f.read().splitlines():
                line_split = line.split(' ')
                audio_name, transcript = line_split[0], ' '.join(line_split[1:])
                data.append([audio_name, transcript.strip().lower()])

        return data

    def get_data(self) -> list[list[str]]:
        data = []

        for _set in self.sets:
            print('Gathering data from:', _set)
            for transcript_path in tqdm(_set.rglob('*.trans.txt')):
                file_data = self.read_transcript_file(path=transcript_path)
                data.extend([[_set.name] + d for d in file_data])

        return data
    
    def get_lengths(self) -> dict:
        print('Extracting dataset lengths...')

        if self.lengths_path.exists():
            with self.lengths_path.open('rb') as f:
                return pickle.load(f)

        lengths = {}
        for i in tqdm(range(len(self))):
            _set, audio_name, _ = self.data[i]
            audio_path = self.get_audio_path(_set, audio_name)

            # load mel-spec filterbank features
            _, mel = self.load_audio_and_mel(audio_path)
            lengths[i] = mel.shape[0]

        # cache lengths for later usage
        with self.lengths_path.open('wb') as f:
            pickle.dump(lengths, f)

        # display lengths frequency
        lengths_freq = {}
        for length in lengths.values():
            lengths_freq[length] = lengths_freq.get(length, 0) + 1

        plot_graph(
            plt.bar, 
            *zip(*lengths_freq.items()), 
            title=f'{self.name} lengths frequency', 
            x_label='Lengths', 
            y_label='Count', 
            save_path=self.lengths_freq_path
        )

        return lengths
    
    def get_vocab(self) -> dict:
        vocab = set()

        for _, _, transcript in self.data:
            for token in transcript.split(' '):
                vocab.add(token)

        # NOTE: excluding blank here (i + 1)
        return {token: i + 1 for i, token in enumerate(vocab)}


def main(args) -> None:
    dataset = LibriSpeechDataset(path=args.dataset_path, sets=args.sets)
    for i in range(5):
        print(dataset[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('sets', type=list_type)

    main(parser.parser_args())
