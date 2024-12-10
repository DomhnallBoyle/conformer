from pathlib import Path

import librosa
import torch

import config


class LibriSpeechDataset(torch.utils.data.Dataset):
    
    def __init__(self, path: str, sets: list[str]) -> None:
        self.path = Path(path)
        self.sets = [self.path.joinpath(_set) for _set in sets]

        for _set in self.sets:
            if not _set.exists():
                print(f'Set "{_set}" does not exist')
                exit()

        self.data = self.get_data()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> None:
        _set, audio_name, transcription = self.data[index]
        speaker_id, chapter_id = audio_name.split('-')[:2]
        audio_path = self.path / _set / speaker_id / chapter_id / (audio_name + '.flac')

        # load mel-spec filterbank features
        wav, sample_rate = librosa.load(audio_path)
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=sample_rate,
            win_length=config.window_size,
            hop_length=config.stride,
            n_mels=config.num_mels,
        ).T
        
        return mel, transcription

    def read_transcription_file(self, path: Path) -> list[list[str]]:
        data = []
        with path.open('r') as f:
            for line in f.read().splitlines():
                line_split = line.split(' ')
                audio_name, transcription = line_split[0], ' '.join(line_split[1:])
                data.append([audio_name, transcription.strip()])

        return data

    def get_data(self) -> list[list[str]]:
        data = []

        for _set in self.sets:
            for transcription_path in _set.rglob('*.trans.txt'):
                file_data = self.read_transcription_file(path=transcription_path)
                data.extend([[_set.name] + d for d in file_data])

        return data


def main() -> None:
    dataset = LibriSpeechDataset(path='LibriSpeech', sets=['dev-clean'])
    for i in range(5):
        print(dataset[i])


if __name__ == '__main__':
    main()
