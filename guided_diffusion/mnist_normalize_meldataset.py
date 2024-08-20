""" from https://github.com/jik876/hifi-gan """

import math
import os,glob
import random
import torch
import torch.utils.data
import numpy as np
import librosa.display as display
import torchaudio
import torchaudio.transforms as AT
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from scipy import signal
import librosa

MAX_WAV_VALUE = 32768.0

# downsampling
def load_wav(full_path):
    sampling_rate, data = read(full_path)
        
    new_sample_rate = 22050 
    num_samples = round(len(data) * float(new_sample_rate) / sampling_rate)
    resampled_data = signal.resample(data, num_samples)
    
    return resampled_data, new_sample_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def normalize_mel(S):
    min_level_db= -100
    return torch.clip((S-min_level_db)/-min_level_db,0,1)

def denormalize_mel(S_normalized, min_level_db=-100):
    return S_normalized * -min_level_db + min_level_db

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)

    spec_db = librosa.amplitude_to_db(spec.numpy(), ref=np.max)
    spec = normalize_mel(torch.tensor(spec_db))

    # spec = spectral_normalize_torch(spec)

    return torch.Tensor(spec)

class MnistMelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, sampling_rate, classes, shard=0, num_shards=1, n_fft=1024, num_mels=80,
                 hop_size=256, win_size=1024,  fmin=0, fmax=8000, split=True, n_cache_reuse=1,
                 fmax_loss=None, fine_tuning=False, base_mels_path=None):
        
        self.audio_files = training_files[shard:][::num_shards]
        # self.audio_files = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

        # for guided diffusion
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes
        self.shard = shard
        self.num_shards = num_shards
        

    def __getitem__(self, index):
        filename = self.audio_files[index]
        
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE

            if not self.fine_tuning:
                audio = normalize(audio) * 0.95

            # self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            # self._cache_ref_count = self.n_cache_reuse
        # else:
        #     audio = self.cached_wav
        #     self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)

        #for guided diffusion
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[index], dtype=np.int64)

        return mel[:, :, :80], out_dict

    def __len__(self):
        return len(self.audio_files)