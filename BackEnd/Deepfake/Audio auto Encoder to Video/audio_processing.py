import librosa
import librosa.filters
import numpy as np

from scipy import signal
from scipy.io import wavfile
from parameters import HyperParameters,params


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size():
    hop_size = params.hop_size
    if hop_size is None:
        assert params.frame_shift_ms is not None
        hop_size = int(params.frame_shift_ms / 1000 * params.sample_rate)
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, params.preemphasis, params.preemphasize))
    S = _amp_to_db(np.abs(D)) - params.ref_level_db
    
    if params.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav):
    D = _stft(preemphasis(wav, params.preemphasis, params.preemphasize))
    # D = _stft(preemphasis(wav, params.preemphasis, params.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - params.ref_level_db
    
    if params.signal_normalization:
        return _normalize(S)
    return S
# def melspectrogram(wav, preemphasis_value=0.97, preemphasize=True):
#     D = _stft(preemphasis(wav, preemphasis_value, preemphasize))
#     S = _amp_to_db(_linear_to_mel(np.abs(D))) - params.ref_level_db
    
#     if params.signal_normalization:
#         return _normalize(S)
#     return S


def _lws_processor():
    import lws
    return lws.lws(params.n_fft, get_hop_size(), fftsize=params.win_size, mode="speech")

# def _stft(y):
    # if params.use_lws:
    #     return _lws_processor(hp).stft(y).T
    # else:
        # return librosa.stft(y=y, n_fft=params.n_fft, hop_length=get_hop_size(), win_length=params.win_size)
def _stft(y):
    if params.use_lws:   # <-- Note the change here
        return _lws_processor().stft(y).T
    else:
        return librosa.stft(y=y, n_fft=params.n_fft, hop_length=get_hop_size(), win_length=params.win_size)   # <-- Note the change here as well

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    assert params.fmax <= params.sample_rate // 2
    # return librosa.filters.mel(params.sample_rate, params.n_fft, n_mels=params.num_mels,
                              #  fmin=params.fmin, fmax=params.fmax)
    return librosa.filters.mel(sr=params.sample_rate, n_fft=params.n_fft, n_mels=params.num_mels, fmin=params.fmin, fmax=params.fmax)


def _amp_to_db(x):
    min_level = np.exp(params.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if params.allow_clipping_in_normalization:
        if params.symmetric_mels:
            return np.clip((2 * params.max_abs_value) * ((S - params.min_level_db) / (-params.min_level_db)) - params.max_abs_value,
                           -params.max_abs_value, params.max_abs_value)
        else:
            return np.clip(params.max_abs_value * ((S - params.min_level_db) / (-params.min_level_db)), 0, params.max_abs_value)
    
    assert S.max() <= 0 and S.min() - params.min_level_db >= 0
    if params.symmetric_mels:
        return (2 * params.max_abs_value) * ((S - params.min_level_db) / (-params.min_level_db)) - params.max_abs_value
    else:
        return params.max_abs_value * ((S - params.min_level_db) / (-params.min_level_db))

def _denormalize(D):
    if params.allow_clipping_in_normalization:
        if params.symmetric_mels:
            return (((np.clip(D, -params.max_abs_value,
                              params.max_abs_value) + params.max_abs_value) * -params.min_level_db / (2 * params.max_abs_value))
                    + params.min_level_db)
        else:
            return ((np.clip(D, 0, params.max_abs_value) * -params.min_level_db / params.max_abs_value) + params.min_level_db)
    
    if params.symmetric_mels:
        return (((D + params.max_abs_value) * -params.min_level_db / (2 * params.max_abs_value)) + params.min_level_db)
    else:
        return ((D * -params.min_level_db / params.max_abs_value) + params.min_level_db)
