import os
import sys
import string
import time
import argparse
import json
import numpy as np
import torch
import subprocess
TTS_PATH = "./TTS/"
sys.path.append("C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\TTS\\Capstone 2\\TTS")
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *
from TTS.tts.utils.speakers import SpeakerManager
from pydub import AudioSegment
import librosa

# Add command-line arguments
parser = argparse.ArgumentParser(description='TTS Conversion')
parser.add_argument('--text_file', type=str, help='Path to the text file')
parser.add_argument('--video_file', type=str, help='Path to the mp4 video file')
args = parser.parse_args()


OUT_PATH = 'out/'
os.makedirs(OUT_PATH, exist_ok=True)
# model vars 
MODEL_PATH = 'best_model_latest.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = "language_ids.json"
TTS_SPEAKERS = "speakers.json"
USE_CUDA = torch.cuda.is_available()

# load the config
C = load_config(CONFIG_PATH)
# load the audio processor
ap = AudioProcessor(**C.audio)
speaker_embedding = None
C.model_args['d_vector_file'] = TTS_SPEAKERS
C.model_args['use_speaker_encoder_as_loss'] = False

model = setup_model(C)
model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# remove speaker encoder
model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
  if "speaker_encoder" in key:
    del model_weights[key]

model.load_state_dict(model_weights)
model.eval()

if USE_CUDA:
    model = model.cuda()

# synthesize voice
use_griffin_lim = False

CONFIG_SE_PATH = "config_se.json"
CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"

SE_speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH, encoder_config_path=CONFIG_SE_PATH, use_cuda=USE_CUDA)

def compute_spec(ref_file):
  y, sr = librosa.load(ref_file, sr=ap.sample_rate)
  spec = ap.spectrogram(y)
  spec = torch.FloatTensor(spec).unsqueeze(0)
  return spec

#Extract Text from Text File
with open(args.text_file, 'r') as file:
    text = file.read().strip()

#Extract .wav file from .mp4 file
audio_output_path = args.video_file.rsplit('.', 1)[0] + ".wav"  # replace .mp4 with .wav
command = ["ffmpeg", "-i", args.video_file, "-q:a", "0", "-map", "a", audio_output_path]
subprocess.run(command)

reference_files = [audio_output_path]

# Normalize the audio file
for sample in reference_files:
    command = [
        "ffmpeg-normalize", sample, 
        "-nt", "rms", 
        "-t", "-27", 
        "-o", sample, 
        "-ar", "16000", 
        "-f"
    ]
    subprocess.run(command)

reference_emb = SE_speaker_manager.compute_d_vector_from_clip(reference_files)

model.length_scale = 1  # scaler for the duration predictor. The larger it is, the slower the speech.
model.inference_noise_scale = 0.3 # defines the noise variance applied to the random z vector at inference.
model.inference_noise_scale_dp = 0.3 # defines the noise variance applied to the duration predictor z vector at inference.

language_id = 0

wav, alignment, _, _ = synthesis(
                    model,
                    text,
                    C,
                    "cuda" in str(next(model.parameters()).device),
                    ap,
                    speaker_id=None,
                    d_vector=reference_emb,
                    style_wav=None,
                    language_id=language_id,
                    enable_eos_bos_chars=C.enable_eos_bos_chars,
                    use_griffin_lim=True,
                    do_trim_silence=False,
                ).values()
print("Generated Audio")

file_name = 'output.wav'
out_path = os.path.join(OUT_PATH, file_name)
print(" > Saving output to {}".format(out_path))
ap.save_wav(wav, out_path)
