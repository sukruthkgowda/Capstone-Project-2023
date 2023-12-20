import torch
from torch import nn
from torch.nn import functional as F
import math
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
class ReverseEncoder(nn.Module):
    def __init__(self):
        super(ReverseEncoder, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            
            Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),

            Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0),
            Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0))

    def forward(self, audio_sequences, face_sequences):

        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
    
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(6, 16, kernel_size=7, stride=1, padding=3),  # 128, 128
    ),
    nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64, 64
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32, 32
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16, 16
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 8, 8
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 4, 4
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 2, 2
        nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
    ),
])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),  # 2, 2
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 2, 2
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 2, 2
    ),
    nn.Sequential(
        nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=0),  # 4, 4
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8, 8
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.ConvTranspose2d(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16, 16
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32, 32
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.ConvTranspose2d(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64, 64
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
    ),
    nn.Sequential(
        nn.ConvTranspose2d(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128, 128
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
    ),
])


        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x

        return outputs
    
class FusionModelForward(nn.Module):
    def __init__(self):
        super(FusionModelForward, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
    nn.Sequential(
        nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3),  # 96, 128
    ),
    nn.Sequential(
        nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 96, 64
        nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2),  # 96, 64
    ),
    nn.Sequential(
        nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 48, 32
        nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2),  # 48, 32
    ),
    nn.Sequential(
        nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 24, 16
        nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2),  # 24, 16
    ),
    nn.Sequential(
        nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 12, 8
        nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 12, 8
    ),
    nn.Sequential(
        nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 6, 4
        nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 6, 4
    ),
    nn.Sequential(
        nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3, 2
        nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 3, 2
    ),
    nn.Sequential(
        nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 1, 1
        nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 1, 1
    ),
])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                        torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)