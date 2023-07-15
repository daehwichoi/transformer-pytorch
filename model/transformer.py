import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class Encoder(MultiHeadAttention, FeedForward):
    def __init__(self):
        super().__init__()


class Decoder(MultiHeadAttention, FeedForward):
    def __init__(self):
        super().__init__()


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
