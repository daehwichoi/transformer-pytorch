import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num=8):
        super().__init__()
        self.head_num = head_num

        self.query_embed = nn.Linear(100, head_num * 64)
        self.key_embed = nn.Linear(100, head_num * 64)
        self.value_embed = nn.Linear(100, head_num * 64)
        self.output_embed = nn.Linear(512, head_num * 64)

    # q, k Shape (Batch X Head_num X token_length X hidden)
    # q는 현재 token을 embedding
    # k는 문장 전체의 token을 embedding
    # output = 문장 내에 어느 token에 주의를 기울일지 선택
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = k.size()[-1]
        k_transpose = torch.transpose(k, 3, 2)

        output = torch.matmul(q, k_transpose)
        output = output / math.sqrt(d_k)

        if mask:
            output = mask(output)

        output = F.softmax(output, -1)
        output = torch.matmul(output, v)

        return output

    def forward(self, q, k, v):
        batch_size = q.size()[0]

        q = self.query_embed(q).view(batch_size, )
        k = self.key_embed(k)
        v = self.value_embed(v)

        output = self.scaled_dot_product_attention(q, k, v)
        batch_num, head_num, seq_num, hidden_num = output.size()
        output = torch.transpose(output, 1, 2).contiguous().view((batch_num, -1, hidden_num * head_num))

        return output


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 64 * 4)
        self.layer2 = nn.Linear(64 * 4, 64)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(F.relu(output))

        return output


# Layer norm
class AddLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def layer_norm(self, input):
        mean = torch.mean(input, dim=-1, keepdim=True)
        std = torch.std(input, dim=-1, keepdim=True)
        output = (input - mean) / std
        return output

    def forward(self, input, residual):
        return residual + self.layer_norm(input)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head = MultiHeadAttention()
        self.residual_layer1 = AddLayerNorm()
        self.feed_forward = FeedForward()
        self.residual_layer2 = AddLayerNorm()

    def forward(self, q, k, v):
        multihead_output = self.multi_head(q, k, v)
        residual1_output = self.residual_layer1(multihead_output, q)
        feedforward_output = self.feed_forward(residual1_output)
        output = self.residual_layer2(feedforward_output, residual1_output)

        return output


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


if __name__ == '__main__':
    model = Encoder()

    q = torch.rand((64, 1, 800))
    k = torch.rand((64, 100, 800))
    v = torch.rand((64, 100, 800))

    output = model(q, k, v)
    print(output.shape)
