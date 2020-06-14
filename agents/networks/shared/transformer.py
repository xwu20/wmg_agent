import math
import torch
from torch import nn
import torch.nn.functional as F
from agents.networks.shared.general import LinearLayer, ResidualLayer, LayerNorm

# Basic Transformer encoder. Adds no positional encodings. Performs no special pooling of the output.

class SelfAttentionLayer(nn.Module):
    def __init__(self, node_size, num_attention_heads, attention_head_size):
        super(SelfAttentionLayer, self).__init__()
        self.node_size = node_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size

        self.query = LinearLayer(node_size, node_size)
        self.key = LinearLayer(node_size, node_size)
        self.value = LinearLayer(node_size, node_size)

        self.dot_product_scale = 1.0 / math.sqrt(attention_head_size)

    def split_heads_apart(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)  # Changes the tensor shape without moving or copying data.
        return x.permute(0, 2, 1, 3)  # Moves data to maintain the meaning of the reordered list of dimensions.

    def concatenate_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.node_size,)
        return x.view(*new_shape)

    def forward(self, tens):  # [B, N, C]
        # B = batch size
        # N = num nodes
        # H = num heads
        # S = head size
        # C = content/node size = H * S

        # Project the input to get the QKV vectors.
        queries = self.query(tens)  # [B, N, C]
        keys = self.key(tens)       # [B, N, C]
        values = self.value(tens)   # [B, N, C]

        # Split the heads apart to operate on the head vectors.
        split_queries = self.split_heads_apart(queries)  # [B, H, N, S]
        split_keys = self.split_heads_apart(keys)        # [B, H, N, S]
        split_values = self.split_heads_apart(values)    # [B, H, N, S]

        # Take the dot product between each query and key to get the raw attention scores.
        transposed_keys = split_keys.transpose(-1, -2)   # [B, H, S, N]
        attention_scores = torch.matmul(split_queries, transposed_keys)  # [B, H, N, N]
        attention_scores = attention_scores * self.dot_product_scale     # [B, H, N, N]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [B, H, N, N]

        # Calculate the weighted value vectors.
        weighted_values = torch.matmul(attention_probs, split_values)  # [B, H, N, S]

        # Concatenate the heads for output.
        output = self.concatenate_heads(weighted_values)  # [B, N, C]
        return output


class TransformerLayer(nn.Module):
    def __init__(self, node_size, num_attention_heads, attention_head_size, hidden_layer_size):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttentionLayer(node_size, num_attention_heads, attention_head_size)
        self.attention_residual = ResidualLayer(node_size, node_size)
        self.attention_layer_norm = LayerNorm(node_size)

        self.feedforward = LinearLayer(node_size, hidden_layer_size)
        self.feedforward_residual = ResidualLayer(hidden_layer_size, node_size)
        self.feedforward_layer_norm = LayerNorm(node_size)

    def forward(self, input):
        # Attention phase.
        att_output = self.attention(input)
        att_output = self.attention_residual(att_output, input)
        att_output = self.attention_layer_norm(att_output)

        # Feedforward phase.
        output = self.feedforward(att_output)
        output = F.relu(output)
        output = self.feedforward_residual(output, att_output)
        output = self.feedforward_layer_norm(output)
        return output


class Transformer(nn.Module):
    def __init__(self, num_attention_heads, attention_head_size, num_layers, hidden_layer_size):
        super(Transformer, self).__init__()
        node_size = num_attention_heads * attention_head_size
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerLayer(node_size, num_attention_heads, attention_head_size, hidden_layer_size))

    def forward(self, tens):
        for layer in self.layers:
            tens = layer(tens)
        return tens
