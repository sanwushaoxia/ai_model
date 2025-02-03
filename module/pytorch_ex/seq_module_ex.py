import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import tanh, relu
from torch.nn.init import xavier_uniform_, constant_

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.weight_ih = Parameter(self.init_(torch.rand(input_size, hidden_size)))
        self.weight_hh = Parameter(self.init_(torch.rand(hidden_size, hidden_size)))

        if self.bias:
            self.bias_ih = Parameter(self.init_(torch.rand(hidden_size)))
            self.bias_hh = Parameter(self.init_(torch.rand(hidden_size)))

        if self.nonlinearity == 'tanh':
            self.activation = tanh
        elif self.nonlinearity == 'relu':
            self.activation = relu

    def init_(self, tensor):
        k = 1 / self.hidden_size
        tensor -= 0.5
        tensor *= 2 * k ** (0.5)
        return tensor

    def forward(self, input, h_0):
        h_1 = input @ self.weight_ih + h_0 @ self.weight_hh
        if self.bias:
            h_1 += self.bias_ih + self.bias_hh
        if self.nonlinearity in ['tanh', 'relu']:
            h_1 = self.activation(h_1)
        return h_1

class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, kdim=None, vdim=None, batch_first=False, attn='none'):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.splited_embed_dim = self.embed_dim // self.num_heads
        assert self.embed_dim == self.num_heads * self.splited_embed_dim

        self.dropout = nn.Dropout(dropout)
        self.bias = bias

        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim

        self.batch_first = batch_first

        self.attn = attn
        assert attn in ['sum', 'product', 'none']

        self.q_proj_weight = Parameter(torch.empty(self.embed_dim, self.embed_dim))
        self.k_proj_weight = Parameter(torch.empty(self.embed_dim, self.kdim))
        self.v_proj_weight = Parameter(torch.empty(self.embed_dim, self.vdim))

        self.out_proj_weight = Parameter(torch.empty(self.embed_dim, self.embed_dim))
        if self.bias:
            self.in_proj_bias = Parameter(torch.empty(3, self.embed_dim))
            self.out_proj_bias = Parameter(torch.empty(self.embed_dim))

        if self.attn == 'sum':
            self.attn_W = Parameter(torch.empty(1, 2 * self.splited_embed_dim, self.num_heads))
        elif self.attn == 'product':
            self.attn_W = Parameter(torch.empty(self.splited_embed_dim, self.splited_embed_dim, self.num_heads))

        self._reset_parameters()
    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.out_proj_weight)
        if self.bias:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj_bias, 0.)
        if self.attn in ['sum', 'product']:
            xavier_uniform_(self.attn_W)

    def forward(self, Q, K, V, key_padding_mask=None):
        # key_padding_mask = key_padding_mask.transpose(0, 1)
        # key_padding_mask = torch.logical_not(key_padding_mask).to(torch.uint8)

        if self.batch_first:
            Q, K, V = Q.transpose(0, 1), K.transpose(0, 1), V.transpose(0, 1)
        
        Q = torch.einsum('lni, oi -> lno', Q, self.q_proj_weight)
        K = torch.einsum('sni, oi -> sno', K, self.k_proj_weight)
        V = torch.einsum('sni, oi -> sno', V, self.v_proj_weight)
        if self.bias:
            Q += self.in_proj_bias[0]
            K += self.in_proj_bias[1]
            V += self.in_proj_bias[2]
        Q = Q.view(Q.size(0), Q.size(1), self.splited_embed_dim, self.num_heads)
        K = K.view(K.size(0), K.size(1), self.splited_embed_dim, self.num_heads)
        V = V.view(V.size(0), V.size(1), self.splited_embed_dim, self.num_heads)

        if self.attn == 'sum':
            Q = torch.einsum('lnik, ik -> lnk', Q, self.attn_W[0, : self.splited_embed_dim])
            K = torch.einsum('snik, ik -> snk', K, self.attn_W[0, self.splited_embed_dim: ])

            Q = Q.view(Q.size(0), 1, Q.size(1), Q.size(2))
            K = K.view(1, K.size(0), K.size(1), K.size(2))

            W = Q + K
        elif self.attn == 'product':
            W = torch.einsum('lnok, oik, snik -> lsnk', Q, self.attn_W, K) / self.embed_dim ** (0.5)
        else:
            W = torch.einsum('lnik, snik -> lsnk', Q, K) / self.embed_dim ** (0.5)

        W = torch.nn.Softmax(1)(W)
        W = self.dropout(W)
        V = torch.einsum('lsnk, snik -> lnik', W, V)
        V = V.reshape(V.size(0), V.size(1), -1)
        V = V @ self.out_proj_weight
        if self.bias:
            V += self.out_proj_bias

        if self.batch_first:
            V = V.transpose(0, 1)
        W = W.sum(dim=-1)
        return V, W.view(-1, W.size(0), W.size(1)) / self.num_heads

class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self, src: Tensor):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor):
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor):
        output = src
        for i in range(self.num_layers):
            output = self.encoder_layer(output)
        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)

        self.multihead_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._mha_block(self.norm2(x), memory)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._mha_block(x, memory))
            x = self.norm3(x + self._ff_block(x))
        return x
    
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)
    
    def _mha_block(self, x: Tensor, mem: Tensor) -> Tensor:
        x = self.multihead_attn(x, mem, mem)[0]
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None) -> None:
        super().__init__()
        self.decoder_layer = decoder_layer
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        output = tgt
        for i in range(self.num_layers):
            output = self.decoder_layer(output, memory)
        if self.norm is not None:
            output = self.norm(output)
        return output

if __name__ == '__main__':
    Q = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)
    '''
    model = MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.2)
    output = model(Q, Q, Q)
    print(output[0].size(), output[1].size(), '\n')
    '''
    '''
    model = TransformerEncoderLayer(d_model=512, nhead=8)
    output = model(Q)
    print(output.size(), '\n')
    '''
    '''
    model = TransformerEncoder(encoder_layer=TransformerEncoderLayer(512, 8), num_layers=6)
    output = model(Q)
    print(output.size(), '\n')
    '''
    '''
    model = TransformerDecoderLayer(d_model=512, nhead=8)
    output = model(tgt, Q)
    print(output.size(), '\n')
    '''
    # '''
    model = TransformerDecoder(decoder_layer=TransformerDecoderLayer(512, 8), num_layers=6)
    output = model(tgt, Q)
    print(output.size(), '\n')
    # '''
