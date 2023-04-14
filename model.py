__author__ = 'Fabio'

from torch.nn import Module, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, Linear
import torch
from torch import Tensor

class TimeSeriesTransformer(Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float,
                 dim_feedforward: int):

        self.encoder_linear_layer = Linear(in_features=input_size, out_features=d_model)
        self.decoder_linear_layer = Linear(in_features=input_size, out_features=d_model)
        self.output_layer = Linear(in_features=d_model, out_features=output_size)

        encoder = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = TransformerEncoder(encoder, num_layers=n_layers, norm=False)

        decoder = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = TransformerEncoder(encoder, num_layers=n_layers, norm=False)


    def forward(self, input_seq):
        h = self.encoder_linear_layer(input_seq)
        h = self.enco

        pass


    def _generate_triangle_mask(self, sz: int) -> Tensor:
        """
        :param sz: int - length of the sequence
        :return: 2d triangle tensor with -inf on the upper triangle and 0 on the diag and
        lower triangle
        """
        if self.cached_mask is None:
            mask = torch.triu(torch.ones(sz, sz), diagonal=1)
            self.cached_mask = mask.masked_fill(mask == 1, float('-inf'))
        return self.cached_mask