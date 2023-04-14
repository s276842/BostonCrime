
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model = 64):
        super().__init__()

        self.encoder_linear = nn.Linear(in_features=1, out_features=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = 1,
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, norm = None)

        self.decoder_linear = nn.Linear(in_features=1, out_features=d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=1,
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1, norm=None)

        self.positional_encoding_layer = PositionalEncoding(emb_size=d_model, dropout=0.2)
        self.out_layer = nn.Linear(in_features=d_model, out_features=1)


    def forward(self, src):
        tgt = src[:, -1].unsqueeze(-1)
        src = self.encoder_linear(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src)


        tgt = self.decoder_linear(tgt)
        out = self.decoder(
            tgt=tgt,
            memory=src
        )

        return self.out_layer(out)

import numpy as np
# def get_batch(data):
#     i = np.random.randint(0, len(data)-4)
#     src = []
#     tgt =
#
#     data[i:i + 3]
#     data[i + 3]


    # return src, tgt
def get_linear_data(n_points, window, min=0, max=100):
    data = torch.empty(n_points, window)

    x = torch.randint(min, max, (n_points,), dtype =torch.float32)
    for i in range(window):
        data[:, i] = x + i

    return TensorDataset(data.unsqueeze(-1), x + window)


if __name__ == '__main__':
    model = TimeSeriesTransformer().to(DEVICE)
    len_seq = 10

    train_data = get_linear_data(200, len_seq, 0, 1000)
    train_loader = DataLoader(train_data, batch_size=8)

    test_data = get_linear_data(200, len_seq, 1010, 1050)
    test_loader = DataLoader(test_data, batch_size=8)

    optimizer = Adam(model.parameters(), lr=2e-3)
    criterion = nn.MSELoss()
    t_losses = []
    v_losses = []

    for epoch in range(10):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)

            loss = criterion(pred.flatten(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        t_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in test_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred.flatten(), y)
                val_loss += loss.item()

            val_loss /= len(test_loader)
        v_losses.append(val_loss)
        print(f"[{epoch + 1:<2}/10] train_loss = {train_loss:.3f} test_loss = {val_loss:.3f}")