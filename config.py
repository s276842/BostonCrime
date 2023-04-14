__author__ = 'Fabio'

from pydantic import BaseModel, Field, BaseSettings
from torch import cuda
from torch.nn import MSELoss

class ModelConfig(BaseModel):
    name: str = Field('TimeSeriesTransformer', description='name of the model')
    input_size: int = Field(1, description='dimension of the input')
    d_model: int = Field(32, description='size of the hidden layer')
    dim_feedforward: int = Field(64, description='Dimension of the ff layer in the transformer')
    n_att_heads: int = Field(1, description="Number of attention heads")
    depth: int = Field(1, description="Number of transformer blocks")
    max_seq_len: int = Field(200, description="Lengths of longest seq supported")
    out_size: int = Field(1, description="Dimension of output embeddings")
    saves: str = Field('model_saves')


class DataConfig(BaseModel):
    pass




class Losses:
    mse: MSELoss

class TrainConfig(BaseModel):
    epochs: int = Field(100, description="Max number of epochs in training.")
    batch_size: int = Field(4, description="Batch size")
    lr: float = Field(1e-3, description="Learning rate")
    patience: int = Field(20, description="Num. epochs without performance improvement before early stopping the training.")
    delta: float = Field(0.001, description="Min value to consider that performance actually improved.")
    p_dropout: float = Field(0.1, description="Droput prob. to apply during trainig.")


class Config(BaseSettings):
    pass