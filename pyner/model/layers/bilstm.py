#encoding:utf-8
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from .model_utils import prepare_pack_padded_sequence

class BILSTM(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layer,
                 input_size,
                 dropout_p,
                 dropout_r,
                 num_classes,
                 bi_tag):

        super(BILSTM,self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.dropout_r = dropout_r


        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layer,
                            batch_first = True,
                            dropout = dropout_p,
                            bidirectional = bi_tag)

    def forward(self,inputs,length):
        # 去除padding元素
        # embeddings_packed: (batch_size*time_steps, embedding_dim)
        # 在使用`pad_packed_sequence`的时候，输入的mini-batch的序列的长度必须是从长到短排序好的

        inputs, length, desorted_indice = prepare_pack_padded_sequence(inputs, length)
        embeddings_packed = pack_padded_sequence(inputs, length, batch_first=True)
        output, _ = self.lstm(embeddings_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[desorted_indice]
        output = F.dropout(output, p=self.dropout_r, training=self.training)
        return output