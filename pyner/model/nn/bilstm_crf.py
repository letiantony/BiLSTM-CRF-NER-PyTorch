#encoding:utf-8
import torch.nn as nn
from ..layers.embed_layer import Embed_Layer
from ..layers.crf import CRF
from ..layers.bilstm import BILSTM
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,model_config,
                 embedding_dim,
                 num_classes,
                 vocab_size,
                 embedding_weight,
                 device):
        super(Model ,self).__init__()
        self.embedding = Embed_Layer(vocab_size = vocab_size,
                                     embedding_weight = embedding_weight,
                                     embedding_dim = embedding_dim,
                                     dropout_emb=model_config['dropout_emb'],
                                     training=True)
        self.lstm = BILSTM(input_size = embedding_dim,
                           hidden_size= model_config['hidden_size'],
                           num_layer  = model_config['num_layer'],
                           bi_tag     = model_config['bi_tag'],
                           dropout_p  = model_config['dropout_p'],
                           dropout_r = model_config['dropout_r'],
                           num_classes= num_classes) # bilstm的类别个数新增两个
        self.dropout_p = model_config['dropout_p']
        bi_num = 2 if model_config['bi_tag'] else 1
        self.linear = nn.Linear(in_features=model_config['hidden_size']*bi_num, out_features=num_classes)
        nn.init.xavier_uniform(self.linear.weight)
        self.crf = CRF(device = device,tagset_size=num_classes)


    def forward(self, inputs,length):
        x = self.embedding(inputs)
        x = self.lstm(x,length)
        output = F.tanh(x)
        logit = self.linear(output)
        output = F.dropout(logit,p=self.dropout_p, training=self.training)
        return output

