"""
@Time : 2021/1/14 9:31
@Author : 犇犇
@File : decoder_indrnn.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.indrnn_gru import IndRNNCell_GRU
from models.temporal_attention import TemporalAttention

class IndRNN_Decoder(nn.Module):
    def __init__(self,num_layers,num_directions, feat_size, feat_len, embedding_size,
                 hidden_size, attn_size, output_size, rnn_dropout,
                 batch_norm=False,hidden_inits=None,recurrent_inits=None,**kwargs):
        super(IndRNN_Decoder, self).__init__()
        self.num_layers= num_layers
        self.num_directions = num_directions
        self.feat_size = feat_size
        self.feat_len = feat_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.output_size = output_size
        self.rnn_dropout_p = rnn_dropout
        self.batch_norm=batch_norm

        self.dropout = nn.Dropout(p=self.rnn_dropout_p)
        self.activation = F.relu
        self.attention_cnn = TemporalAttention(
            hidden_size=self.num_directions * self.hidden_size,
            feat_size=self.feat_size[0],
            bottleneck_size=self.attn_size)

        if len(self.feat_size)>=2:
            self.attention_i3d = TemporalAttention(
                hidden_size=self.num_directions * self.hidden_size,
                feat_size=self.feat_size[1],
                bottleneck_size=self.attn_size)

        cells=[]
        cells_bi=[]
        for i in range(self.num_layers):
            if recurrent_inits is not None:
                kwargs["recurrent_init"] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs["hidden_init"] = hidden_inits[i]

            if i == 0:
                in_size = sum(self.feat_size)+ \
                          self.embedding_size
            else :
                in_size=hidden_size * num_directions
            cells.append(IndRNNCell_GRU(in_size, hidden_size, **kwargs))
            cells_bi.append(IndRNNCell_GRU(in_size, hidden_size, **kwargs))
        self.cells = nn.ModuleList(cells)
        self.cells_bi = nn.ModuleList(cells_bi)
        if batch_norm:
            lns = []

            for i in range(self.num_layers):
                 lns.append(nn.LayerNorm(hidden_size * num_directions))
            self.lns = nn.ModuleList(lns)

        self.out = nn.Linear(self.num_directions * self.hidden_size, self.output_size)
        self.hid_w = nn.Linear(self.hidden_size, 1)
        self.hid_fuse=nn.Linear(num_layers* self.hidden_size, self.hidden_size)
    def get_last_hidden(self, hidden):
        last_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        last_hidden = last_hidden.view(self.num_layers, self.num_directions,
                                       last_hidden.size(1), last_hidden.size(2))
        last_hidden = last_hidden.transpose(2, 1).contiguous()
        last_hidden = last_hidden.view(self.num_layers, last_hidden.size(1),
                                       self.num_directions * last_hidden.size(3))
        # # last_hidden = last_hidden[-1]
        # last_hidden=self.hid_fuse(last_hidden.transpose(0, 1).reshape(last_hidden.size(1),-1).contiguous())
        last_hidden=last_hidden.transpose(0, 1).contiguous()
        weights_h =F.softmax( torch.tanh(self.hid_w(last_hidden)),dim=1)
        last_hidden = (last_hidden*weights_h).sum(1)
        return last_hidden

    def Big_zhuanpan(self, hidden):
        mid_layer_num=int(self.num_layers/2)
        out = torch.zeros_like(hidden)
        out[mid_layer_num:,:,:]=hidden[:-mid_layer_num,:,:]
        out[:mid_layer_num,:,:]=hidden[-mid_layer_num:,:,:]
        return out
    def forward(self, embedded, hidden, feats):
        last_hidden = self.get_last_hidden(hidden)
        last_hidden= self.dropout(last_hidden)
        feats = self.dropout(feats)
        feats_cnn=feats[:,:,:self.feat_size[0]]
        feats_i3d=feats[:,:,self.feat_size[0]:] #-self.feat_size[2]
        feats_cnn, attn_weights = self.attention_cnn(last_hidden, feats_cnn)
        if len(self.feat_size)>=2:
            feats_i3d, attn_weights = self.attention_i3d(last_hidden, feats_i3d)
        else:
            feats_i3d=feats_i3d.mean(1)
        # feats_audio = feats[:, :, -self.feat_size[2]:].mean(1)
        input_combined = torch.cat((embedded.squeeze(0), feats_cnn,feats_i3d), dim=1)

        input_combined=self.dropout(input_combined)
        if self.num_layers>1:
            hidden=self.Big_zhuanpan(hidden)
        hidden_cells=hidden[:,:,:self.hidden_size*1]
        hidden_cells_bi=hidden[:,:,self.hidden_size:]
        hidden_current=[]
        x=input_combined
        for i,cell in enumerate(self.cells):
            h_cell=cell(x,hidden_cells[i])
            if self.num_directions>=2:
                h_cell_bi = self.cells_bi[i](x, hidden_cells_bi[i])
                h_cell = torch.cat([h_cell, h_cell_bi], 1)
            hidden_current.append(h_cell)

            if self.batch_norm:
                # h_cell = self.bns[i](h_cell)
                h_cell = self.lns[i](h_cell)

            if i>3 and i%3==1:
                x = h_cell + x
            else:
                x=h_cell
            x=self.dropout(x)

        output = self.out(x)
        output = F.log_softmax(output, dim=1)
        return output, torch.stack(hidden_current, 0), None