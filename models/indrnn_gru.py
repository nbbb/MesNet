"""
@Time : 2021/1/15 9:29 
@Author : 犇犇
@File : indrnn_gru.py 
@Software: PyCharm
"""
from models.indrnn import IndRNNCell
import torch
from torch.nn import Parameter, ParameterList
import torch.nn as nn
import torch.nn.functional as F
from models.indrnn import check_bounds
class IndRNNCell_GRU(IndRNNCell):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="relu",
                 hidden_min_abs=0, hidden_max_abs=None,
                 hidden_init=None, recurrent_init=None,
                 gradient_clip=None):
        super(IndRNNCell_GRU,self).__init__(
            input_size, hidden_size, bias,
            nonlinearity,
            hidden_min_abs, hidden_max_abs,
            hidden_init, recurrent_init,
            gradient_clip)

        self.tanh=torch.tanh
        self.sigmoid=torch.sigmoid
        self.relu=F.relu
        '''parameters'''

        self.weight_iz = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hz = Parameter(torch.Tensor(hidden_size))
        if bias:
            self.bias_z = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_z', None)

        self.weight_ir = Parameter(torch.Tensor(hidden_size,input_size))
        self.weight_hr = Parameter(torch.Tensor(hidden_size))
        if bias:
            self.bias_r = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_r', None)

        # self.weight_hpiao=Parameter(torch.Tensor(hidden_size, input_size+hidden_size))
        self.weight_ihpiao = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hhpiao = Parameter(torch.Tensor(hidden_size))
        if bias:
            self.bias_hpiao = Parameter(torch.Tensor(hidden_size))

        else:
            self.register_parameter('bias_hpiao', None)
        '''gradient_clip'''
        if gradient_clip:
            if isinstance(gradient_clip, tuple):
                min_g, max_g = gradient_clip
            else:
                max_g = gradient_clip
                min_g = -max_g
            self.weight_iz.register_hook(
                lambda x: x.clamp(min=min_g, max=max_g))
            self.weight_hz.register_hook(
                lambda x: x.clamp(min=min_g, max=max_g))
            if bias:
                self.bias_z.register_hook(
                    lambda x: x.clamp(min=min_g, max=max_g))

            self.weight_ir.register_hook(
                lambda x: x.clamp(min=min_g, max=max_g))
            self.weight_hr.register_hook(
                lambda x: x.clamp(min=min_g, max=max_g))
            if bias:
                self.bias_r.register_hook(
                    lambda x: x.clamp(min=min_g, max=max_g))

            self.weight_ihpiao.register_hook(
                lambda x: x.clamp(min=min_g, max=max_g))
            self.weight_hhpiao.register_hook(
                lambda x: x.clamp(min=min_g, max=max_g))

            if bias:
                self.bias_hpiao.register_hook(
                    lambda x: x.clamp(min=min_g, max=max_g))
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "bias" in name:
                weight.data.zero_()
            elif "weight_h" in name:
                if self.recurrent_init is None:
                    nn.init.constant_(weight, 1)
                else:
                    self.recurrent_init(weight)
            elif "weight_i" in name:
                if self.hidden_init is None:
                    nn.init.normal_(weight, 0, 0.01)
                else:
                    self.hidden_init(weight)
            else:
                weight.data.normal_(0, 0.01)
                # weight.data.uniform_(-stdv, stdv)
        self.check_bounds()

    def check_bounds(self):
        self.weight_hz.data = check_bounds(
            self.weight_hz.data, self.hidden_min_abs, self.hidden_max_abs
        )
        self.weight_hr.data = check_bounds(
            self.weight_hr.data, self.hidden_min_abs, self.hidden_max_abs
        )
        self.weight_hhpiao.data = check_bounds(
            self.weight_hhpiao.data, self.hidden_min_abs, self.hidden_max_abs
        )
    def forward(self, input, hx):

        zt=self.sigmoid(F.linear(input,self.weight_iz,self.bias_z)+torch.mul(self.weight_hz,hx))
        rt = self.sigmoid(F.linear(input, self.weight_ir, self.bias_r) + torch.mul(self.weight_hr, hx))
        # zt = self.relu(F.linear(input, self.weight_iz, self.bias_z) + torch.mul(self.weight_hz, hx))
        # rt = self.relu(F.linear(input, self.weight_ir, self.bias_r) + torch.mul(self.weight_hr, hx))
        #####################################
        hpiao=F.linear(input,self.weight_ihpiao,self.bias_hpiao)
        #hpiao=rt*(hpiao+torch.mul(self.weight_hhpiao,hx))
        hpiao = hpiao + rt*torch.mul(self.weight_hhpiao, hx)
        hpiao=self.tanh(hpiao)
        # hpiao = self.relu(hpiao)
        ################################
        ht=(1-zt)*hx+zt*hpiao
        # zt = self.sigmoid(F.linear(torch.cat((hx,input),dim=1), self.weight_iz, self.bias_z))
        # rt = self.sigmoid(F.linear(torch.cat((hx,input),dim=1), self.weight_ir, self.bias_r) )
        # hpiao = self.tanh(F.linear(torch.cat((rt * hx, input), dim=1), self.weight_hpiao, self.bias_hpiao))
        # ht = (1 - zt) * hx + zt * hpiao
        return ht




