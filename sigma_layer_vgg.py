import torch
import torch.nn as nn
import torch.nn.init as init
import math
from utils import get_activation_function

class SigmaLinear(nn.Module):
    def __init__(self, args, in_features, out_features):
        super(SigmaLinear, self).__init__()
        self.forward_layer = nn.Linear(in_features, out_features, bias=False)
        self.backward_layer = nn.Linear(out_features, in_features, bias=False)
        self.backward_bn = nn.BatchNorm1d(in_features)
        
        self.forward_act = get_activation_function(args)
        self.backward_act = get_activation_function(args)
        
    def get_parameters(self):
        self.forward_params = list(self.forward_layer.parameters())
        self.backward_params = list(self.backward_layer.parameters())
        return self.forward_params, self.backward_params
    def forward(self, x, detach_grad=False):
        if detach_grad: x = x.detach()
        return self.forward_act(self.forward_layer(x))
    def reverse(self, x, detach_grad=False):
        if detach_grad: x = x.detach()
        return self.backward_act(self.backward_bn(self.backward_layer(x)))

class SigmaConv(nn.Module):
    def __init__(self, args, in_channels, out_channels, conv_kernel, conv_stride, conv_padding, pool_kernel, pool_stride, pool_padding):
        super(SigmaConv, self).__init__()
        self.forward_layer = nn.Conv2d(in_channels, out_channels, conv_kernel, stride=conv_stride, padding=conv_padding, bias=False)
        self.max_pool = nn.MaxPool2d(pool_kernel, stride=pool_stride, padding=pool_padding)
        self.backward_layer = nn.Conv2d(out_channels, in_channels, conv_kernel, padding=conv_padding, bias=False)
        self.upsample = nn.Upsample(scale_factor=pool_kernel, mode='nearest')
        self.backward_bn1 = nn.BatchNorm2d(out_channels)
        self.backward_bn = nn.BatchNorm2d(in_channels)
        self.forward_act = get_activation_function(args)
        self.backward_act = get_activation_function(args)

        self.hooked_features = 0

    def get_parameters(self):
        self.forward_params = list(self.forward_layer.parameters())
        self.backward_params = list(self.backward_layer.parameters())
        return self.forward_params, self.backward_params
            
    def forward(self, x, detach_grad=False):
        if detach_grad: x = x.detach()
        x = self.forward_act(self.forward_layer(x))
        x = self.max_pool(x)
        self.hooked_features = x.detach()
        return x

    def reverse(self, x, detach_grad=False):
        if detach_grad: x = x.detach()
        # x = x if self.hooked_features is 0 else x * (self.hooked_features > 0)
        x = self.upsample(x)
        # x = self.backward_act(self.backward_bn(self.backward_layer(self.backward_bn1(x))))
        # x = self.backward_act(self.backward_layer(self.backward_bn1(x)))
        x = self.backward_act(self.backward_bn(self.backward_layer(x)))
        return x
    
class SigmaView(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SigmaView, self).__init__()
        self.input_shape = input_shape if not isinstance(input_shape,int) else [input_shape]
        self.output_shape = output_shape if not isinstance(output_shape,int) else [output_shape]
    def forward(self, x, detach_grad=False):
        return x.view([len(x)] + list(self.output_shape))
    def reverse(self, x, detach_grad=False):
        return x.view([len(x)] + list(self.input_shape))