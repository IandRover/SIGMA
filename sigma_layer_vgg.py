import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F
from utils import get_activation_function

class SigmaLinear(nn.Module):
    def __init__(self, args, in_features, out_features):
        super(SigmaLinear, self).__init__()
        self.forward_layer = nn.Linear(in_features, out_features, bias=True)
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
        self.forward_layer = nn.Conv2d(in_channels, out_channels, conv_kernel, stride=conv_stride, padding=conv_padding, bias=True)
        self.max_pool = nn.MaxPool2d(pool_kernel, stride=pool_stride, padding=pool_padding)
        self.backward_layer = nn.Conv2d(out_channels, in_channels, conv_kernel, padding=conv_padding, bias=False)
        self.upsample = nn.Upsample(scale_factor=pool_kernel, mode='nearest')
        # self.backward_bn1 = nn.BatchNorm2d(out_channels)
        self.backward_bn = nn.BatchNorm2d(in_channels)
        self.forward_act = get_activation_function(args)
        self.backward_act = get_activation_function(args)

        print("in_channels", in_channels, "out_channels", out_channels)
        self.backward_gn2 = nn.GroupNorm(2, out_channels)
        if in_channels == 3:
            self.backward_gn1 = nn.GroupNorm(1, in_channels)
        else:
            self.backward_gn1 = nn.GroupNorm(2, in_channels)
        # self.backward_in = nn.InstanceNorm2d(out_channels)
        # self.backward_in2 = nn.InstanceNorm2d(in_channels)

        # self.hooked_features = 0

    def get_parameters(self):
        self.forward_params = list(self.forward_layer.parameters())
        self.backward_params = list(self.backward_layer.parameters())
        return self.forward_params, self.backward_params
            
    def forward(self, x, detach_grad=False):
        if detach_grad: x = x.detach()
        x = self.forward_act(self.forward_layer(x))
        x = self.max_pool(x)
        return x
        
    def reverse(self, x, detach_grad=False):
        if detach_grad: x = x.detach()
        x = self.upsample(x)        
        # x = self.backward_act(self.backward_bn(self.backward_layer(x)))
        x = self.backward_act(self.backward_gn1(self.backward_layer(x)))

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
    
class SigmaModel_SimpleCNN(nn.Module):
    def __init__(self, args):
        super(SigmaModel_SimpleCNN, self).__init__()
        if args.dataset == "CIFAR10":
            self.conv1 = SigmaConv(args, 3, 128, 3, 1, 1, 2, 2, 0)
            self.conv2 = SigmaConv(args, 128, 128, 3, 1, 1, 2, 2, 0)
            self.conv3 = SigmaConv(args, 128, 256, 3, 1, 1, 2, 2, 0)
            self.conv4 = SigmaConv(args, 256, 256, 3, 1, 1, 2, 2, 0)
            self.conv5 = SigmaConv(args, 256, 512, 3, 1, 1, 2, 2, 0)
            self.view1 = SigmaView((512, 1, 1), 512 * 1 * 1)
            self.fc1 = SigmaLinear(args, 512 * 1 * 1, 10)
                
        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params

    def forward(self, x, detach_grad=False, return_activations=True):   
        a1 = self.conv1(x, detach_grad)
        a2 = self.conv2(a1, detach_grad)
        a3 = self.conv3(a2, detach_grad)
        a4 = self.conv4(a3, detach_grad)
        a5 = self.conv5(a4, detach_grad)
        a5 = self.view1(a5, detach_grad)
        a6 = self.fc1(a5, detach_grad)
        return [a1, a2, a3, a4, a5, a6]

    def reverse(self, target, detach_grad=True, return_activations=True):
        if target.shape == torch.Size([10]): 
            target = F.one_hot(target, num_classes=10).float().to(target.device)
        b5 = self.fc1.reverse(target, detach_grad)
        b5 = self.view1.reverse(b5, detach_grad)
        b4 = self.conv5.reverse(b5, detach_grad)
        b3 = self.conv4.reverse(b4, detach_grad)
        b2 = self.conv3.reverse(b3, detach_grad)
        b1 = self.conv2.reverse(b2, detach_grad)
        return [b1, b2, b3, b4, b5, target]