from mnasnet import MnasNet
from nasnet import NASNetALarge
from nasnet_mobile import NASNetAMobile
from mobilenet_v2 import MobileNetV2
from torchvision import models
from torch.autograd import Variable
import numpy as np
import torch

def get_model_parm_flops(model, padding=False):

    # prods = {}
    # def save_prods(self, input, output):
        # print 'flops:{}'.format(self.__class__.__name__)
        # print 'input:{}'.format(input)
        # print '_dim:{}'.format(input[0].dim())
        # print 'input_shape:{}'.format(np.prod(input[0].shape))
        # grads.append(np.prod(input[0].shape))

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    multiply_adds = False
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)


    list_linear=[] 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[] 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[] 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)



    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(simple_hook)
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    if padding:
        input = Variable(torch.rand(3,331,331).unsqueeze(0), requires_grad = True)
    else:
        input = Variable(torch.rand(3,224,224).unsqueeze(0), requires_grad = True)
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear)) 
    total_flops += (sum(list_bn) + sum(list_relu) + sum(list_pooling))
    total_memory = (sum(list_1))

    return total_flops, total_memory

def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total 

if __name__ == '__main__':
    model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16', 'MnasNet',  'NasNet-A-Large', 'NasNet-A-Mobile', 'MobileNet-v2']
    models = [models.resnet18(), models.resnet34(), models.resnet50(), models.resnet101(), models.resnet152(), models.vgg16(),  MnasNet(), NASNetALarge(), NASNetAMobile(), MobileNetV2()]

    for i, model_name in enumerate(model_names):
        if model_name.find('NasNet-A-Large')>=0:
            total_flops, total_memory = get_model_parm_flops(models[i], True)
        else:
            total_flops, total_memory = get_model_parm_flops(models[i])
        total_param = get_model_parm_nums(models[i])
        print('%s, %.2f, %.2f, %.2f, %.2f, %.2f' %(model_name, total_flops/1e9, total_memory/1e6, total_param/1e6, total_flops/(total_memory+total_param), total_flops*256/(total_memory*256+total_param)))
        #print('%s:' %(model_name))
        #print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))
        #print('  + Number of feature map memory access: %.2fM' % (total_memory / 1e6))
        #print('  + Number of parameters: %.2fM' % (total_param / 1e6))
        #print('  + Operation Intensity (bs=1): %.2f' % (total_flops/(total_memory+total_param)))
        #print('  + Operation Intensity (bs=256): %.2f' % (total_flops*256/(total_memory*256+total_param)))
