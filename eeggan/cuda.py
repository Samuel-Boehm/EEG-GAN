#  Author: Kay Hartmann <kg.hartma@gmail.com>
#          Samuel Böhm  <samuel-boehm@web.de>

import os

import torch

import GPUtil


activate_cuda: bool = False


def init_cuda():
    global activate_cuda
    activate_cuda = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def set_activate_cuda(use_cuda: bool):
    global activate_cuda
    activate_cuda = use_cuda


def get_activate_cuda():
    global activate_cuda
    return activate_cuda


def to_cuda(*elements):
    global activate_cuda
    if activate_cuda:
        ret = [element.cuda() for element in elements]
    else:
        ret = [element.cpu() for element in elements]

    if len(ret) == 1:
        ret = ret[0]
    return ret


def to_device(device, *elements):
    return [element.to(device) for element in elements]


def display_GPU_load(infostring:str = 'GPU Info'):
    
    GPU_idx = torch.cuda.current_device()
    
    GPUs = GPUtil.getGPUs()

    print(infostring)
    print('GPU ID: ', GPU_idx)
    print(f'Used Memory {GPUs[GPU_idx].memoryUsed} / {GPUs[GPU_idx].memoryTotal} -> Free: {GPUs[GPU_idx].memoryFree}')
    print(' ')