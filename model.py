import torch

def ConvolutionalLayer(input_channels, output_channels, kernel_size, stride, padding):
    filter_height = 3
    filter_width = 3
    weights = torch.randn(input_channels, output_channels, filter_height, filter_width)
    