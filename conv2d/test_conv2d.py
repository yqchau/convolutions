import torch
import torch.nn.functional as F

from conv2d.conv2d import conv2d


def test_conv2d():

    # parameters
    batch_size = 10
    in_chans, out_chans = 2, 10
    h, w = 10, 10
    H, W = 5, 5
    stride = 2

    input = torch.randint(0, 5, (batch_size, in_chans, h, w))
    kernel = torch.randint(-5, 5, (out_chans, in_chans, H, W))

    output = conv2d(input, kernel, stride=stride)
    torch_output = F.conv2d(input, kernel, stride=stride)

    assert torch.sum(torch_output - output) == 0
