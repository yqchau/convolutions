import torch
import torch.nn.functional as F

from conv1d import conv1d


def test_conv1d():

    # parameters
    batch_size = 10
    in_chans, out_chans = 2, 10
    L, l = 10, 5
    stride = 2

    input = torch.randint(0, 5, (batch_size, in_chans, L))
    kernel = torch.randint(-5, 5, (out_chans, in_chans, l))

    output = conv1d(input, kernel, stride=stride)
    torch_output = F.conv1d(input, kernel, stride=stride)

    assert torch.sum(torch_output - output) == 0
