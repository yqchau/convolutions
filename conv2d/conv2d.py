import torch


def conv2d_single_channel_single_batch(
    input: torch.tensor, kernel: torch.tensor, stride: int
):
    """
    Single Channel, Single Batch implementation of 2D convolution.

    Input Shape: (H, W)
    Kernel Shape: (h, w)
    Output Shape: ((H-h)/stride + 1, (W-w)/stride + 1)
    """
    H, W = input.shape
    h, w = kernel.shape
    output = torch.zeros((H - h) // stride + 1, (W - w) // stride + 1)

    for i in range(0, H - h + 1, stride):
        for j in range(0, W - w + 1, stride):
            window = input[i : i + h, j : j + w]
            output[i // stride, j // stride] = torch.sum(kernel * window)

    return output


def conv2d_single_batch(input: torch.tensor, weights: torch.tensor, stride: int):
    """
    Multi Channel, Single Batch implementation of 2D convolution.

    Input Shape: (C_in, H, W)
    Weights Shape: (C_out, C_in, h, w)
    Output Shape: (C_out, (H-h)/stride + 1, (W-w)/stride + 1)
    """
    in_chans, H, W = input.shape
    out_chans, in_chans, h, w = weights.shape
    output = torch.zeros(out_chans, (H - h) // stride + 1, (W - w) // stride + 1)

    for out_chan in range(out_chans):
        chan_weights = weights[out_chan]
        for in_chan in range(in_chans):
            single_chan_input = input[in_chan]
            kernel = chan_weights[in_chan]
            output[out_chan] += conv2d_single_channel_single_batch(
                single_chan_input, kernel, stride
            )

    return output


def conv2d(input: torch.tensor, weights: torch.tensor, stride: int):
    """
    Multi Channel, Multi Batch implementation of 2D convolution.

    Input Shape: (B, C_in, H, W)
    Weights Shape: (C_out, C_in, h, w)
    Output Shape: (B, C_out, (H-h)/stride + 1, (W-w)/stride + 1)
    """
    batch_size, in_chans_input, H, W = input.shape
    out_chans, in_chans_weights, h, w = weights.shape

    assert len(input.shape) == 4
    assert len(weights.shape) == 4
    assert in_chans_input == in_chans_weights
    assert H >= h
    assert W >= w

    output = torch.zeros(
        batch_size, out_chans, (H - h) // stride + 1, (W - w) // stride + 1
    )
    for batch in range(batch_size):
        output[batch] = conv2d_single_batch(input[batch], weights, stride=stride)

    return output
