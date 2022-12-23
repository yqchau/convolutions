import torch


def conv1d_single_channel_single_batch(
    input: torch.tensor, kernel: torch.tensor, stride: int
):
    """
    Single Channel, Single Batch implementation of 1D convolution.

    Input Shape: (L,)
    Kernel Shape: (l,)
    Output Shape: ((L-l)/stride + 1,)
    """
    (L,) = input.shape
    (l,) = kernel.shape
    output = torch.zeros((L - l) // stride + 1)

    for i in range(0, L - l + 1, stride):
        window = input[i : i + l]
        output[i // stride] = torch.sum(kernel * window)

    return output


def conv1d_single_batch(input: torch.tensor, weights: torch.tensor, stride: int):
    """
    Multi Channel, Single Batch implementation of 1D convolution.

    Input Shape: (C_in, L)
    Weights Shape: (C_out, C_in, L)
    Output Shape: (C_out, (L-l)/stride + 1)
    """
    in_chans, L = input.shape
    out_chans, in_chans, l = weights.shape
    output = torch.zeros(out_chans, (L - l) // stride + 1)

    for out_chan in range(out_chans):
        chan_weights = weights[out_chan]
        for in_chan in range(in_chans):
            single_chan_input = input[in_chan]
            kernel = chan_weights[in_chan]
            output[out_chan] += conv1d_single_channel_single_batch(
                single_chan_input, kernel, stride
            )

    return output


def conv1d(input: torch.tensor, weights: torch.tensor, stride: int):
    """
    Multi Channel, Multi Batch implementation of 1D convolution.

    Input Shape: (B, C_in, L)
    Weights Shape: (C_out, C_in, l)
    Output Shape: (B, C_out, (L-l)/stride + 1)
    """
    batch_size, in_chans_input, L = input.shape
    out_chans, in_chans_weights, l = weights.shape

    assert len(input.shape) == 3
    assert len(weights.shape) == 3
    assert in_chans_input == in_chans_weights
    assert L >= l

    output = torch.zeros(
        batch_size,
        out_chans,
        (L - l) // stride + 1,
    )
    for batch in range(batch_size):
        output[batch] = conv1d_single_batch(input[batch], weights, stride=stride)

    return output
