from tinygrad import Tensor, dtypes


def _pad_reflect_101(x: Tensor, padding: tuple[int, int, int, int]) -> Tensor:
    l, r, t, b = padding
    x = x[..., :, 1 : l + 1].flip(-1).cat(x, dim=-1)
    x = x.cat(x[..., :, -r - 1 : -1].flip(-1), dim=-1)
    x = x[..., 1 : t + 1, :].flip(-2).cat(x, dim=-2)
    x = x.cat(x[..., -b - 1 : -1, :].flip(-2), dim=-2)
    return x


def _apply_filter_iterative(
    x: Tensor, kernel: Tensor, scale: float, delta: float, padding_mode: str = "reflect"
) -> Tensor:
    input_dtype = x.dtype
    if input_dtype == dtypes.uint8:
        x = x.cast(dtypes.float32)

    ksize = kernel.shape[0]
    orig_shape = x.shape
    padding_val = ksize // 2

    def pad_func(t):
        if padding_mode == "reflect":
            return _pad_reflect_101(t, (padding_val, padding_val, padding_val, padding_val))
        elif padding_mode == "constant":
            return t.pad(((0, 0), (0, 0), (padding_val, padding_val), (padding_val, padding_val)))
        else:
            raise ValueError(f"Unsupported padding mode: {padding_mode}")

    if len(orig_shape) == 2:  # Grayscale (H, W)
        H, W = orig_shape
        x_reshaped = x.reshape(1, 1, H, W)
        x_padded = pad_func(x_reshaped)
        y = x_padded.conv2d(kernel.reshape(1, 1, ksize, ksize))
        y = y.reshape(H, W)

    elif len(orig_shape) == 3:  # Color (H, W, C)
        H, W, C = orig_shape
        channels = []
        for i in range(C):
            channel = x[..., i].reshape(1, 1, H, W)
            channel_padded = pad_func(channel)
            y_channel = channel_padded.conv2d(kernel.reshape(1, 1, ksize, ksize))
            channels.append(y_channel.reshape(H, W))
        y = Tensor.stack(channels).permute(1, 2, 0)

    elif len(orig_shape) == 4:  # Batch (N, H, W, C)
        N, H, W, C = orig_shape
        results = []
        for i in range(N):
            img_tensor = x[i]
            channels = []
            for j in range(C):
                channel = img_tensor[..., j].reshape(1, 1, H, W)
                channel_padded = pad_func(channel)
                y_channel = channel_padded.conv2d(kernel.reshape(1, 1, ksize, ksize))
                channels.append(y_channel.reshape(H, W))
            img_result = Tensor.stack(channels).permute(1, 2, 0)
            results.append(img_result)
        y = Tensor.stack(results)

    else:
        raise ValueError(f"Unsupported input shape: {orig_shape}")

    return y * scale + delta
