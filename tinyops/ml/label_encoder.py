import numpy as np
from tinygrad import Tensor


def label_encoder(y: Tensor) -> Tensor:
    # "Fit" step: Find unique labels. Using numpy is a pragmatic choice for this
    # operation, as a pure-tensor unique is complex. This is analogous to
    # sklearn's .fit() method.
    unique_labels_np = np.unique(y.numpy())
    unique_labels = Tensor(unique_labels_np, requires_grad=False, device=y.device)

    # "Transform" step: Map original labels to their indices in the unique list.
    # This is done with pure tensor operations to stay within the computation graph.
    # Use broadcasting to create a comparison matrix: (N, 1) == (1, C) -> (N, C)
    comparison_matrix = y.unsqueeze(1) == unique_labels.unsqueeze(0)

    # The index of the 'True' value in each row is the encoded label.
    encoded = comparison_matrix.argmax(axis=1)

    return encoded.cast(y.dtype)
