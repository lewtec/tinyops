from tinygrad import Tensor


def einsum(subscripts: str, *operands: Tensor) -> Tensor:
    """
    Evaluates the Einstein summation convention on the operands.

    This implementation extends the standard `tinygrad.Tensor.einsum` by adding support
    for repeated indices within a single operand (e.g., 'ii->i'), which implies
    extracting the diagonal. It preprocesses the inputs to handle these cases explicitly
    using slicing and `arange` before passing the simplified expression to the backend.

    Args:
        subscripts: Specifies the subscripts for summation as comma-separated list of subscript labels.
            An implicit (classical Einstein summation) calculation is performed unless the explicit
            indicator '->' is included as well as subscript labels of the precise output form.
        *operands: The tensors to compute the einsum of.

    Returns:
        The calculation based on the Einstein summation convention.

    Raises:
        ValueError: If the number of operands does not match the subscripts, or if dimensions
            corresponding to the same label mismatch.

    Example:
        >>> a = Tensor.eye(3)
        >>> einsum('ii->i', a).numpy()
        array([1., 1., 1.], dtype=float32)
    """
    if "->" in subscripts:
        input_str, output_str = subscripts.split("->")
        explicit = True
    else:
        input_str, output_str, explicit = subscripts, "", False
    input_subs = input_str.split(",")
    if len(input_subs) != len(operands):
        raise ValueError("Subscripts mismatch")
    new_operands: list[Tensor] = []
    new_input_subs: list[str] = []
    for sub, op in zip(input_subs, operands):
        sub = sub.strip()
        while True:
            counts = {c: sub.count(c) for c in set(sub) if c.isalpha()}
            repeated = [c for c, count in counts.items() if count > 1]
            if not repeated:
                break
            c = repeated[0]
            idxs = [i for i, char in enumerate(sub) if char == c]
            dim = op.shape[idxs[0]]
            for i in idxs[1:]:
                if op.shape[i] != dim:
                    raise ValueError(f"Dim mismatch for {c}")
            arange_idx = Tensor.arange(dim)
            indexer = [slice(None)] * op.ndim
            for i in idxs:
                indexer[i] = arange_idx
            op = op[tuple(indexer)]
            sorted_idxs = sorted(idxs)
            is_contiguous = all(sorted_idxs[k] == sorted_idxs[k - 1] + 1 for k in range(1, len(sorted_idxs)))
            new_chars = []
            if is_contiguous:
                remove_idxs = set(sorted_idxs[1:])
                for i, char in enumerate(sub):
                    if i in remove_idxs:
                        continue
                    new_chars.append(char)
            else:
                new_chars.append(c)
                for i, char in enumerate(sub):
                    if i not in idxs:
                        new_chars.append(char)
            sub = "".join(new_chars)
        new_operands.append(op)
        new_input_subs.append(sub)
    new_subscripts = ",".join(new_input_subs)
    if explicit:
        new_subscripts += "->" + output_str
    return Tensor.einsum(new_subscripts, *new_operands)
