from tinygrad import Tensor


def _gather_diagonal_along_positions(operand: Tensor, positions: list[int]) -> Tensor:
    """Index the shared diagonal along the given axis positions."""
    dimension_size = operand.shape[positions[0]]
    indexer_range = Tensor.arange(dimension_size)
    indexer = [slice(None)] * operand.ndim
    for position in positions:
        indexer[position] = indexer_range
    return operand[tuple(indexer)]


def _rewrite_subscript_after_diagonal(
    subscript: str,
    character: str,
    positions: list[int],
) -> str:
    """Drop collapsed axes from the subscript after a diagonal extraction."""
    sorted_positions = sorted(positions)
    contiguous = all(
        sorted_positions[k] == sorted_positions[k - 1] + 1 for k in range(1, len(sorted_positions))
    )
    if contiguous:
        removal_set = set(sorted_positions[1:])
        return "".join(char for i, char in enumerate(subscript) if i not in removal_set)

    kept = [character]
    kept.extend(char for i, char in enumerate(subscript) if i not in positions)
    return "".join(kept)


def _extract_one_repeated_index(subscript: str, operand: Tensor) -> tuple[str, Tensor] | None:
    """Extract one trace-like repeated index, or None when none remain."""
    character_counts = {c: subscript.count(c) for c in set(subscript) if c.isalpha()}
    repeated_characters = [c for c, count in character_counts.items() if count > 1]
    if not repeated_characters:
        return None

    character = repeated_characters[0]
    positions = [i for i, char in enumerate(subscript) if char == character]
    dimension_size = operand.shape[positions[0]]
    for position in positions[1:]:
        if operand.shape[position] != dimension_size:
            raise ValueError(f"Dimension mismatch for index '{character}'")

    operand = _gather_diagonal_along_positions(operand, positions)
    subscript = _rewrite_subscript_after_diagonal(subscript, character, positions)
    return subscript, operand


def _extract_diagonal(subscript: str, operand: Tensor) -> tuple[str, Tensor]:
    """Extract diagonals for trace-like repeated indices in a single operand.

    Args:
        subscript: Einstein summation notation string for a single operand.
        operand: The operand tensor.

    Returns:
        A tuple of (processed_subscript, processed_operand).
    """
    subscript = subscript.strip()
    while True:
        step = _extract_one_repeated_index(subscript, operand)
        if step is None:
            break
        subscript, operand = step
    return subscript, operand


def einstein_summation(subscripts: str, *operands: Tensor) -> Tensor:
    """Evaluate the Einstein summation convention on the operands.

    Supports both implicit and explicit mode (with ``->``). Handles
    trace-like repeated indices within a single operand by extracting
    the diagonal.

    Args:
        subscripts: Einstein summation notation string.
        *operands: Input tensors.

    Returns:
        Result of the Einstein summation.
    """
    if "->" in subscripts:
        input_string, output_string = subscripts.split("->")
        explicit = True
    else:
        input_string, output_string, explicit = subscripts, "", False

    input_subscripts = input_string.split(",")
    if len(input_subscripts) != len(operands):
        raise ValueError("Number of subscripts does not match number of operands")

    processed_operands: list[Tensor] = []
    processed_subscripts: list[str] = []

    for subscript, operand in zip(input_subscripts, operands):
        subscript, operand = _extract_diagonal(subscript, operand)
        processed_operands.append(operand)
        processed_subscripts.append(subscript)

    rebuilt = ",".join(processed_subscripts)
    if explicit:
        rebuilt += "->" + output_string

    return Tensor.einsum(rebuilt, *processed_operands)
