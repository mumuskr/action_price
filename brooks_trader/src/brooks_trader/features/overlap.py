"""Price-range overlap calculations for adjacent bars."""


def calculate_overlap(
    current_high: float,
    current_low: float,
    previous_high: float,
    previous_low: float,
) -> float:
    """Return adjacent-bar range overlap divided by their combined price span.

    COMPUTATIONAL_PROXY: range intersection over range union. This is a transparent
    overlap score in ``[0, 1]``; it is not a formula specified by Brooks.
    """
    if current_high < current_low or previous_high < previous_low:
        raise ValueError("bar high cannot be below bar low")
    union = max(current_high, previous_high) - min(current_low, previous_low)
    if union == 0:
        return 1.0
    intersection = max(
        0.0,
        min(current_high, previous_high) - max(current_low, previous_low),
    )
    return intersection / union
