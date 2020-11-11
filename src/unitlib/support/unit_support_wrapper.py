from functools import wraps
from typing import Callable

import numpy as np
from unitlib import Array


def add_unit_support(
    function: Callable[[np.ndarray, "..."], "..."]
) -> Callable[[Array, "..."], "..."]:
    """
    Allow unitlib to be used with libraries that have no knowledge of unitlib `Array`s,
    such as SciPy.

    Example:
        from scipy.signal import find_peaks
        from unitlib import add_unit_support

        find_peaks_ = add_unit_support(find_peaks)
        peaks, _ = find_peaks_(voltage_array, height=10 * mV)
    """

    @wraps(function)
    def function_with_unit_support(*args, **kwargs):
        numpy_args = (arg.data if isinstance(arg, Array) else arg for arg in args)
        numpy_kwargs = {
            kw: arg.data if isinstance(arg, Array) else arg
            for kw, arg in kwargs.items()
        }
        return function(*numpy_args, **numpy_kwargs)

    return function_with_unit_support
