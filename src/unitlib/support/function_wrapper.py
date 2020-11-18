from functools import wraps
from typing import Callable

import numpy as np
from unitlib import Array


def add_unit_support(
    function: Callable[[np.ndarray, "..."], "..."]
) -> Callable[[Array, "..."], "..."]:
    """
    Function wrapper that allows Unitlib to be used with libraries that can only handle
    NumPy arrays (and not Unitlib `Array`s), such as SciPy or Numba in 'nopython' (i.e.
    fast) mode.

    Example:
        from scipy.signal import find_peaks as find_peaks_orig
        from unitlib import add_unit_support

        find_peaks = add_unit_support(find_peaks_orig)
        peaks, _ = find_peaks(voltage_array, height=10 * mV)

    This wrapper replaces all Unitlib arguments by their `.data` when the function is
    called (and leaves arguments of other types as is).
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
