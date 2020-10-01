import numpy as np

from ._3_unit import Unit, DataUnit


class Array(np.lib.mixins.NDArrayOperatorsMixin):
    """
    A NumPy array with a physical unit.

    This class is a wrapper around a NumPy `ndarray` (the `data` attribute), augmented
    with a `Unit` (`display_unit`) and an optional descriptive string (`name`).

    The `display_unit` is only used when interfacting with the user; that is: at Array
    creation time, and when printing or plotting the Array. Internally, the data is
    stored -- and calculations with the data are done -- in `data_unit`s.

    `data_unit` is a scalar multiple or submultiple of `display_unit`. (You can access
    it through this Array's `data_unit` property, or equivalently, via
    `display_unit.data_unit`). We could for example have a "mV" display unit with a
    "volt" data unit. All arrays containing voltage data -- whether their `display_unit`
    is "mV", "μV", or "kV" -- would have their data stored in volts. (Floating point
    number representation makes this possible even for very large or small multipliers,
    like eg attoseconds).

    The advantage of this is that you can push the `data` of different Arrays through a
    pipeline of speed-optimized functions (like Numba JIT-compiled functions) without
    the overhead of unit checking and unit conversions. Because all data is in the same
    relative unit system, there will be no order-of-magnitude unit errors.

    You thus get the best of both worlds:
    - The convenience of units when inputing and displaying your data.
    - The processing speed of raw NumPy arrays and Numba JIT-compiled functions.
    """

    # See "Writing custom array containers"[1] from the NumPy manual for info on this
    # class's `__array_ufunc__` and `__array_function__` methods.
    #
    # # [1](https://numpy.org/doc/stable/user/basics.dispatch.html)

    # The base class `NDArrayOperatorsMixin` implements Python dunder methods like
    # `__mul__` and `__imul__`, so that we can use standard Python syntax like `*` and
    # `*=` with our `Array`s.
    #
    # (`NDArrayOperatorsMixin` implements these by calling the
    # corresponding NumPy ufuncs [like `np.multiply`], which in turn defer to our
    # `__array_ufunc__` method).

    data: np.ndarray
    display_unit: Unit
    name: str

    @property
    def data_unit(self) -> DataUnit:
        return self.display_unit.data_unit

    @property
    def data_in_display_units(self) -> np.ndarray:
        return self.data * self.display_unit.scale
