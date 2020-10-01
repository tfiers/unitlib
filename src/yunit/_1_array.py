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
