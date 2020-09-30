# See `array.pyi` for additional typing information.

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from ..unit import Unit


class UnitError(Exception):
    pass


class Array(NDArrayOperatorsMixin):
    """
    A NumPy array with a physical unit.

    This class is a wrapper around a NumPy `ndarray` (the `data` attribute), augmented
    with a `Unit` (the `display_unit` attribute). The data is stored -- and calculations
    with the data are done -- in the `data_unit` of this `display_unit`. The
    `display_unit` is only used at Array creation time and when printing the Array.
    """

    # See "Writing custom array containers"[1] from the NumPy manual for info on this
    # class's `__array_ufunc__` and `__array_function__` methods.
    #
    # `NDArrayOperatorsMixin` implements Python dunder methods like `__mul__` and
    # `__imul__` so that we can use standard Python syntax like `*` and `*=` with our
    # `Array`s. (`NDArrayOperatorsMixin` implements these by calling the corresponding
    # NumPy ufuncs [like `np.multiply`], which in turn defer to our `__array_ufunc__`
    # method).
    #
    # [1](https://numpy.org/doc/stable/user/basics.dispatch.html)
    #

    data: np.ndarray
    display_unit: Unit

    def __init__(
        self,
        data,
        display_unit: Unit,
        data_are_given_in_display_units: bool = False,
    ):
        """
        :param data:  Array-like.
        :param display_unit:  Units in which to display the data.
        :param data_are_given_in_display_units:  If True, the given `data` is taken to
                    be expressed in `display_unit`s, and is converted to and stored
                    internally in `display_unit.data_unit`s. If False (default), `data`
                    is taken to be already expressed in `display_unit.data_unit`s, and
                    no conversion is done.
        """
        data_as_array = np.asarray(data)
        if not issubclass(data_as_array.dtype.type, np.number):
            raise TypeError(f'Input data must be numeric. Got "{repr(data_as_array)}"')
        if data_are_given_in_display_units:
            self.data = data_as_array * display_unit.data_scale
        else:
            self.data = data_as_array
        self.display_unit = display_unit

    @property
    def data_unit(self):
        return self.display_unit.data_unit

    @property
    def data_in_display_units(self) -> np.ndarray:
        return self.data / self.display_unit.data_scale

    dd: np.ndarray = data_in_display_units  # Shorthand

    #
    #
    # -------------------
    # Text representation

    def __str__(self):
        return format(self)

    def __format__(self, format_spec: str) -> str:
        # When no spec is given -- as is the case for `format(array)` and
        # `f"f-strings such as this one, {array}"` -- Python calls this `__format__`
        # method with `format_spec = ""`
        if not format_spec:
            format_spec = ".4G"
        array_string = np.array2string(
            self.data_in_display_units,
            formatter={"float_kind": lambda x: format(x, format_spec)},
        )
        return f"{array_string} {self.display_unit}"

    __repr__ = __str__

    #
    #
    # --------------------------------------------
    # Elementwise operations (+, >, cos, sign, ..)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **ufunc_kwargs):
        # Delegate implementation to a separate module, to keep this file overview-able.
        from .array_ufunc import array_ufunc

        return array_ufunc(self, ufunc, method, *inputs, **ufunc_kwargs)

    #
    #
    # ------------------------------
    # NumPy methods (mean, sum, ...)
    #

    def __array_function__(self, func, _types, _args, _kwargs):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` does not yet support being used "
            f"with function `{func.__name__}`. {self._DIY_help_text}"
        )

    _DIY_help_text = (  # Shown when a NumPy operation is not implemented yet for our Array.
        "You can get the bare numeric data (a plain NumPy array) "
        "via `array.data`, and work with it manually."
    )
