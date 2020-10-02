"""
Implementation of `Array.__array_ufunc__`.
"""

import numpy as np

from yunit.old_array.array import Array, UnitError
from yunit.old_array.quantity import Quantity
from yunit.type_aliases import Either, ArrayLike
from yunit.old_unit import OldUnitABC, dimensionless


def array_ufunc(
    self: Array, ufunc: np.ufunc, method: str, *inputs, **ufunc_kwargs
) -> Either[Array, ArrayLike]:

    check_if_implemented(self, ufunc, method)
    other_data, other_display_unit = parse_inputs(self, inputs, ufunc)
    new_display_unit = combine_units(self, other_display_unit, ufunc)
    in_place = is_in_place(self, ufunc_kwargs)
    new_data = apply_ufunc(self, other_data, ufunc, ufunc_kwargs, in_place)
    return create_output(self, new_data, new_display_unit, in_place)


def check_if_implemented(self: Array, ufunc: np.ufunc, method: str):
    if method != "__call__":
        # method == "reduce", "accumulate", ...
        raise NotImplementedError(
            f'`{self.__class__.__name__}` does not yet support "{method}" ufuncs. '
            + self._DIY_help_text
        )
    if ufunc not in (np.add, np.subtract, np.multiply, np.divide):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` does not yet support "
            f"the NumPy ufunc `{ufunc.__name__}`. {self._DIY_help_text}"
        )


def parse_inputs(self: Array, inputs, ufunc: np.ufunc):
    # - For unary operators (cos, sign, ..), `len(inputs)` is 1.
    # - For binary operators (+, >, ..), it is 2.
    # - `input[0] == self`.
    #   (So we could make `__array_ufunc__` a static method -- as we don't need the
    #   `self` first argument` -- but NumPy doesn't like that).
    other = inputs[1]
    if isinstance(other, Array):  # (8 mV) * (1 pF)
        other_data = other.data
        other_display_unit = other.display_unit
    elif isinstance(other, OldUnitABC):  # (8 mV) * pF
        if ufunc in (np.add, np.subtract):
            raise UnitError(
                f"Cannot {ufunc.__name__} a `{OldUnitABC.__name__}` "
                f"and a `{self.__class__.__name__}`."
            )
        other_display_unit = other
        other_data = other_display_unit.data_scale
        #   Treat eg `pF` as if it was `1 * pF`.
    else:  # `other` is purely numeric (scalar or array-like): (8 mV) * 2
        other_data = other
        other_display_unit = dimensionless
    return other_data, other_display_unit


def combine_units(self: Array, other_display_unit: OldUnitABC, ufunc: np.ufunc) -> OldUnitABC:
    if ufunc in (np.add, np.subtract):
        if self.display_unit.data_unit != other_display_unit.data_unit:
            # 1*mV + (3*nS)
            raise UnitError(
                f"Cannot {ufunc.__name__} incompatible units {self.display_unit} "
                f"and {other_display_unit}."
            )
        # 1*mV + (3*volt)
        # Copy units from operand with largest units (nV + mV -> mV)
        if self.display_unit.data_scale > other_display_unit.data_scale:
            new_display_unit = self.display_unit
        else:
            new_display_unit = other_display_unit
    elif ufunc == np.multiply:
        # 1*mV * (3*nS)
        new_display_unit = self.display_unit * other_display_unit
    elif ufunc == np.divide:
        # 1*mV / (3*nS)
        new_display_unit = self.display_unit / other_display_unit
    return new_display_unit


def is_in_place(self: Array, ufunc_kwargs) -> bool:
    """
    Whether this __array_ufunc__ call is an in-place operation such as `array *= 2`.
    [See `numpy.lib.mixins._inplace_binary_method`, which added the `out` kwarg].
    """
    return ufunc_kwargs.get("out") is self


def apply_ufunc(
    self: Array, other_data: np.ndarray, ufunc: np.ufunc, ufunc_kwargs, in_place: bool
) -> np.ndarray:

    if in_place:
        ufunc_kwargs.update(out=self.data)

    return ufunc(self.data, other_data, **ufunc_kwargs)
    #   `ufunc` will return a pointer to the result even when `out` is given.


def create_output(
    self: Array, new_data: np.ndarray, new_display_unit: OldUnitABC, in_place: bool
):
    if new_display_unit == dimensionless:
        if new_data.ndim == 0:
            return new_data.item()
        else:
            return new_data
        # Note that when this was an in-place operation, we will mutate the LHS
        # variable from an Array to a purely numeric value.
    elif in_place:
        self.display_unit = new_display_unit
        return self
    else:
        # Create a new Array or Quantity.
        if new_data.size == 1:
            return Quantity(
                new_data, new_display_unit, value_is_given_in_display_units=False
            )
        else:
            return Array(
                new_data, new_display_unit, data_are_given_in_display_units=False
            )
