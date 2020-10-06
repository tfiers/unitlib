"""
Implementation of `Array.__array_ufunc__`.

This handles all Python math syntax (*, *=, **, <, %, …)
as well as all NumPy elementwise functions (`cos`, `sign`, `abs`, …),
for both Arrays, Quantities, and Units.
"""

from typing import Union

import numpy as np

from . import Array, Unit, dimensionless, Quantity


equality_comparators = (np.equal, np.not_equal)
ordering_comparators = (
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
)
comparators = equality_comparators + ordering_comparators

numeric_methods = (
    np.add,
    np.subtract,
    np.multiply,
    np.divide,
    np.power,
)  # There's more in `NDArrayOperatorsMixin`

supported_ufuncs = comparators + numeric_methods


def __array_ufunc__(
    self: Union[Unit, Quantity, Array],
    ufunc: np.ufunc,
    method: str,
    *inputs,
    **ufunc_kwargs,
) -> Union[Unit, Quantity, Array, np.ndarray, bool]:

    #
    #
    # -------------------
    # Check function type

    if method != "__call__":  # method = "reduce", "accumulate", ...
        raise NotImplementedError(
            f'yunit objects do not yet support "{method}" ufuncs. '
            + self._DIY_help_text
        )

    if ufunc not in supported_ufuncs:
        raise NotImplementedError(
            f"yunit objects do not yet support the `{ufunc.__name__}` operation. "
            + self._DIY_help_text
        )

    #
    #
    # ------------
    # Parse inputs

    # - For unary operators (cos, sign, abs, …), `len(inputs)` is 1.
    # - For binary operators (+, >, **, …), it is 2.
    # - `input[0] == self`.
    #   (So we could make `__array_ufunc__` a static method of `Array` -- as we don't
    #   need the `self` first argument` -- but NumPy doesn't like that).
    other = inputs[1]

    if not isinstance(other, Array):
        # `other` is purely numeric (scalar or array-like). Eg: (8 mV) * 2
        other = Array(other, dimensionless)

    is_in_place = ufunc_kwargs.get("out") == self
    #      Whether this __array_ufunc__ call is an in-place operation such as
    #      `array *= 2`. [See `numpy.lib.mixins._inplace_binary_method`, which added the
    #      `out=self` kwarg].
    if is_in_place:
        ufunc_kwargs.update(out=self.data)

    #
    #
    # Exponentiation
    # --------------
    #
    # Examples:
    #   - mV ** 2
    #   - (8 mV) ** 2
    #   - ([3 8 1] mV) ** 2
    #   - (8 mV) ** [[2]]      (This becomes [[64]] mV²).
    # But not:
    #   - mV ** (8 mV)
    #   - mV ** [3, 8]         (Would yield a list with different unit per element).
    #   - mV ** (1/2)
    if (
        ufunc == np.power
        and other.data_unit == dimensionless
        and other.data.size == 1
        and issubclass(other.data.dtype.type, np.integer)
    ):
        power = other.data.item()
        new_display_unit = self.display_unit._raised_to(power)
        if isinstance(self, Unit):
            return new_display_unit
        else:
            # ufunc application & output creation:
            new_data = ufunc(self.data, other.data, **ufunc_kwargs)
            if is_in_place:
                self.display_unit = new_display_unit
                return self
            else:
                return self.__class__(new_data, new_display_unit, self.name, False)

    #
    #
    # Addition & subtraction
    # ----------------------
    #
    # Examples:
    #   - 800 mV - 23 mV
    #   - 800 mV + [3 8 2] volt
    #   - 800 mV + 1 volt
    # But not:
    #   - 800 mV + newton
    #   - 800 mV + volt
    #
    # Note that order of operands doesn't matter for unit checks.
    # (It does for the number subtraction itself ofc).
    #
    elif np.ufunc in (np.add, np.subtract):

        if isinstance(self, Unit) or isinstance(other, Unit):

            if isinstance(self, Unit):
                if ufunc == np.add:
                    preposition = "to "
                else:
                    preposition = "from "
            else:
                preposition = ""
            raise UnitError(
                f"Cannot {ufunc.__name__} {preposition}a bare unit. "
                f'Operands were "{self}" and "{other}".'
            )

            # We _could_ add or subtract bare units (if they're compatible), by
            # interpreting the `Unit` as a `Quantity` with value = 1. But that's
            # probably not what the user intended.

        if self.data_unit != other.data_unit:
            raise UnitError(
                f"Cannot {ufunc.__name__} incompatible units "
                f'"{self.display_unit}" and "{other.display_unit}".'
            )

        # todo:
        # [ufunc application & output creation]

    #
    #
    # Comparison
    # ----------
    #
    # Examples:
    #   - mV >= volt
    #   - 0.7 volt == 800 volt
    #   - [5 5] volt != [5 5] mV
    #   - 5 volt == [5 5 6] volt
    elif ufunc in comparators:

        # volt == mV
        if isinstance(self, Unit) and isinstance(other, Unit):
            if ufunc == np.equal:
                return hash(self) == hash(other)
            elif ufunc == np.not_equal:
                return not (self == other)

        # 8 mV > 9 newton
        if ufunc in ordering_comparators and self.data_unit != other.data_unit:
            raise UnitError(
                f"Ordering comparator '{ufunc.__name__}' cannot be used between "
                f'incompatible Units "{self.display_unit}" and "{other.display_unit}".'
            )

        # Note that there are no in-place versions of comparator dunders (i.e. __lt__
        # etc). They wouldn't make sense anyway: the type changes from `yunit.Array` to
        # `np.ndarray`.

        #  - [80 200] mV > 0.1 volt   (becomes `[False True]`)
        #  - mV > μV                  (.data = .scale of the first is indeed larger)
        data_comparison_result = ufunc(self.data, other.data, **ufunc_kwargs)
        unit_comparison_result = self.data_unit == other.data_unit
        return np.logical_and(data_comparison_result, unit_comparison_result)

    else:
        ...


class UnitError(Exception):
    """
    Raised when an operation with incompatible units is attempted
    (eg `8 volt + 2 newton`).
    """
