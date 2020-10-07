"""
Implementation of `Array.__array_ufunc__`.

This handles all Python math syntax (*, *=, **, <, %, …)
as well as all NumPy elementwise functions (`cos`, `sign`, `abs`, …),
for both Arrays, Quantities, and Units.
"""

from typing import Union, Tuple

import numpy as np

from . import Array, Unit, dimensionless, Quantity
from ..type_aliases import NDArrayLike

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
    np.divide,  # == np.true_divide
    np.power,
)  # There's more in `NDArrayOperatorsMixin`

supported_ufuncs = comparators + numeric_methods


def __array_ufunc__(
    self: Union[Unit, Quantity, Array],
    ufunc: np.ufunc,
    method: str,
    *inputs: Tuple[Union[Unit, Quantity, Array, NDArrayLike], ...],
    **ufunc_kwargs,
) -> Union[Unit, Quantity, Array, np.ndarray, bool]:

    # Docs for __array_ufunc__:
    # https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__

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
    # - For 'normal' operations (`8 mV * 3`, __mul__) and in-place operations
    #   (`8 mV *= 3`, __imul__), `inputs[0] == self`
    # - For reflected operations (`3 * 8 mV`, __rmul__), `inputs[1] == self`.

    left_operand, right_operand = inputs

    def as_array(operand: Union[Array, NDArrayLike]) -> Array:
        if not isinstance(operand, Array):
            # `operand` is purely numeric (scalar or array-like). Eg. the right operand
            # in `(8 mV) * 2`.
            operand = Array(operand, dimensionless)
        return operand

    left_array, right_array = map(as_array, inputs)

    is_in_place = ufunc_kwargs.get("out") == self
    #      Whether this __array_ufunc__ call is an in-place operation such as
    #      `array *= 2`. See `numpy.lib.mixins._inplace_binary_method`, which added the
    #      `out=self` kwarg.
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
        and right_array.data_unit == dimensionless
        and right_array.data.size == 1
        and issubclass(right_array.data.dtype.type, np.integer)
    ):
        power = right_array.data.item()
        new_display_unit = left_array.display_unit._raised_to(power)
        if isinstance(left_array, Unit):
            return new_display_unit
        else:
            # ufunc application & output creation:
            new_data = ufunc(left_array.data, right_array.data, **ufunc_kwargs)
            if is_in_place:
                self.display_unit = new_display_unit
                return self
            else:
                return left_array.__class__(
                    new_data, new_display_unit, left_array.name, False
                )

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

        if isinstance(left_array, Unit) or isinstance(right_array, Unit):

            if isinstance(self, Unit):
                if ufunc == np.add:
                    preposition = "to "
                else:
                    preposition = "from "
            else:
                preposition = ""
            raise UnitError(
                f"Cannot {ufunc.__name__} {preposition}a bare unit. "
                f'Operands were "{left_array}" and "{right_array}".'
            )

            # We _could_ add or subtract bare units (if they're compatible), by
            # interpreting the `Unit` as a `Quantity` with value = 1. But that's
            # probably not what the user intended.

        if left_array.data_unit != right_array.data_unit:
            raise UnitError(
                f"Cannot {ufunc.__name__} incompatible units "
                f'"{left_array.display_unit}" and "{right_array.display_unit}".'
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
        if isinstance(left_array, Unit) and isinstance(right_array, Unit):
            if ufunc == np.equal:
                return hash(left_array) == hash(right_array)
            elif ufunc == np.not_equal:
                return not (left_array == right_array)

        # 8 mV > 9 newton
        if (
            ufunc in ordering_comparators
            and left_array.data_unit != right_array.data_unit
        ):
            raise UnitError(
                f"Ordering comparator '{ufunc.__name__}' cannot be used between "
                f'incompatible Units "{left_array.display_unit}" '
                f'and "{right_array.display_unit}".'
            )

        # Note that there are no in-place versions of comparator dunders (i.e. __lt__
        # etc). They wouldn't make sense anyway: the type changes from `yunit.Array` to
        # `np.ndarray`.

        #  - [80 200] mV > 0.1 volt   (becomes `[False True]`)
        #  - mV > μV                  (.data = .scale of the first is indeed larger)
        data_comparison_result = ufunc(
            left_array.data, right_array.data, **ufunc_kwargs
        )
        unit_comparison_result = left_array.data_unit == right_array.data_unit
        return np.logical_and(data_comparison_result, unit_comparison_result)

    #
    #
    # Division
    # --------
    #
    # Examples:
    #   - 8 mV / 3 mV
    #   - 8 mV / mV
    #   - mV / second
    #   - [8 3] mV / 6 second
    #   - 8 / second
    elif ufunc == np.divide:

        # Allow `1 / mV` as a special case to create a pure Unit (`mV⁻¹`), and not a
        # Quantity with value = 1.
        if (
            isinstance(left_operand, int)
            and left_operand == 1
            and isinstance(right_array, Unit)
        ):
            return right_array ** -1

        else:
            return left_array * (right_array ** -1)

    else:
        ...


class UnitError(Exception):
    """
    Raised when an operation with incompatible units is attempted
    (eg `8 volt + 2 newton`).
    """
