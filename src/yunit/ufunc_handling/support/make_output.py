from typing import Optional

import numpy as np

from yunit.core_objects import Unit
from .registry import UfuncOutput
from .ufunc_args import UfuncArgs


def make_ufunc_output(
    new_display_unit: Unit,
    numpy_ufunc_output: np.ndarray,
) -> UfuncOutput:
    """ Select/create output object of correct type """

    from yunit.core_objects import dimensionless, Quantity, Array

    if new_display_unit == dimensionless:
        return numpy_ufunc_output

    elif numpy_ufunc_output.size == 1:
        return Quantity(
            numpy_ufunc_output,
            new_display_unit,
            name=None,
            value_is_given_in_display_units=False,
        )

    else:
        return Array(
            numpy_ufunc_output,
            new_display_unit,
            name=None,
            data_are_given_in_display_units=False,
        )


def make_binary_ufunc_output(
    args: UfuncArgs,
    new_display_unit: Unit,
    left_numpy_ufunc_arg: Optional[np.ndarray] = None,
    right_numpy_ufunc_arg: Optional[np.ndarray] = None,
) -> UfuncOutput:

    is_in_place = args.ufunc_kwargs.get("out") is (args.left_operand,)
    #      Whether this __array_ufunc__ call is an in-place operation such as
    #      `array *= 2`. See `numpy.lib.mixins._inplace_binary_method`, which added
    #      the "out" kwarg (as `out=(self,)`).

    if is_in_place:
        args.ufunc_kwargs.update(out=args.left_operand.data)

    if left_numpy_ufunc_arg is None:
        left_numpy_ufunc_arg = args.left_array.data
    if right_numpy_ufunc_arg is None:
        right_numpy_ufunc_arg = args.right_array.data

    numpy_ufunc_output = args.ufunc(
        left_numpy_ufunc_arg, right_numpy_ufunc_arg, **args.ufunc_kwargs
    )

    if is_in_place:
        args.left_operand.display_unit = new_display_unit
        return args.left_operand
    else:
        return make_ufunc_output(new_display_unit, numpy_ufunc_output)
