from typing import Optional, Union, Tuple

import numpy as np

from unitlib.core_objects import Unit, dimensionless
from unitlib.core_objects.support.util import create_Array_or_Quantity
from .registry import UfuncOutput
from .ufunc_args import UfuncArgs, BinaryUfuncInputs


NumPyObject = Union[np.ndarray, np.number]


def make_ufunc_output(
    new_display_unit: Unit,
    numpy_ufunc_output: NumPyObject,
) -> UfuncOutput:
    """ Select/create output object of correct type """
    if new_display_unit == dimensionless:
        return numpy_ufunc_output
    else:
        return create_Array_or_Quantity(numpy_ufunc_output, new_display_unit)


def make_binary_ufunc_output(
    args: UfuncArgs,
    inputs: BinaryUfuncInputs,
    new_display_unit: Unit,
    custom_numpy_ufunc_inputs: Optional[Tuple[NumPyObject, NumPyObject]] = None,
) -> UfuncOutput:

    is_in_place = args.ufunc_kwargs.get("out") is (inputs.left_operand,)
    #      Whether this __array_ufunc__ call is an in-place operation such as
    #      `array *= 2`. See `numpy.lib.mixins._inplace_binary_method`, which added
    #      the "out" kwarg (as `out=(self,)`).

    if is_in_place:
        args.ufunc_kwargs.update(out=inputs.left_operand.data)

    if custom_numpy_ufunc_inputs:
        numpy_ufunc_inputs = custom_numpy_ufunc_inputs
    else:
        numpy_ufunc_inputs = (inputs.left_array.data, inputs.right_array.data)

    numpy_ufunc_output = args.ufunc(*numpy_ufunc_inputs, **args.ufunc_kwargs)

    if is_in_place:
        inputs.left_operand.display_unit = new_display_unit
        return inputs.left_operand
    else:
        return make_ufunc_output(new_display_unit, numpy_ufunc_output)
