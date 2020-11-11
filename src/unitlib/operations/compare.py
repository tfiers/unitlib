"""
Comparison
----------

Examples:
  - mV >= volt
  - 0.7 volt == 800 volt
  - [5 5] volt != [5 5] mV
  - 5 volt == [5 5 6] volt
"""
from copy import copy
from typing import Union

import numpy as np

from unitlib.core_objects import Unit, IncompatibleUnitsError, NonNumericDataException
from .support import implements, UfuncArgs

equality_comparators = (
    np.equal,
    np.not_equal,
)

ordering_comparators = (
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
)


@implements(equality_comparators + ordering_comparators)
def compare(args: UfuncArgs) -> Union[np.ndarray, bool]:

    if args.ufunc == np.not_equal:
        new_args = copy(args)
        new_args.ufunc = np.equal
        is_equal = compare(new_args)
        if isinstance(is_equal, bool):
            return not is_equal
        else:
            return np.logical_not(is_equal)

    try:
        inputs = args.parse_binary_inputs()
    except NonNumericDataException as exception:
        # One of the operands is e.g. `None`, as in `8 mV == None`.
        if args.ufunc == np.equal:
            return False
        else:
            raise exception

    # volt == mV
    if (
        isinstance(inputs.left_array, Unit)
        and isinstance(inputs.right_array, Unit)
        and args.ufunc == np.equal
    ):
        return hash(inputs.left_array) == hash(inputs.right_array)

    # 8 mV > 9 newton
    elif (
        args.ufunc in ordering_comparators
        and inputs.left_array.data_unit != inputs.right_array.data_unit
    ):
        raise IncompatibleUnitsError(
            f"Ordering comparator '{args.ufunc.__name__}' cannot be used between "
            f'incompatible Units "{inputs.left_array.display_unit}" '
            f'and "{inputs.right_array.display_unit}".'
        )

    #  - [80 200] mV > 0.1 volt       -> [False True]
    #  - mV > μV                      -> True  (`.data` = `.scale` of mV is larger)
    #  - [8 3] newton == [8 3] volt   -> [False, False]
    #  - [8 3] newton != [8 3] volt   -> [True, True]
    #  - [8 3] newton != [8 2] newton -> [False, True]
    #  - [8 3] newton == [8 2] newton -> [True, False]
    else:
        data_comparison_result = args.ufunc(
            inputs.left_array.data,
            inputs.right_array.data,
            **args.ufunc_kwargs,
        )
        # Note that there are no in-place versions of comparason dunders. They wouldn't
        # make sense anyway: the type changes from `unitlib.Array` to `np.ndarray`.

        if args.ufunc in ordering_comparators:
            return data_comparison_result
        else:
            unit_comparison_result = (
                inputs.left_array.data_unit == inputs.right_array.data_unit
            )
            return np.logical_and(data_comparison_result, unit_comparison_result)
