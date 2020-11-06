"""
Comparison
----------

Examples:
  - mV >= volt
  - 0.7 volt == 800 volt
  - [5 5] volt != [5 5] mV
  - 5 volt == [5 5 6] volt
"""

import numpy as np

from yunit.core_objects import Unit, IncompatibleUnitsError
from .support import implements, UfuncOutput, UfuncArgs

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
def compare(args: UfuncArgs) -> UfuncOutput:

    # volt == mV
    if isinstance(args.left_array, Unit) and isinstance(args.right_array, Unit):
        if args.ufunc == np.equal:
            return hash(args.left_array) == hash(args.right_array)
        elif args.ufunc == np.not_equal:
            return not (args.left_array == args.right_array)

    # 8 mV > 9 newton
    elif (
        args.ufunc in ordering_comparators
        and args.left_array.data_unit != args.right_array.data_unit
    ):
        raise IncompatibleUnitsError(
            f"Ordering comparator '{args.ufunc.__name__}' cannot be used between "
            f'incompatible Units "{args.left_array.display_unit}" '
            f'and "{args.right_array.display_unit}".'
        )

    #  - [80 200] mV > 0.1 volt       -> [False True]
    #  - mV > μV                      -> True  (`.data` = `.scale` of mV is larger)
    #  - [8 3] newton == [8 3] volt   -> [False, False]
    else:
        data_comparison_result = args.ufunc(
            args.left_array.data,
            args.right_array.data,
            **args.ufunc_kwargs,
        )
        unit_comparison_result = args.left_array.data_unit == args.right_array.data_unit
        return np.logical_and(data_comparison_result, unit_comparison_result)
        # Note that there are no in-place versions of comparator dunders (i.e. __lt__
        # etc). They wouldn't make sense anyway: the type changes from `yunit.Array` to
        # `np.ndarray`.
