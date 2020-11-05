"""
Exponentiation
--------------

Examples:
  - mV ** 2
  - (8 mV) ** 2
  - ([3 8 1] mV) ** 2
  - (8 mV) ** [[2]]      (This becomes [[64]] mV²).
But not:
  - mV ** (8 mV)
  - mV ** [3, 8]         (Would yield a list with different unit per element).
  - mV ** (1/2)
"""

import numpy as np

from yunit.core_objects import Unit, dimensionless
from .support import make_binary_ufunc_output, implements, UfuncOutput, UfuncArgs


@implements([np.power])
def power(args: UfuncArgs) -> UfuncOutput:

    if args.right_array.data_unit is not dimensionless:
        raise ValueError(
            "Cannot raise to powers with a unit "
            f'(power "{args.right_operand}" is not dimensionless).'
        )

    if args.right_array.data.size != 1:
        raise ValueError(
            f"Can only raise to scalar powers "
            f'(power "{args.right_operand}" does not have size 1).'
        )
        # An sequence with different units per element is not supported by yunit.

    if not issubclass(args.right_array.data.dtype.type, np.integer):
        raise NotImplementedError(
            f"Can not yet raise to non-integer powers "
            f'(power "{args.right_operand}" '
            f'has non-integer dtype "{args.right_array.data.dtype}").'
        )

    power = args.right_array.data.item()
    new_display_unit = args.left_array.display_unit._raised_to(power)

    # Unit to a power remains a pure Unit (`mV ** 2`)
    if isinstance(args.left_array, Unit):
        return new_display_unit

    else:
        if power < 0 and issubclass(args.left_array.data.dtype.type, np.integer):
            # NumPy doesn't allow exponentiating integers to negative powers.
            # (Because it changes the data type to no longer be integer,
            # presumably). We -- just like plain Python -- do allow it however.
            left_numpy_ufunc_arg = args.left_array.data.astype(float)
        else:
            left_numpy_ufunc_arg = args.left_array.data

        return make_binary_ufunc_output(args, new_display_unit, left_numpy_ufunc_arg)
