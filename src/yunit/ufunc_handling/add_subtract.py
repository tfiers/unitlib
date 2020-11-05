"""
Addition & subtraction
----------------------

Examples:
  - 800 mV - 23 mV
  - 800 mV + [3 8 2] volt
  - 800 mV + 1 volt
But not:
  - 800 mV + newton
  - 800 mV + volt

Note that order of operands doesn't matter for unit checks.
(It does for numeric subtraction itself ofc).
"""

import numpy as np

from yunit.core_objects import Unit, UnitError
from .support import make_binary_ufunc_output, UfuncOutput, implements, UfuncArgs


@implements([np.add, np.subtract])
def add_subtract(args: UfuncArgs) -> UfuncOutput:

    # 800 mV + volt
    if isinstance(args.left_array, Unit) or isinstance(args.right_array, Unit):
        # Can't add (to) or subtract (from) a bare unit. We _could_ add or subtract bare
        # units (if they're compatible), by interpreting the `Unit` as a `Quantity` with
        # value = 1. But that's probably not what the user intended.
        if isinstance(args.left_array, Unit):
            if args.ufunc == np.add:  # add to a Unit
                preposition = "to "
            else:  # subtract from a Unit
                preposition = "from "
        else:  # add/subtract a Unit
            preposition = ""
        raise UnitError(
            f"Cannot {args.ufunc.__name__} {preposition}a bare unit. "
            f'Operands were "{args.left_array}" and "{args.right_array}".'
        )

    # 800 mV + 1 newton
    if args.left_array.data_unit != args.right_array.data_unit:
        raise UnitError(
            f"Cannot {args.ufunc.__name__} incompatible units "
            f'"{args.left_array.display_unit}" and "{args.right_array.display_unit}".'
        )

    # For e.g. "800 mV + 1 volt", choose the largest unit as new unit (i.e. volt).
    if args.left_array.display_unit > args.right_array.display_unit:
        new_display_unit = args.left_array.display_unit
    else:
        new_display_unit = args.right_array.display_unit

    return make_binary_ufunc_output(args, new_display_unit)
