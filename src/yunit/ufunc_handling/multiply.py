"""
Multiplication
--------------

Examples:
  - mV * newton
  - 8 mV * 3
  - 8 mV * (mV ** -1)
"""

import numpy as np

from yunit.core_objects import Unit
from .support import make_binary_ufunc_output, UfuncOutput, implements, UfuncArgs
from ..core_objects.unit_internals import CompoundUnit


@implements([np.multiply])
def multiply(args: UfuncArgs) -> UfuncOutput:

    new_display_unit = CompoundUnit.squeeze(
        [
            args.left_array.display_unit,
            args.right_array.display_unit,
        ]
    )

    # Unit * Unit remains pure Unit.
    if isinstance(args.left_array, Unit) and isinstance(args.right_array, Unit):
        return new_display_unit

    else:
        return make_binary_ufunc_output(args, new_display_unit)
