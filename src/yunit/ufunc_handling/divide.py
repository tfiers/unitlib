"""
Division
--------

Examples:
  - 8 mV / 3 mV
  - 8 mV / mV
  - mV / second
  - [8 3] mV / 6 second
  - 8 / second
"""

import numpy as np

from yunit.core_objects import Unit
from .support import UfuncOutput, implements, UfuncArgs


@implements([np.divide])  # np.divide == np.true_divide
def divide(args: UfuncArgs) -> UfuncOutput:

    # Allow `1 / mV` as a special case to create a pure Unit (`mV⁻¹`), and not a
    # Quantity with value = 1.
    if (
        isinstance(args.left_operand, int)
        and args.left_operand == 1
        and isinstance(args.right_array, Unit)
    ):
        return args.right_array ** -1

    else:
        return args.left_array * (args.right_array ** -1)
