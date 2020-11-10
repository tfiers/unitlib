"""
Multiplication
--------------

Examples:
  - mV * newton
  - 8 mV * 3
  - 8 mV * (mV ** -1)
"""

import numpy as np

from unitlib.core_objects import Unit, NonNumericDataException
from unitlib.core_objects.unit_internals import CompoundUnit, DataUnitAtom
from unitlib.prefixes import Prefix
from .support import make_binary_ufunc_output, UfuncOutput, implements, UfuncArgs


@implements([np.multiply])
def multiply(ufunc_args: UfuncArgs) -> UfuncOutput:

    try:
        inputs = ufunc_args.parse_binary_inputs()
    except NonNumericDataException as exception:
        if isinstance(ufunc_args.inputs[0], Prefix):
            # Shorthand to create a new prefixed unit. Example: `mV = milli * volt`.
            left_operand, right_operand = ufunc_args.inputs
            if isinstance(right_operand, DataUnitAtom):
                return Unit.from_prefix(left_operand, right_operand)
            else:
                raise ValueError(
                    "Can only create prefixed units from `DataUnitAtom`s. "
                    f"(Got {repr(right_operand)})."
                )
        else:
            raise exception

    new_display_unit = CompoundUnit.squeeze(
        [
            inputs.left_array.display_unit,
            inputs.right_array.display_unit,
        ]
    )

    # Unit * Unit remains pure Unit.
    if isinstance(inputs.left_array, Unit) and isinstance(inputs.right_array, Unit):
        return new_display_unit

    else:
        return make_binary_ufunc_output(ufunc_args, inputs, new_display_unit)
