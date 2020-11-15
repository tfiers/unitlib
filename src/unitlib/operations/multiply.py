"""
Multiplication
--------------

Examples:
  - mV * newton
  - 8 mV * 3
  - 8 mV * (mV ** -1)
"""
from collections import defaultdict
from typing import Dict

import numpy as np

from unitlib.core_objects import Unit, NonNumericDataException, dimensionless
from unitlib.core_objects.support.type_aliases import UfuncInput
from unitlib.core_objects.unit import DataUnit
from unitlib.core_objects.unit_internals import CompoundUnit, UnitAtom, CompoundDataUnit
from unitlib.prefixes import Prefix
from .support import make_binary_ufunc_output, UfuncOutput, implements, UfuncArgs


@implements([np.multiply])
def multiply(args: UfuncArgs) -> UfuncOutput:

    try:
        inputs = args.parse_binary_inputs()
    except NonNumericDataException as exception:
        left_operand, right_operand = args.inputs
        if isinstance(left_operand, Prefix):
            return create_prefixed_unit(left_operand, right_operand)
        else:
            raise exception

    new_display_unit = combine(
        inputs.left_array.display_unit,
        inputs.right_array.display_unit,
    )

    # Unit * Unit remains pure Unit.
    if isinstance(inputs.left_array, Unit) and isinstance(inputs.right_array, Unit):
        return new_display_unit

    else:
        return make_binary_ufunc_output(args, inputs, new_display_unit)


def combine(left_unit: CompoundUnit, right_unit: CompoundUnit) -> Unit:

    components = left_unit.components + right_unit.components

    # Aggregate all `PoweredUnitAtom`s that have the same `unit_atom`, and sum their
    # powers. Throw away dimensionless components and components with a combined
    # power of zero.
    combined_powers: Dict[UnitAtom, int] = defaultdict(lambda: 0)
    for unit in components:
        combined_powers[unit.unit_atom] += unit.power

    squeezed_components = tuple(
        unit_atom._raised_to(combined_power)
        for unit_atom, combined_power in combined_powers.items()
        if unit_atom != dimensionless and combined_power != 0
    )

    if len(squeezed_components) == 0:
        return dimensionless

    elif len(squeezed_components) == 1:
        return squeezed_components[0]

    else:
        if all(isinstance(component, DataUnit) for component in squeezed_components):
            cls = CompoundDataUnit
        else:
            cls = CompoundUnit
        return cls(squeezed_components)


def create_prefixed_unit(prefix: Prefix, right_operand: UfuncInput):
    # Syntax to create a new prefixed unit.
    # Examples: `mV = milli * volt`, `μg = micro * gram`.
    if isinstance(right_operand, UnitAtom):
        unit_atom = right_operand
        return UnitAtom(
            name=f"{prefix.symbol}{unit_atom.name}",
            data_unit=unit_atom.data_unit,
            scale=prefix.factor * unit_atom.scale,
        )
    else:
        raise ValueError(
            "Can only create prefixed units from `UnitAtom`s. "
            f"(Got {repr(right_operand)})."
        )
