from collections import defaultdict
from itertools import chain
from typing import Iterable, Dict, Tuple

import numpy as np

from unitlib.core_objects.unit import DataUnit, Unit
from unitlib.support.backwards_compatibility import prod, TYPE_CHECKING

if TYPE_CHECKING:
    from .powered_unit_atom import PoweredUnitAtom, PoweredDataUnitAtom


class CompoundUnit(Unit):
    """
    A multiplicative combination of `PoweredUnitAtom`s.
    Eg "mV²·μm·nS⁻¹".

    Defining attribute:
        components: Tuple[PoweredUnitAtom, ...]
    """

    def __init__(self, components: Tuple["PoweredUnitAtom", ...]):
        self.components = components

    @property
    def name(self):
        return "·".join([component.name for component in self.components])

    @property
    def data_unit(self) -> "CompoundDataUnit":
        # `np.prod` will automatically call our custom `np.multiply` implementation
        # repeatedly (no need for us to implement `method="reduce"`).
        return np.prod([component.data_unit for component in self.components])

    @property
    def scale(self) -> float:
        return prod([component.scale for component in self.components])

    def __hash__(self):
        return hash(self.components)

    def _raised_to(self, power: int):
        return np.prod([component ** power for component in self.components])


class CompoundDataUnit(DataUnit, CompoundUnit):
    def __init__(self, units: Tuple["PoweredDataUnitAtom", ...]):
        CompoundUnit.__init__(self, units)
