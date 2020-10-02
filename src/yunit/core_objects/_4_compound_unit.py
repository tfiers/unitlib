from collections import defaultdict
from itertools import chain
from typing import Iterable, Dict

from ._3_unit import DataUnit, Unit
from ._5_powered_unit_atom import PoweredUnitAtom
from ._6_unit_atom import UnitAtom, dimensionless
from yunit.backwards_compatibility import prod


class CompoundUnit(Unit):
    """
    A multiplicative combination of `PoweredUnitAtom`s.
    Eg "mV²·μm·nS⁻¹".

    Characteristic attribute:
        - components: Tuple[PoweredUnitAtom, ...]
    """

    def __init__(self, units: Iterable["CompoundUnit"]):
        flattened_components = chain(unit.components for unit in units)
        squeezed_components = self._squeeze_components(flattened_components)
        self.components = tuple(squeezed_components)

    @staticmethod
    def _squeeze_components(
        components: Iterable[PoweredUnitAtom],
    ) -> Iterable[PoweredUnitAtom]:
        """
        Aggregate all `PoweredUnitAtom`s that have the same `unit_atom`, and sum their
        powers. Throw away dimensionless components and components with a combined power
        of zero.
        """
        combined_powers: Dict[UnitAtom, int] = defaultdict(lambda: 0)
        for component in components:
            combined_powers[component.unit_atom] += component.power
        return (
            atom ** power
            for atom, power in combined_powers.items()
            if atom != dimensionless and power != 0
        )

    @property
    def name(self):
        return "·".join([component.name for component in self.components])

    @property
    def data_unit(self) -> "CompoundDataUnit":
        return CompoundDataUnit([component.data_unit for component in self.components])

    @property
    def scale(self) -> float:
        return prod([component.scale for component in self.components])

    def __hash__(self):
        return hash(self.components)

    def _raised_to(self, power: int):
        return self.__class__([component ** power for component in self.components])
        #   `__class__` can be `CompoundUnit` or `CompoundDataUnit`.


class CompoundDataUnit(DataUnit, CompoundUnit):
    def __init__(self, components: Iterable["CompoundDataUnit"]):
        CompoundUnit.__init__(self, components)
