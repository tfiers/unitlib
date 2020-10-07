from collections import defaultdict
from itertools import chain
from typing import Iterable, Dict, Tuple

from ._4_unit import DataUnit, Unit
from ..backwards_compatibility import prod, TYPE_CHECKING

if TYPE_CHECKING:
    from ._2_powered_unit_atom import PoweredUnitAtom


class CompoundUnit(Unit):
    """
    A multiplicative combination of `PoweredUnitAtom`s.
    Eg "mV²·μm·nS⁻¹".

    Characteristic attribute:
        - components: Tuple[PoweredUnitAtom, ...]
    """

    def __init__(self, units: Iterable["CompoundUnit"]):
        flattened_components = chain(unit.components for unit in units)
        self.components = self._squeeze(flattened_components)

    @staticmethod
    def _squeeze(
        components: Iterable["PoweredUnitAtom"],
    ) -> Tuple["PoweredUnitAtom", ...]:
        """
        Aggregate all `PoweredUnitAtom`s that have the same `unit_atom`, and sum their
        powers. Throw away dimensionless components and components with a combined power
        of zero.
        """
        from ._1_unit_atom import UnitAtom, dimensionless

        combined_powers: Dict[UnitAtom, int] = defaultdict(lambda: 0)
        for component in components:
            combined_powers[component.unit_atom] += component.power
        squeezed_components = tuple(
            atom ** power
            for atom, power in combined_powers.items()
            if atom != dimensionless and power != 0
        )
        if len(squeezed_components) == 0:
            return (dimensionless,)
        else:
            return squeezed_components

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
