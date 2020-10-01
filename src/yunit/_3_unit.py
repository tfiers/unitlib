from typing import Tuple

from ._2_quantity import Quantity
from ._4_unit_component import UnitComponent


class Unit(Quantity):

    components: Tuple[UnitComponent]

    @property
    def scale(self) -> float:
        return self.data.item()

    @property
    def data_unit(self) -> 'DataUnit':
        return DataUnit([])


class DataUnit(Unit):
    ...
