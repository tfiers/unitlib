from ._3_unit import Unit
from ._5_unit_atom import UnitAtom
from ._4_unit_component import UnitComponent




class DataUnitComponent(DataUnit, UnitComponent):
    ...


class DataUnitAtom(DataUnitComponent, UnitAtom):
    ...
