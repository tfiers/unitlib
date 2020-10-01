from ._4_unit_component import UnitComponent, DataUnitComponent


class UnitAtom(UnitComponent):
    ...


class DataUnitAtom(DataUnitComponent, UnitAtom):
    ...
