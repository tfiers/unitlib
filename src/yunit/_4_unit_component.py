from ._3_unit import Unit, DataUnit


class UnitComponent(Unit):
    ...


class DataUnitComponent(DataUnit, UnitComponent):
    ...
