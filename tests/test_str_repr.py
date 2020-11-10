from unitlib import Quantity, Unit


def test_unit_str_repr():
    volt = Unit('volt')
    assert repr(volt) == '<DataUnitAtom "volt">'
    assert str(volt) == f"{volt}" == "volt"

def test_quantity_format():
    volt = Unit('volt')
    q =  Quantity(8, volt)
    assert str(q) == repr(q) == f"{q}" == "8 volt"
