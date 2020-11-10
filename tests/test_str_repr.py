from unitlib import Quantity, define_unit


def test_unit_str_repr():
    volt = define_unit("volt")
    assert repr(volt) == '<DataUnitAtom "volt">'
    assert str(volt) == f"{volt}" == "volt"


def test_quantity_format():
    volt = define_unit("volt")
    q = Quantity(8, volt)
    assert str(q) == repr(q) == f"{q}" == "8 volt"
