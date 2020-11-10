from unitlib import define_unit
from unitlib.prefixes import milli


def test_new():
    second = define_unit("s")
    minute = define_unit("min", 60 * second)

    assert minute.data_unit == second
    assert minute.scale == 60


def test_prefix_mul():
    volt = define_unit("V")
    mV = milli * volt

    assert mV.name == "mV"
    assert mV.data_unit == volt
    assert mV.scale == 1e-3
