from unitlib import Unit
from unitlib.prefixes import milli


def test_new():
    second = Unit.define("s")
    minute = Unit.define("min", data_unit=second, scale=60)

    assert minute.data_unit == second
    assert minute.scale == 60


def test_prefix_mul():
    volt = Unit.define("V")
    mV = milli * volt

    assert mV.name == "mV"
    assert mV.data_unit == volt
    assert mV.scale == 1e-3
