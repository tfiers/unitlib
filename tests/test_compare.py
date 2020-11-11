from unitlib.units import volt
from unitlib.prefixes import milli

mV = milli * volt

x = [1, 2] * mV

def test_ne_ndarray():
    assert all((x != [1, 2]) == [True, True])
