from scipy.signal import find_peaks
from unitlib import add_unit_support
from unitlib.units import km, meter


def test_add_unit_support():
    height_array = [4, 6, 3, 2, 4, 2] * km
    find_peaks_ = add_unit_support(find_peaks)
    assert find_peaks_.__doc__ == find_peaks.__doc__
    peaks, _ = find_peaks_(height_array, height=5000 * meter)
    assert peaks == [1]
