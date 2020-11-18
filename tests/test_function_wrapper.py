from scipy.signal import find_peaks as find_peaks_orig
from unitlib import add_unit_support
from unitlib.units import km, meter


def test_add_unit_support():
    height_array = [4, 6, 3, 2, 4, 2] * km
    find_peaks = add_unit_support(find_peaks_orig)
    assert find_peaks.__doc__ == find_peaks_orig.__doc__
    peak_indices, _ = find_peaks(
        height_array,
        height=5000 * meter,  # minimum height
    )
    assert peak_indices == [1]
