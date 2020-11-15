from unitlib.units import kilogram, gram, μg, micro


def test_unit_defs():
    assert 1 * kilogram == 1000 * gram
    assert μg.data_unit == kilogram
    assert μg.scale == micro.factor * gram.scale
    assert gram.scale == 0.001
