from ._1_array import Array


class Quantity(Array):
    """
    A scalar number with a physical unit, such as `8 * mV`.

    In addition to `Array`'s functionality, it has a `value` property, which is the data
    in display units as a Python scalar (and not a 0-dimensional NumPy array).
    """
    @property
    def value(self):
        return self.data_in_display_units.item()