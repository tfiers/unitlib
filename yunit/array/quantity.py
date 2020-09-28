import numpy as np

from .array import Array
from ..type_aliases import Scalar
from ..unit import Unit


class Quantity(Array):
    """
    A scalar number with a physical unit, such as `8 * mV`.

    See `Array`.
    """

    def __init__(
        self,
        value: Scalar,
        display_unit: Unit,
        value_is_given_in_display_units: bool = True,
    ):
        """
        :param value:  A Python or NumPy scalar number.
        :param display_unit:  Units in which to display the value.
        :param value_is_given_in_display_units:  If True (default), the given `value` is
                    taken to be expressed in `display_unit`s, and is converted to and
                    stored internally in `display_unit.data_unit`s. If False, `value`
                    is taken to be already expressed in `display_unit.data_unit`s,
                    and no conversion is done.
        """
        if np.size(value) > 1:
            raise ValueError(
                f"`value` must not have more than one element. Got `{value}`."
            )
        # We implicitly allow not only scalars, but also any size-1 array, like
        # `array(3)`, `[3]`, `[[3]]`, etc.
        super().__init__(value, display_unit, value_is_given_in_display_units)

    @property
    def value(self):
        return self.data_in_display_units.item()

    @property
    def value_unit(self):
        return self.display_unit

    _DIY_help_text = (
        "You can get the bare numeric value via `quantity.value` "
        "and work with it manually."
    )
