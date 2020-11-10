from typing import Dict
from weakref import WeakKeyDictionary

import numpy as np
from matplotlib.axis import Axis
from matplotlib.units import AxisInfo, ConversionInterface, registry

import unitlib


# The matplotlib module of interest is called `units`; but really it is used to allow
# arbitrary classes to be plotted -- nothing unit specific.


# Trick learned from `unyt`.
array_names: Dict[Axis, str] = WeakKeyDictionary()


class ArrayPlotInterface(ConversionInterface):
    @staticmethod
    def convert(array: unitlib.Array, unit: unitlib.Unit, axis: Axis) -> np.ndarray:
        """
        `unit` argument is specified by the user via the (not well-documented) options
        `xaxis.set_units()` or `plot(..., xunits=...)`. When these are not set, this
        class's `default_units` is used.
        """
        return array.data_in_display_units

    @staticmethod
    def default_units(array: unitlib.Array, axis: Axis) -> unitlib.Unit:
        if array.name is None:
            name_to_display = ""
        else:
            name_to_display = array.name
        array_names[axis] = name_to_display
        return array.display_unit

    @staticmethod
    def axisinfo(unit: unitlib.Unit, axis: Axis) -> AxisInfo:
        return AxisInfo(label=f"{array_names[axis]} ({unit.name})")


registry[unitlib.Array] = ArrayPlotInterface()
