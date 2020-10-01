from ._1_array import Array
from ._2_quantity import Quantity
from ._3_unit import Unit


try:
    import matplotlib
except ModuleNotFoundError:
    pass  # A matplotlib installation is not required.
else:
    # But if it is installed, allow it to plot yunit Arrays.
    from . import matplotlib_support
