from .array import Array, Quantity
from .unit import Unit

# We do not require a matplotlib installation.
try:
    import matplotlib
except ModuleNotFoundError:
    pass
else:
    from . import matplotlib_support
