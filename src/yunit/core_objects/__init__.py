from typing import Union

from .array import Array
from .quantity import Quantity
from .unit import Unit, UnitError
from .unit_internals import dimensionless

YunitObject = Union[Array, Quantity, Unit]
