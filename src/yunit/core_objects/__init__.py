from typing import Union

from ._1_array import Array
from ._2_quantity import Quantity
from ._3_unit import Unit, UnitError
from .unit_internals import dimensionless

YunitObject = Union[Array, Quantity, Unit]
