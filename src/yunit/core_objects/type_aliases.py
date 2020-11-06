from typing import Union, Tuple

from . import Array, Quantity, Unit
from ..type_aliases import NDArrayLike

YunitObject = Union[Array, Quantity, Unit]
UfuncInput = Union[YunitObject, NDArrayLike]
ArrayIndex = Union[int, Tuple[int, ...]]
