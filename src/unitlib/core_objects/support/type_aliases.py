from typing import Union, Tuple

from unitlib.core_objects import Array, Quantity, Unit
from unitlib.type_aliases import NDArrayLike

UnitlibObject = Union[Array, Quantity, Unit]
UfuncInput = Union[UnitlibObject, NDArrayLike]
ArrayIndex = Union[int, Tuple[int, ...]]
ArraySlice = Union[Array, Quantity]
