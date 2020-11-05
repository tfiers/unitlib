from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any

import numpy as np

from yunit.core_objects import YunitObject, Array, dimensionless
from yunit.type_aliases import NDArrayLike


@dataclass
class UfuncArgs:
    """ Arguments passed to an `Array.__array_ufunc__` call. """

    ufunc: np.ufunc
    method: str
    inputs: Tuple[Union[YunitObject, NDArrayLike], ...]
    ufunc_kwargs: Dict[str, Any]

    @property
    def is_unary(self) -> bool:
        # cos, sign, abs, negative, …
        return len(self.inputs) == 1

    @property
    def is_binary(self) -> bool:
        # +, >, **, *=, …
        return len(self.inputs) == 2

    def __post_init__(self):
        if self.is_binary:
            self.left_operand, self.right_operand = self.inputs
            self.left_array = as_array(self.left_operand)
            self.right_array = as_array(self.right_operand)


def as_array(operand: Union[NDArrayLike, YunitObject]) -> YunitObject:

    if not isinstance(operand, Array):  # `operand` is purely numeric (scalar or
        #                               #  array-like). Eg. the right operand in
        #                               #  `(8 mV) * 2`.
        operand = Array(operand, dimensionless)

    return operand
