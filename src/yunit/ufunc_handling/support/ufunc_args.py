from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

from yunit.core_objects.type_aliases import UfuncInput
from yunit.core_objects.util import as_array


@dataclass
class UfuncArgs:
    """ Arguments passed to an `Array.__array_ufunc__` call. """

    ufunc: np.ufunc
    method: str
    inputs: Tuple[UfuncInput, ...]
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
