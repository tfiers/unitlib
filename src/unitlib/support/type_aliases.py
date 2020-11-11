from numbers import Number
from typing import Sequence, Union

import numpy as np


scalar_types = (Number, np.number)
Scalar = Union[scalar_types]
NDArrayLike = Union[Sequence, np.ndarray, Scalar]
# To be correct, `NDArrayLike` should include a `SupportsArray` Prototype class, as is
# done in the `NDArrayLike` object of NumPy 1.20 does [1]. We could recreate it here
# (using `typing.Protocol`), but PyCharm resolves that to `Any`, cluttering up
# downstream auto-completes of unitlib objects.
#
# [1]: https://github.com/numpy/numpy/blob/master/numpy/typing/_array_like.py
