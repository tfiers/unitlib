import sys
from typing import Iterable

import numpy as np


if sys.version_info >= (3, 8):
    from typing import Protocol, Literal, TYPE_CHECKING
    from math import prod

else:  # Support Python 3.6 and 3.7

    # Backports of new objects from `typing`.
    from typing_extensions import Protocol, Literal, TYPE_CHECKING

    def prod(numbers: Iterable[float]) -> float:
        return np.prod(numbers).item()


# Trick to make PyCharm IDE:
#  - Not complain about unused imports
#  - In other modules, suggest importing these objects from this module instead of from
#    their original packages.
Protocol, Literal, TYPE_CHECKING, prod = Protocol, Literal, TYPE_CHECKING, prod
