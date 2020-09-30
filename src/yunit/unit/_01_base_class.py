from abc import ABC, abstractmethod, abstractproperty
from numbers import Number
from typing import Optional, overload

from ..prefixes import Prefix
from ..type_aliases import ArrayLike, Scalar, scalar_types
from ..backwards_compatibility import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    # See the explanation of `if TYPE_CHECKING` in ../ReadMe.
    from ..array import Array, Quantity
    from ._02_simple_unit import SimpleUnit, SimpleDataUnit
    from ._04_compound_unit import CompoundUnit


class Unit(ABC):
    """
    A physical unit. For example, "farad", "μm²", or "mV/nS".

    Units can be:
     - raised to a power (`meter**2`);
     - composed with other units (`newton * meter`);
     - applied to numeric data (`8*farad`, `[3,5]*mV`);
        (even if that data already has units: `5*mV / nS`, `5*mV / mV`).

    This abstract base class ('ABC') defines an interface that all {..}Unit classes in
    this package implement.
    """

    #
    #
    # ---------------
    # Core properties

    @abstractproperty
    def name(self) -> str:
        """
        How this unit is displayed textually (eg. in print statements or in axis labels
        of data plots). Examples: "min", "mV".
        """
        ...  # For subclasses to implement.

    @abstractproperty
    def data_scale(self) -> Number:
        """
        Factor with which numeric data annotated with this unit is multiplied before
        being stored in memory.

        For example, if this `unit` is "mV" (with a `data_unit` of "volt") and its
        `data_scale` is 1E-3, the numeric data underlying the expression `8 * mV` will
        be stored as `0.008` in memory.
        """
        ...  # For subclasses to implement.

    @abstractproperty
    def data_unit(self) -> "DataUnit":
        """
        A scalar multiple or submultiple of this `unit`.

        Numeric data annotated with this `unit` is stored in `data_unit`s in memory.
        See `data_scale`.

        One `unit` equals `data_scale` of its `data_unit`s.
        E.g. 1 minute = 60 seconds.
        """
        ...  # For subclasses to implement.

    #
    #
    # --------------------------------
    # Behave as a proper Python object

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<{self.__class__.__name__} "{self.name}">'

    def __eq__(self, other):
        return hash(self) == hash(other)

    @abstractmethod
    def __hash__(self) -> int:
        ...  # For subclasses to implement.

    #
    #
    # -----------------
    # Unit creation API

    def __new__(
        cls,
        name: str,
        data_unit: Optional["SimpleDataUnit"] = None,
        data_scale: Optional[float] = 1,
    ):
        # Use `Unit`'s constructor as a shorthand to create new
        # `SimpleUnit`s and `SimpleDataUnit`s.

        from ._02_simple_unit import SimpleUnit, SimpleDataUnit

        if data_unit is None:
            return SimpleDataUnit(name)
        else:
            return SimpleUnit(name, data_unit, data_scale)

    @staticmethod
    def from_prefix(prefix: Prefix, data_unit: "SimpleDataUnit") -> "SimpleUnit":
        from ._02_simple_unit import SimpleUnit

        return SimpleUnit(
            name=f"{prefix.symbol}{data_unit.name}",
            data_unit=data_unit,
            data_scale=prefix.factor,
        )

    #
    #
    # ----------------
    # Unit composition

    # mV * farad
    def __mul__(self, other) -> "CompoundUnit":
        from ._04_compound_unit import CompoundUnit

        if isinstance(other, Unit):
            return CompoundUnit([self, other])
        else:
            return NotImplemented  # mV * 3

    # mV / farad
    def __truediv__(self, other):
        if isinstance(other, Unit):
            return self * (1 / other)
            # `1 / other` will defer to `Unit.__rtruediv__`.
        else:
            return NotImplemented  # mV / 3

    # mV ** 2
    def __pow__(self, power, modulo=None) -> "Unit":
        if modulo is not None:
            return NotImplemented
        elif isinstance(power, int):
            return self._raised_to(power)
        else:
            raise NotImplementedError("Fractional unit powers are not yet supported.")

    @abstractmethod
    def _raised_to(self, power: int) -> "Unit":
        ...  # For subclasses to implement.

    #
    #
    # -----------------------------------
    # `Quantity` and `Array` creation API
    #
    # ..via `8 * mV` syntax

    @overload
    def __rmul__(self, other: Scalar) -> "Quantity":
        ...

    @overload
    def __rmul__(self, other: ArrayLike) -> "Array":
        ...

    # 8 * mV  (`other` = 8)
    def __rmul__(self, other):
        # Why import here? To prevent a circular import (`array.py` already imports from
        # `unit.py` at the module level).
        from ..array import Array, Quantity

        if isinstance(other, scalar_types):
            return Quantity(value=other, display_unit=self)
        else:
            return Array(data=other, display_unit=self)

    @overload
    def __rtruediv__(self, other: Literal[1]) -> "Unit":
        ...

    # 8 / mV  (`other` = 8)
    def __rtruediv__(self, other):
        reciprocal_unit = self ** (-1)
        if other == 1:
            # Allow `1 / unit` as a special case, to create a pure `Unit` (and not a
            # `Quantity` with `value = 1`).
            return reciprocal_unit
        else:
            return other * reciprocal_unit


class DataUnit(Unit, ABC):
    """
    A `Unit` in which numeric data is stored in memory.

    See the `Unit.data_unit` property.
    """

    @property
    def data_unit(self):
        return self

    @property
    def data_scale(self):
        return 1
