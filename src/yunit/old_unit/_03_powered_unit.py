from ._01_base_class import DataUnit, OldUnitABC
from ..backwards_compatibility import TYPE_CHECKING

if TYPE_CHECKING:
    # See the explanation of `if TYPE_CHECKING` in ../ReadMe.
    from ._02_simple_unit import UnitAtom, DataUnitAtom


class UnitCompoment(OldUnitABC):
    """
    Eg "mm²"; as contrasted with "mm" (a `SimpleUnit`) or "N/mm²" (a `CompoundUnit`).

    Attributes:
        - ground_unit: SimpleUnit
        - power: int
    """

    def __new__(cls, ground_unit: "UnitAtom", power: int):

        if power == 0:
            from ._02_simple_unit import dimensionless

            return dimensionless

        elif power == 1:
            return ground_unit

        else:
            powered_unit = object.__new__(cls)
            #   `cls` can be `PoweredUnit` or `PoweredDataUnit`.
            powered_unit.ground_unit = ground_unit
            powered_unit.power = power
            return powered_unit

    @property
    def name(self):
        return f"{self.ground_unit.name}{self._power_as_superscript}"

    @property
    def data_unit(self):
        return self.ground_unit.data_unit ** self.power

    @property
    def data_scale(self):
        return self.ground_unit.data_scale ** self.power

    def __hash__(self):
        return hash((self.ground_unit, self.power))

    # (mV⁻³)**2
    def _raised_to(self, power: int):
        return self.__class__(self.ground_unit, self.power * power)
        #   `self.__class__` can be `PoweredUnit` or `PoweredDataUnit`.

    @property
    def _power_as_superscript(self) -> str:
        # Transform eg the int `-87` to the string `"⁻⁸⁷"`.
        superscript_chars = "⁰¹²³⁴⁵⁶⁷⁸⁹"
        digits = str(abs(self.power))
        output_chars = [superscript_chars[int(digit)] for digit in digits]
        if self.power < 0:
            output_chars.insert(0, "⁻")
        return "".join(output_chars)


class DataUnitCompoment(DataUnit, UnitCompoment):
    def __new__(cls, ground_unit: "DataUnitAtom", power: int):
        return UnitCompoment.__new__(cls, ground_unit, power)
