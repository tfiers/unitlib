from ._3_compound_unit import CompoundUnit
from .._3_unit import DataUnit
from ...backwards_compatibility import TYPE_CHECKING

if TYPE_CHECKING:
    from ._1_unit_atom import UnitAtom, DataUnitAtom, dimensionless


class PoweredUnitAtom(CompoundUnit):
    """
    Eg "mm²"; as contrasted with "mm" (a `UnitAtom`) or "N/mm²" (a `CompoundUnit`).

    Defining attributes:
        - unit_atom: UnitAtom
        - power: int
    """

    def __new__(cls, unit_atom: "UnitAtom", power: int):
        if power == 0:
            return dimensionless
        elif power == 1:
            return unit_atom
        else:
            return object.__new__(cls)
            #   `cls` can be `PoweredUnitAtom` or `PoweredDataUnitAtom`.

    def __init__(self, unit_atom: "UnitAtom", power: int):
        CompoundUnit.__init__(self, components=(self,))
        self.unit_atom = unit_atom
        self.power = power

    @property
    def name(self):
        return f"{self.unit_atom.name}{self._power_as_superscript}"

    @property
    def data_unit(self):
        return self.unit_atom.data_unit ** self.power

    @property
    def scale(self):
        return self.unit_atom.scale ** self.power

    def __hash__(self):
        return hash((self.unit_atom, self.power))

    # (mV⁻³)**2
    def _raised_to(self, power: int):
        return self.__class__(self.unit_atom, self.power * power)
        #   `self.__class__` can be `PoweredUnitAtom` or `PoweredDataUnitAtom`.

    @property
    def _power_as_superscript(self) -> str:
        # Transform eg the int `-87` to the string `"⁻⁸⁷"`.
        superscript_chars = "⁰¹²³⁴⁵⁶⁷⁸⁹"
        digits = str(abs(self.power))
        output_chars = [superscript_chars[int(digit)] for digit in digits]
        if self.power < 0:
            output_chars.insert(0, "⁻")
        return "".join(output_chars)


class PoweredDataUnitAtom(DataUnit, PoweredUnitAtom):
    def __new__(cls, unit_atom: "DataUnitAtom", power: int):
        return PoweredUnitAtom.__new__(cls, unit_atom, power)
