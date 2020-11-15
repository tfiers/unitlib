from unitlib import define_unit
from unitlib.prefixes import kilo, milli, micro

from .SI_base import meter, second, kilogram

minute = define_unit("min", 60 * second)
hour = define_unit("h", 60 * minute)
day = define_unit("day", 24 * hour)
year = define_unit("year", 365 * day)

liter = define_unit("L")
km = kilometre = kilometer = kilo * meter
ha = hectare = define_unit("ha")

ton = tonne = define_unit("t")
gram = gramme = define_unit("g", kilogram / kilo.factor)
mg = milli * gram
μg = ug = micro * gram
