import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from unitlib import define_unit, enable_auto_axis_labelling
from unitlib.prefixes import milli


enable_auto_axis_labelling()

second = define_unit("s")
ms = milli * second
volt = define_unit("V")
mV = milli * volt

x = np.linspace(0, 10) * ms
x.name = "Time"

y = np.random.randn(len(x)) * mV
y.name = "Membrane potential"

fig, ax = plt.subplots()
ax: Axes
ax.plot(x, y)


def test_axlabels():
    assert ax.get_xlabel() == "Time (ms)"
    assert ax.get_ylabel() == "Membrane potential (mV)"


def test_plotted_data_is_in_display_units():
    line: Line2D = ax.lines[0]
    assert all(line.get_xdata() == x)
    assert all(line.get_xdata() != x.data)
    assert all(line.get_ydata() == y)
    assert all(line.get_ydata() != y.data)


if __name__ == "__main__":
    plt.show()
