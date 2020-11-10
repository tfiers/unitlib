import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from unitlib import define_unit
from unitlib.prefixes import milli

second = define_unit("s")
ms = milli * second
volt = define_unit("V")
mV = milli * volt

x = np.linspace(0, 10) * ms
y = np.random.randn(len(x)) * mV

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


if __name__ == "__main__":
    plt.show()
