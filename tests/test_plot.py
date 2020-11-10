import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from unitlib import Unit
from unitlib.prefixes import milli

second = Unit.define("s")
ms = Unit.from_prefix(milli, second)
volt = Unit.define("V")
mV = Unit.from_prefix(milli, volt)

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
