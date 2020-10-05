import matplotlib.pyplot as plt
import numpy as np

from yunit import Unit
from yunit.prefixes import milli

second = Unit.define("s")
ms = Unit.from_prefix(milli, second)
volt = Unit.define("V")
mV = Unit.from_prefix(milli, volt)

# x = np.linspace(0, 10) * ms
# y = np.random.randn(len(x)) * mV

# plt.plot(x, y)
