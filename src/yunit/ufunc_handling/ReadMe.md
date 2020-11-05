Implementation of `Array.__array_ufunc__`.

This handles all Python math syntax (`*`, `*=`, `**`, `<`, `%`, …)
and all NumPy elementwise functions (`cos`, `sign`, `abs`, …),
for both `Array`s, `Quantity`s, and `Unit`s.
