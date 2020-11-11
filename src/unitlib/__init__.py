from .core_objects import Unit, Quantity, Array, define_unit

# Do not import `.support.auto_axis_labelling` by default, to spare the heavy matplotlib
# import when it is not needed.
def enable_auto_axis_labelling():
    """
    After calling this, when plotting unitlib `Array`s with matplotlib, their axes will
    automatically be labelled with their `display_unit` and `name` (if specified).
    """
    from .support import auto_axis_labelling
