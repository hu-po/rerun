# DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/python.rs
# Based on "crates/re_types/definitions/rerun/archetypes/bar_chart.fbs".

# You can extend this class by creating a "BarChartExt" class in "bar_chart_ext.py".

from __future__ import annotations

from attrs import define, field

from .. import components
from .._baseclasses import (
    Archetype,
)
from .bar_chart_ext import BarChartExt

__all__ = ["BarChart"]


@define(str=False, repr=False)
class BarChart(BarChartExt, Archetype):
    """
    A Barchart.

    The x values will be the indices of the array, and the bar heights will be the provided values.

    Example
    -------
    ```python

    import rerun as rr
    import rerun.experimental as rr2

    rr.init("rerun_example_bar_chart", spawn=True)
    rr2.log("bar_chart", rr2.BarChart([8, 4, 0, 9, 1, 4, 1, 6, 9, 0]))
    ```
    """

    # You can define your own __init__ function as a member of BarChartExt in bar_chart_ext.py

    values: components.TensorDataArray = field(
        metadata={"component": "required"},
        converter=BarChartExt.values__field_converter_override,  # type: ignore[misc]
    )
    """
    The values. Should always be a rank-1 tensor.
    """

    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__
