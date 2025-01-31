# DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/python.rs
# Based on "crates/re_types/definitions/rerun/components/resolution.fbs".

# You can extend this class by creating a "ResolutionExt" class in "resolution_ext.py".

from __future__ import annotations

from .. import datatypes
from .._baseclasses import ComponentBatchMixin

__all__ = ["Resolution", "ResolutionBatch", "ResolutionType"]


class Resolution(datatypes.Vec2D):
    """
    Pixel resolution width & height, e.g. of a camera sensor.

    Typically in integer units, but for some usecases floating point may be used.
    """

    # You can define your own __init__ function as a member of ResolutionExt in resolution_ext.py

    # Note: there are no fields here because Resolution delegates to datatypes.Vec2D
    pass


class ResolutionType(datatypes.Vec2DType):
    _TYPE_NAME: str = "rerun.components.Resolution"


class ResolutionBatch(datatypes.Vec2DBatch, ComponentBatchMixin):
    _ARROW_TYPE = ResolutionType()


# TODO(cmc): bring back registration to pyarrow once legacy types are gone
# pa.register_extension_type(ResolutionType())
