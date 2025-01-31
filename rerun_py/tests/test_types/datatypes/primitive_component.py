# DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/python.rs
# Based on "crates/re_types/definitions/rerun/testing/components/fuzzy_deps.fbs".

# You can extend this class by creating a "PrimitiveComponentExt" class in "primitive_component_ext.py".

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field
from rerun._baseclasses import BaseBatch, BaseExtensionType

__all__ = [
    "PrimitiveComponent",
    "PrimitiveComponentArrayLike",
    "PrimitiveComponentBatch",
    "PrimitiveComponentLike",
    "PrimitiveComponentType",
]


@define(init=False)
class PrimitiveComponent:
    def __init__(self: Any, value: PrimitiveComponentLike):
        """Create a new instance of the PrimitiveComponent datatype."""

        # You can define your own __init__ function as a member of PrimitiveComponentExt in primitive_component_ext.py
        self.__attrs_init__(value=value)

    value: int = field(converter=int)

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        # You can define your own __array__ function as a member of PrimitiveComponentExt in primitive_component_ext.py
        return np.asarray(self.value, dtype=dtype)

    def __int__(self) -> int:
        return int(self.value)


PrimitiveComponentLike = PrimitiveComponent
PrimitiveComponentArrayLike = Union[
    PrimitiveComponent,
    Sequence[PrimitiveComponentLike],
]


class PrimitiveComponentType(BaseExtensionType):
    _TYPE_NAME: str = "rerun.testing.datatypes.PrimitiveComponent"

    def __init__(self) -> None:
        pa.ExtensionType.__init__(self, pa.uint32(), self._TYPE_NAME)


class PrimitiveComponentBatch(BaseBatch[PrimitiveComponentArrayLike]):
    _ARROW_TYPE = PrimitiveComponentType()

    @staticmethod
    def _native_to_pa_array(data: PrimitiveComponentArrayLike, data_type: pa.DataType) -> pa.Array:
        raise NotImplementedError  # You need to implement native_to_pa_array_override in primitive_component_ext.py


# TODO(cmc): bring back registration to pyarrow once legacy types are gone
# pa.register_extension_type(PrimitiveComponentType())
