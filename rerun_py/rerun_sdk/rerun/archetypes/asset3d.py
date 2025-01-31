# DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/python.rs
# Based on "crates/re_types/definitions/rerun/archetypes/asset3d.fbs".

# You can extend this class by creating a "Asset3DExt" class in "asset3d_ext.py".

from __future__ import annotations

from typing import Any

from attrs import define, field

from .. import components, datatypes
from .._baseclasses import Archetype
from .asset3d_ext import Asset3DExt

__all__ = ["Asset3D"]


@define(str=False, repr=False, init=False)
class Asset3D(Asset3DExt, Archetype):
    """
    A prepacked 3D asset (`.gltf`, `.glb`, `.obj`, etc).

    Examples
    --------
    Simple 3D asset:
    ```python
    import sys

    import rerun as rr

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_asset.[gltf|glb]>")
        sys.exit(1)

    rr.init("rerun_example_asset3d_simple", spawn=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)  # Set an up-axis
    rr.log("world/asset", rr.Asset3D.from_file(sys.argv[1]))
    ```

    3D asset with out-of-tree transform:
    ```python
    import sys

    import numpy as np
    import rerun as rr
    from rerun.components import OutOfTreeTransform3DBatch
    from rerun.datatypes import TranslationRotationScale3D

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_asset.[gltf|glb]>")
        sys.exit(1)

    rr.init("rerun_example_asset3d_out_of_tree", spawn=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)  # Set an up-axis

    rr.set_time_sequence("frame", 0)
    rr.log("world/asset", rr.Asset3D.from_file(sys.argv[1]))
    # Those points will not be affected by their parent's out-of-tree transform!
    rr.log(
        "world/asset/points",
        rr.Points3D(np.vstack([xyz.ravel() for xyz in np.mgrid[3 * [slice(-10, 10, 10j)]]]).T),
    )

    asset = rr.Asset3D.from_file(sys.argv[1])
    for i in range(1, 20):
        rr.set_time_sequence("frame", i)

        translation = TranslationRotationScale3D(translation=[0, 0, i - 10.0])
        rr.log_components("asset", [OutOfTreeTransform3DBatch(translation)])
    ```
    """

    def __init__(
        self: Any,
        data: components.BlobLike,
        media_type: datatypes.Utf8Like | None = None,
        transform: datatypes.Transform3DLike | None = None,
    ):
        """
        Create a new instance of the Asset3D archetype.

        Parameters
        ----------
        data:
             The asset's bytes.
        media_type:
             The Media Type of the asset.

             For instance:
             * `model/gltf-binary`
             * `model/obj`

             If omitted, the viewer will try to guess from the data.
             If it cannot guess, it won't be able to render the asset.
        transform:
             An out-of-tree transform.

             Applies a transformation to the asset itself without impacting its children.
        """

        # You can define your own __init__ function as a member of Asset3DExt in asset3d_ext.py
        self.__attrs_init__(data=data, media_type=media_type, transform=transform)

    data: components.BlobBatch = field(
        metadata={"component": "required"},
        converter=components.BlobBatch,  # type: ignore[misc]
    )
    """
    The asset's bytes.
    """

    media_type: components.MediaTypeBatch | None = field(
        metadata={"component": "optional"},
        default=None,
        converter=components.MediaTypeBatch._optional,  # type: ignore[misc]
    )
    """
    The Media Type of the asset.

    For instance:
    * `model/gltf-binary`
    * `model/obj`

    If omitted, the viewer will try to guess from the data.
    If it cannot guess, it won't be able to render the asset.
    """

    transform: components.OutOfTreeTransform3DBatch | None = field(
        metadata={"component": "optional"},
        default=None,
        converter=components.OutOfTreeTransform3DBatch._optional,  # type: ignore[misc]
    )
    """
    An out-of-tree transform.

    Applies a transformation to the asset itself without impacting its children.
    """

    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__


if hasattr(Asset3DExt, "deferred_patch_class"):
    Asset3DExt.deferred_patch_class(Asset3D)
