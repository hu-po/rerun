// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/cpp/mod.rs
// Based on "crates/re_types/definitions/rerun/archetypes/segmentation_image.fbs".

#pragma once

#include "../arrow.hpp"
#include "../component_batch.hpp"
#include "../components/draw_order.hpp"
#include "../components/tensor_data.hpp"
#include "../data_cell.hpp"
#include "../result.hpp"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace rerun {
    namespace archetypes {
        /// An image made up of integer class-ids
        ///
        /// The shape of the `TensorData` must be mappable to an `HxW` tensor.
        /// Each pixel corresponds to a depth value in units specified by meter.
        ///
        /// Leading and trailing unit-dimensions are ignored, so that
        /// `1x640x480x1` is treated as a `640x480` image.
        struct SegmentationImage {
            /// The image data. Should always be a rank-2 tensor.
            rerun::components::TensorData data;

            /// An optional floating point value that specifies the 2D drawing order.
            /// Objects with higher values are drawn on top of those with lower values.
            std::optional<rerun::components::DrawOrder> draw_order;

            /// Name of the indicator component, used to identify the archetype when converting to a
            /// list of components.
            static const char INDICATOR_COMPONENT_NAME[];

          public:
            SegmentationImage() = default;

            SegmentationImage(rerun::components::TensorData _data) : data(std::move(_data)) {}

            /// An optional floating point value that specifies the 2D drawing order.
            /// Objects with higher values are drawn on top of those with lower values.
            SegmentationImage& with_draw_order(rerun::components::DrawOrder _draw_order) {
                draw_order = std::move(_draw_order);
                return *this;
            }

            /// Returns the number of primary instances of this archetype.
            size_t num_instances() const {
                return 1;
            }

            /// Creates an `AnonymousComponentBatch` out of the associated indicator component. This
            /// allows for associating arbitrary indicator components with arbitrary data. Check out
            /// the `manual_indicator` API example to see what's possible.
            static AnonymousComponentBatch indicator();

            /// Collections all component lists into a list of component collections. *Attention:*
            /// The returned vector references this instance and does not take ownership of any
            /// data. Adding any new components to this archetype will invalidate the returned
            /// component lists!
            std::vector<AnonymousComponentBatch> as_component_batches() const;
        };
    } // namespace archetypes
} // namespace rerun
