/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module implements the CSS Sizing Module Level 3 (`css-sizing`) for layout calculations
//! in Servo's Layout 2020 engine. It defines how intrinsic and extrinsic sizing works,
//! including concepts like `min-content`, `max-content`, `fit-content`, and the handling
//! of content sizes, padding, border, and margin (PBM).
//!
//! @see <https://drafts.csswg.org/css-sizing/>

use std::cell::LazyCell;
use std::ops::{Add, AddAssign};

use app_units::Au;
use style::Zero;
use style::values::computed::LengthPercentage;

use crate::context::LayoutContext;
use crate::geom::Size;
use crate::style_ext::{AspectRatio, Clamp, ComputedValuesExt, ContentBoxSizesAndPBM, LayoutStyle};
use crate::{ConstraintSpace, IndefiniteContainingBlock, LogicalVec2};

/// `IntrinsicSizingMode` specifies the mode for intrinsic sizing calculations.
#[derive(PartialEq)]
pub(crate) enum IntrinsicSizingMode {
    /// Used to refer to a min-content contribution or max-content contribution.
    /// This is the size that a box contributes to its containing block’s min-content
    /// or max-content size. Note this is based on the outer size of the box,
    /// and takes into account the relevant sizing properties of the element.
    /// <https://drafts.csswg.org/css-sizing-3/#contributions>
    Contribution,
    /// Used to refer to a min-content size or max-content size.
    /// This is the size based on the contents of an element, without regard for its context.
    /// Note this is usually based on the inner (content-box) size of the box,
    /// and ignores the relevant sizing properties of the element.
    /// <https://drafts.csswg.org/css-sizing-3/#intrinsic>
    Size,
}

/// `ContentSizes` stores the min-content and max-content sizes of an element.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ContentSizes {
    /// The min-content size.
    pub min_content: Au,
    /// The max-content size.
    pub max_content: Au,
}

/// <https://drafts.csswg.org/css-sizing/#intrinsic-sizes>
impl ContentSizes {
    /// Returns a new `ContentSizes` where each component is the maximum of `self` and `other`.
    ///
    /// # Arguments
    /// * `other` - The other `ContentSizes` to compare with.
    ///
    /// Post-condition: A new `ContentSizes` instance with the maximum of corresponding components is returned.
    pub fn max(&self, other: Self) -> Self {
        Self {
            min_content: self.min_content.max(other.min_content),
            max_content: self.max_content.max(other.max_content),
        }
    }

    /// Assigns the maximum of `self` and `other` to `self`.
    ///
    /// # Arguments
    /// * `other` - The other `ContentSizes` to compare with.
    ///
    /// Post-condition: `self` is updated to hold the maximum of its original values and `other`'s.
    pub fn max_assign(&mut self, other: Self) {
        *self = self.max(other);
    }

    /// Computes the union of two `ContentSizes`.
    ///
    /// The min-content of the union is the maximum of the two min-contents,
    /// and the max-content of the union is the sum of the two max-contents.
    ///
    /// # Arguments
    /// * `other` - The other `ContentSizes` to union with.
    ///
    /// Post-condition: A new `ContentSizes` instance representing the union is returned.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min_content: self.min_content.max(other.min_content),
            max_content: self.max_content + other.max_content,
        }
    }

    /// Applies a function to both `min_content` and `max_content`, returning a new `ContentSizes`.
    ///
    /// # Arguments
    /// * `f` - The function to apply to each `Au` component.
    ///
    /// Post-condition: A new `ContentSizes` instance with transformed components is returned.
    pub fn map(&self, f: impl Fn(Au) -> Au) -> Self {
        Self {
            min_content: f(self.min_content),
            max_content: f(self.max_content),
        }
    }
}

impl Zero for ContentSizes {
    /// Returns a `ContentSizes` with both `min_content` and `max_content` set to zero.
    fn zero() -> Self {
        Au::zero().into()
    }

    /// Checks if both `min_content` and `max_content` are zero.
    ///
    /// Post-condition: Returns `true` if both components are zero, `false` otherwise.
    fn is_zero(&self) -> bool {
        self.min_content.is_zero() && self.max_content.is_zero()
    }
}

impl Add for ContentSizes {
    type Output = Self;

    /// Adds two `ContentSizes` by summing their corresponding `min_content` and `max_content` values.
    ///
    /// # Arguments
    /// * `rhs` - The right-hand side `ContentSizes` to add.
    ///
    /// Post-condition: A new `ContentSizes` instance with summed components is returned.
    fn add(self, rhs: Self) -> Self {
        Self {
            min_content: self.min_content + rhs.min_content,
            max_content: self.max_content + rhs.max_content,
        }
    }
}

impl AddAssign for ContentSizes {
    /// Adds `rhs` to `self` by summing their corresponding `min_content` and `max_content` values.
    ///
    /// # Arguments
    /// * `rhs` - The right-hand side `ContentSizes` to add.
    ///
    /// Post-condition: `self` is updated to reflect the sum of its original values and `rhs`.
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs)
    }
}

impl ContentSizes {
    /// Clamps an available size to be between the min-content and max-content sizes.
    /// This is functionally equivalent to CSS "shrink-to-fit" or "fit-content".
    ///
    /// # Arguments
    /// * `available_size` - The size to clamp.
    ///
    /// Post-condition: The `available_size` is returned, clamped between `min_content` and `max_content`.
    /// This formula prioritizes the minimum size for malformed `ContentSizes` where `min_content` exceeds `max_content`.
    pub fn shrink_to_fit(&self, available_size: Au) -> Au {
        // This formula is slightly different than what the spec says,
        // to ensure that the minimum wins for a malformed ContentSize
        // whose min_content is larger than its max_content.
        available_size.min(self.max_content).max(self.min_content)
    }
}

impl From<Au> for ContentSizes {
    /// Converts a single `Au` value into `ContentSizes` where both `min_content` and `max_content` are equal to the input `Au`.
    ///
    /// # Arguments
    /// * `size` - The `Au` value to convert.
    ///
    /// Post-condition: A new `ContentSizes` instance is returned, with `min_content` and `max_content` equal to `size`.
    fn from(size: Au) -> Self {
        Self {
            min_content: size,
            max_content: size,
        }
    }
}

/// Computes the outer inline content sizes for an element, taking into account
/// its layout style, containing block, and intrinsic sizing properties.
///
/// This function calculates `min-content` and `max-content` contributions and sizes
/// for an element's inline axis, handling various CSS sizing properties and their
/// interactions, such as `min-width`, `max-width`, and aspect ratios.
///
/// # Arguments
/// * `layout_style` - The computed layout style of the element.
/// * `containing_block` - Information about the element's containing block.
/// * `auto_minimum` - The automatic minimum size for the element.
/// * `auto_block_size_stretches_to_containing_block` - Indicates if block size stretches to containing block.
/// * `is_replaced` - `true` if the element is a replaced element (e.g., `<img>`, `<video>`).
/// * `establishes_containing_block` - `true` if the element establishes a containing block.
/// * `get_preferred_aspect_ratio` - A closure to get the preferred aspect ratio.
/// * `get_content_size` - A closure to compute the intrinsic content size.
///
/// Pre-condition: Input `layout_style` and `containing_block` are valid.
/// Post-condition: An `InlineContentSizesResult` is returned, containing the computed
/// min-content and max-content sizes for the inline axis, and whether these depend
/// on block constraints.
#[allow(clippy::too_many_arguments)]
pub(crate) fn outer_inline(
    layout_style: &LayoutStyle,
    containing_block: &IndefiniteContainingBlock,
    auto_minimum: &LogicalVec2<Au>,
    auto_block_size_stretches_to_containing_block: bool,
    is_replaced: bool,
    establishes_containing_block: bool,
    get_preferred_aspect_ratio: impl FnOnce(&LogicalVec2<Au>) -> Option<AspectRatio>,
    get_content_size: impl FnOnce(&ConstraintSpace) -> InlineContentSizesResult,
) -> InlineContentSizesResult {
    let ContentBoxSizesAndPBM {
        content_box_sizes,
        pbm,
        mut depends_on_block_constraints,
        preferred_size_computes_to_auto,
    } = layout_style.content_box_sizes_and_padding_border_margin(containing_block);
    let margin = pbm.margin.map(|v| v.auto_is(Au::zero));
    let pbm_sums = LogicalVec2 {
        block: pbm.padding_border_sums.block + margin.block_sum(),
        inline: pbm.padding_border_sums.inline + margin.inline_sum(),
    };
    let style = layout_style.style();
    // Functional Utility: Lazily computes the intrinsic content size.
    let content_size = LazyCell::new(|| {
        let constraint_space = if establishes_containing_block {
            let available_block_size = containing_block
                .size
                .block
                .map(|v| Au::zero().max(v - pbm_sums.block));
            // Block Logic: Determines automatic block size based on preferred size and stretching.
            let automatic_size = if preferred_size_computes_to_auto.block &&
                auto_block_size_stretches_to_containing_block
            {
                depends_on_block_constraints = true;
                Size::Stretch
            } else {
                Size::FitContent
            };
            ConstraintSpace::new(
                content_box_sizes.block.resolve_extrinsic(
                    automatic_size,
                    auto_minimum.block,
                    available_block_size,
                ),
                style.writing_mode,
                get_preferred_aspect_ratio(&pbm.padding_border_sums),
            )
        } else {
            // This assumes that there is no preferred aspect ratio, or that there is no
            // block size constraint to be transferred so the ratio is irrelevant.
            // We only get into here for anonymous blocks, for which the assumption holds.
            ConstraintSpace::new(
                containing_block.size.block.into(),
                containing_block.writing_mode,
                None,
            )
        };
        get_content_size(&constraint_space)
    });
    // Functional Utility: Resolves non-initial inline sizes (MinContent, MaxContent, FitContent, Stretch).
    let resolve_non_initial = |inline_size, stretch_values| {
        Some(match inline_size {
            Size::Initial => return None,
            Size::Numeric(numeric) => (numeric, numeric, false),
            Size::MinContent => (
                content_size.sizes.min_content,
                content_size.sizes.min_content,
                content_size.depends_on_block_constraints,
            ),
            Size::MaxContent => (
                content_size.sizes.max_content,
                content_size.sizes.max_content,
                content_size.depends_on_block_constraints,
            ),
            Size::FitContent => (
                content_size.sizes.min_content,
                content_size.sizes.max_content,
                content_size.depends_on_block_constraints,
            ),
            Size::FitContentFunction(size) => {
                let size = content_size.sizes.shrink_to_fit(size);
                (size, size, content_size.depends_on_block_constraints)
            },
            Size::Stretch => return stretch_values,
        })
    };
    let (mut preferred_min_content, preferred_max_content, preferred_depends_on_block_constraints) =
        resolve_non_initial(content_box_sizes.inline.preferred, None)
            .unwrap_or_else(|| resolve_non_initial(Size::FitContent, None).unwrap());
    let (mut min_min_content, mut min_max_content, mut min_depends_on_block_constraints) =
        resolve_non_initial(
            content_box_sizes.inline.min,
            Some((Au::zero(), Au::zero(), false)),
        )
        .unwrap_or((auto_minimum.inline, auto_minimum.inline, false));
    let (mut max_min_content, max_max_content, max_depends_on_block_constraints) =
        resolve_non_initial(content_box_sizes.inline.max, None)
            .map(|(min_content, max_content, depends_on_block_constraints)| {
                (
                    Some(min_content),
                    Some(max_content),
                    depends_on_block_constraints,
                )
            })
            .unwrap_or_default();

    // https://drafts.csswg.org/css-sizing-3/#replaced-percentage-min-contribution
    // > If the box is replaced, a cyclic percentage in the value of any max size property
    // > or preferred size property (width/max-width/height/max-height), is resolved against
    // > zero when calculating the min-content contribution in the corresponding axis.
    //
    // This means that e.g. the min-content contribution of `width: calc(100% + 100px)`
    // should be 100px, but it's just zero on other browsers, so we do the same.
    // Block Logic: Adjusts min-content for replaced elements with percentage-based sizing.
    if is_replaced {
        // Functional Utility: Checks if a `Size<LengthPercentage>` contains a percentage value.
        let has_percentage = |size: Size<LengthPercentage>| {
            // We need a comment here to avoid breaking `./mach test-tidy`.
            matches!(size, Size::Numeric(numeric) if numeric.has_percentage())
        };
        if content_box_sizes.inline.preferred.is_initial() &&
            has_percentage(style.box_size(containing_block.writing_mode).inline)
        {
            preferred_min_content = Au::zero();
        }
        if content_box_sizes.inline.max.is_initial() &&
            has_percentage(style.max_box_size(containing_block.writing_mode).inline)
        {
            max_min_content = Some(Au::zero());
        }
    }

    // Regardless of their sizing properties, tables are always forced to be at least
    // as big as their min-content size, so floor the minimums.
    // Block Logic: Ensures tables are at least their min-content size.
    if layout_style.is_table() {
        min_min_content.max_assign(content_size.sizes.min_content);
        min_max_content.max_assign(content_size.sizes.min_content);
        min_depends_on_block_constraints |= content_size.depends_on_block_constraints;
    }

    InlineContentSizesResult {
        sizes: ContentSizes {
            min_content: preferred_min_content
                .clamp_between_extremums(min_min_content, max_min_content) +
                pbm_sums.inline,
            max_content: preferred_max_content
                .clamp_between_extremums(min_max_content, max_max_content) +
                pbm_sums.inline,
        },
        depends_on_block_constraints: depends_on_block_constraints &&
            (preferred_depends_on_block_constraints ||
                min_depends_on_block_constraints ||
                max_depends_on_block_constraints),
    }
}

/// `InlineContentSizesResult` encapsulates the computed intrinsic content sizes for the inline axis.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InlineContentSizesResult {
    /// The computed `ContentSizes` (min-content and max-content) for the inline axis.
    pub sizes: ContentSizes,
    /// A boolean indicating whether the computed sizes depend on block constraints.
    pub depends_on_block_constraints: bool,
}

/// `ComputeInlineContentSizes` is a trait for types that can compute their intrinsic
/// inline content sizes based on a given layout context and constraint space.
pub(crate) trait ComputeInlineContentSizes {
    /// Computes the intrinsic inline content sizes.
    ///
    /// # Arguments
    /// * `layout_context` - The current layout context.
    /// * `constraint_space` - The constraint space for sizing.
    ///
    /// Post-condition: An `InlineContentSizesResult` containing the computed sizes is returned.
    fn compute_inline_content_sizes(
        &self,
        layout_context: &LayoutContext,
        constraint_space: &ConstraintSpace,
    ) -> InlineContentSizesResult;
}
