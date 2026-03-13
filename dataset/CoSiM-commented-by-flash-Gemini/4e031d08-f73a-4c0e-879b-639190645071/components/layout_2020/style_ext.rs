/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module provides extensions and utility functions for Stylo's
//! `ComputedValues` and other style-related structures within Servo's
//! Layout 2020 engine. It focuses on bridging the gap between CSS
//! computed styles and the layout-relevant information required for
//! rendering, particularly concerning box model calculations, sizing,
//! overflow handling, stacking contexts, and transformations.

use app_units::Au;
use style::Zero;
use style::color::AbsoluteColor;
use style::computed_values::direction::T as Direction;
use style::computed_values::isolation::T as ComputedIsolation;
use style::computed_values::mix_blend_mode::T as ComputedMixBlendMode;
use style::computed_values::position::T as ComputedPosition;
use style::computed_values::transform_style::T as ComputedTransformStyle;
use style::computed_values::unicode_bidi::T as UnicodeBidi;
use style::logical_geometry::{Direction as AxisDirection, WritingMode};
use style::properties::ComputedValues;
use style::properties::longhands::backface_visibility::computed_value::T as BackfaceVisiblity;
use style::properties::longhands::box_sizing::computed_value::T as BoxSizing;
use style::properties::longhands::column_span::computed_value::T as ColumnSpan;
use style::properties::style_structs::Border;
use style::servo::selector_parser::PseudoElement;
use style::values::CSSFloat;
use style::values::computed::basic_shape::ClipPath;
use style::values::computed::image::Image as ComputedImageLayer;
use style::values::computed::{AlignItems, BorderStyle, Color, Inset, LengthPercentage, Margin};
use style::values::generics::box_::Perspective;
use style::values::generics::position::{GenericAspectRatio, PreferredRatio};
use style::values::generics::transform::{GenericRotate, GenericScale, GenericTranslate};
use style::values::specified::align::AlignFlags;
use style::values::specified::{Overflow, WillChangeBits, box_ as stylo};
use webrender_api as wr;
use webrender_api::units::LayoutTransform;

use crate::dom_traversal::{Contents, NonReplacedContents};
use crate::fragment_tree::FragmentFlags;
use crate::geom::{
    AuOrAuto, LengthPercentageOrAuto, LogicalSides, LogicalSides1D, LogicalVec2, PhysicalSides,
    PhysicalSize, Size, Sizes,
};
use crate::table::TableLayoutStyle;
use crate::{ContainingBlock, IndefiniteContainingBlock};

/// `Display` represents the resolved display type of a box, handling special cases
/// like `none`, `contents`, and generating boxes.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum Display {
    /// The element is not rendered.
    None,
    /// The element itself does not generate a box, but its contents and pseudo-elements do.
    Contents,
    /// The element generates a box, which can be an outside-inside box or a layout-internal box.
    GeneratingBox(DisplayGeneratingBox),
}

/// `DisplayGeneratingBox` specifies how a box generates its main box, either as
/// an outside-inside box or a layout-internal box.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DisplayGeneratingBox {
    /// A box with an outer display type (e.g., `block`, `inline`) and an inner display type (e.g., `flow`, `flex`).
    OutsideInside {
        outside: DisplayOutside,
        inside: DisplayInside,
    },
    /// A layout-internal box type (e.g., table parts).
    /// <https://drafts.csswg.org/css-display-3/#layout-specific-display>
    LayoutInternal(DisplayLayoutInternal),
}

/// `AxesOverflow` specifies the overflow behavior for both the X and Y axes.
#[derive(Clone, Copy, Debug)]
pub struct AxesOverflow {
    /// Overflow behavior for the horizontal (X) axis.
    pub x: Overflow,
    /// Overflow behavior for the vertical (Y) axis.
    pub y: Overflow,
}

impl DisplayGeneratingBox {
    /// Returns the inner display type of the generating box.
    ///
    /// Post-condition: A `DisplayInside` enum representing the inner display type is returned.
    pub(crate) fn display_inside(&self) -> DisplayInside {
        match *self {
            DisplayGeneratingBox::OutsideInside { inside, .. } => inside,
            DisplayGeneratingBox::LayoutInternal(layout_internal) => {
                layout_internal.display_inside()
            },
        }
    }

    /// Returns the used value for the contents of the display generating box.
    ///
    /// This method adjusts the display type based on whether the element is replaced
    /// or a text control (e.g., input, textarea), aligning with CSS specifications.
    ///
    /// # Arguments
    /// * `contents` - The `Contents` of the element.
    ///
    /// Post-condition: A `DisplayGeneratingBox` representing the used value for contents is returned.
    pub(crate) fn used_value_for_contents(&self, contents: &Contents) -> Self {
        // From <https://www.w3.org/TR/css-display-3/#layout-specific-display>:
        // > When the display property of a replaced element computes to one of
        // > the layout-internal values, it is handled as having a used value of
        // > inline.
        if matches!(self, Self::LayoutInternal(_)) && contents.is_replaced() {
            Self::OutsideInside {
                outside: DisplayOutside::Inline,
                inside: DisplayInside::Flow {
                    is_list_item: false,
                },
            }
        } else if matches!(
            contents,
            Contents::NonReplaced(NonReplacedContents::OfTextControl)
        ) {
            // If it's an input or textarea, make sure the display-inside is flow-root.
            // <https://html.spec.whatwg.org/multipage/#form-controls>
            // Block Logic: Ensures text controls use `flow-root` inner display.
            if let DisplayGeneratingBox::OutsideInside { outside, .. } = self {
                DisplayGeneratingBox::OutsideInside {
                    outside: *outside,
                    inside: DisplayInside::FlowRoot {
                        is_list_item: false,
                    },
                }
            } else {
                *self
            }
        } else {
            *self
        }
    }
}

/// `DisplayOutside` specifies the outer display type of a box.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DisplayOutside {
    Block,
    Inline,
}

/// `DisplayInside` specifies the inner display type of a box.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DisplayInside {
    // “list-items are limited to the Flow Layout display types”
    // <https://drafts.csswg.org/css-display/#list-items>
    Flow { is_list_item: bool },
    FlowRoot { is_list_item: bool },
    Flex,
    Grid,
    Table,
}

/// `DisplayLayoutInternal` specifies layout-internal display types,
/// typically for table-related elements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::enum_variant_names)]
/// <https://drafts.csswg.org/css-display-3/#layout-specific-display>
pub(crate) enum DisplayLayoutInternal {
    TableCaption,
    TableCell,
    TableColumn,
    TableColumnGroup,
    TableFooterGroup,
    TableHeaderGroup,
    TableRow,
    TableRowGroup,
}

impl DisplayLayoutInternal {
    /// Returns the inner display type for layout-internal boxes.
    ///
    /// <https://drafts.csswg.org/css-display-3/#layout-specific-displa>
    /// Post-condition: Always returns `DisplayInside::FlowRoot { is_list_item: false }`.
    pub(crate) fn display_inside(&self) -> DisplayInside {
        // When we add ruby, the display_inside of ruby must be Flow.
        // TODO: this should be unreachable for everything but
        // table cell and caption, once we have box tree fixups.
        DisplayInside::FlowRoot {
            is_list_item: false,
        }
    }
}

/// `PaddingBorderMargin` stores resolved padding, border, and margin values
/// for an element, including pre-computed sums for layout efficiency.
#[derive(Clone, Debug)]
pub(crate) struct PaddingBorderMargin {
    /// The resolved padding values for each logical side.
    pub padding: LogicalSides<Au>,
    /// The resolved border values for each logical side.
    pub border: LogicalSides<Au>,
    /// The resolved margin values for each logical side, allowing for `auto` values.
    pub margin: LogicalSides<AuOrAuto>,

    /// Pre-computed sums of padding and border in each axis for quick access.
    pub padding_border_sums: LogicalVec2<Au>,
}

impl PaddingBorderMargin {
    /// Returns a `PaddingBorderMargin` with all values set to zero.
    pub(crate) fn zero() -> Self {
        Self {
            padding: LogicalSides::zero(),
            border: LogicalSides::zero(),
            margin: LogicalSides::zero(),
            padding_border_sums: LogicalVec2::zero(),
        }
    }

    /// Computes the sums of padding, border, and margin, treating `auto` margins as zero.
    ///
    /// # Arguments
    /// * `ignore_block_margins` - Specifies whether to ignore block margins in the sum.
    ///
    /// Post-condition: A `LogicalVec2<Au>` representing the total sums is returned.
    pub(crate) fn sums_auto_is_zero(
        &self,
        ignore_block_margins: LogicalSides1D<bool>,
    ) -> LogicalVec2<Au> {
        let margin = self.margin.auto_is(Au::zero);
        let mut sums = self.padding_border_sums;
        sums.inline += margin.inline_sum();
        // Block Logic: Conditionally adds block margins to sums based on `ignore_block_margins`.
        if !ignore_block_margins.start {
            sums.block += margin.block_start;
        }
        if !ignore_block_margins.end {
            sums.block += margin.block_end;
        }
        sums
    }
}

/// `AspectRatio` represents the resolved aspect ratio property of an element,
/// including adjustments for `box-sizing`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AspectRatio {
    /// If the element that this aspect ratio belongs to uses box-sizing:
    /// border-box, and the aspect-ratio property does not contain "auto", then
    /// the aspect ratio is in respect to the border box. This will then contain
    /// the summed sizes of the padding and border. Otherwise, it's 0.
    box_sizing_adjustment: LogicalVec2<Au>,
    /// The ratio itself (inline over block).
    i_over_b: CSSFloat,
}

impl AspectRatio {
    /// Given one side length, computes the other side length based on the aspect ratio.
    ///
    /// # Arguments
    /// * `ratio_dependent_axis` - The axis whose size is to be computed.
    /// * `ratio_determining_size` - The size of the determining axis.
    ///
    /// Post-condition: The computed dependent size in `Au` is returned.
    pub(crate) fn compute_dependent_size(
        &self,
        ratio_dependent_axis: AxisDirection,
        ratio_determining_size: Au,
    ) -> Au {
        match ratio_dependent_axis {
            // Calculate the inline size from the block size
            AxisDirection::Inline => {
                (ratio_determining_size + self.box_sizing_adjustment.block).scale_by(self.i_over_b) -
                    self.box_sizing_adjustment.inline
            },
            // Calculate the block size from the inline size
            AxisDirection::Block => {
                (ratio_determining_size + self.box_sizing_adjustment.inline)
                    .scale_by(1.0 / self.i_over_b) -
                    self.box_sizing_adjustment.block
            },
        }
    }

    /// Creates an `AspectRatio` from a content ratio, assuming no box-sizing adjustment.
    ///
    /// # Arguments
    /// * `i_over_b` - The inline-over-block ratio.
    ///
    /// Post-condition: A new `AspectRatio` instance is returned with zero `box_sizing_adjustment`.
    pub(crate) fn from_content_ratio(i_over_b: CSSFloat) -> Self {
        Self {
            box_sizing_adjustment: LogicalVec2::zero(),
            i_over_b,
        }
    }
}

/// `ContentBoxSizesAndPBM` combines content box sizes (preferred, min, max) with
/// padding, border, and margin information.
#[derive(Clone)]
pub(crate) struct ContentBoxSizesAndPBM {
    /// The preferred, min, and max content box sizes in both block and inline dimensions.
    pub content_box_sizes: LogicalVec2<Sizes>,
    /// Resolved padding, border, and margin values.
    pub pbm: PaddingBorderMargin,
    /// Indicates whether the computed sizes depend on block constraints.
    pub depends_on_block_constraints: bool,
    /// Indicates whether the preferred size for each axis computes to `auto`.
    pub preferred_size_computes_to_auto: LogicalVec2<bool>,
}

/// `BorderStyleColor` represents the style and color of a single border side.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BorderStyleColor {
    /// The style of the border (e.g., `solid`, `dashed`, `none`).
    pub style: BorderStyle,
    /// The absolute color of the border.
    pub color: AbsoluteColor,
}

impl BorderStyleColor {
    /// Creates a new `BorderStyleColor` instance.
    ///
    /// # Arguments
    /// * `style` - The `BorderStyle`.
    /// * `color` - The `AbsoluteColor`.
    ///
    /// Post-condition: A new `BorderStyleColor` instance is returned.
    pub(crate) fn new(style: BorderStyle, color: AbsoluteColor) -> Self {
        Self { style, color }
    }

    /// Creates `PhysicalSides<BorderStyleColor>` from a `Border` struct, resolving colors.
    ///
    /// # Arguments
    /// * `border` - The `Border` struct containing border properties.
    /// * `current_color` - The current inherited color to resolve `currentColor` against.
    ///
    /// Post-condition: A `PhysicalSides<BorderStyleColor>` representing the resolved
    /// style and color for each physical border side is returned.
    pub(crate) fn from_border(
        border: &Border,
        current_color: &AbsoluteColor,
    ) -> PhysicalSides<Self> {
        let resolve = |color: &Color| color.resolve_to_absolute(current_color);
        PhysicalSides::<Self>::new(
            Self::new(border.border_top_style, resolve(&border.border_top_color)),
            Self::new(
                border.border_right_style,
                resolve(&border.border_right_color),
            ),
            Self::new(
                border.border_bottom_style,
                resolve(&border.border_bottom_color),
            ),
            Self::new(border.border_left_style, resolve(&border.border_left_color)),
        )
    }

    /// Returns a `BorderStyleColor` representing a hidden border.
    ///
    /// Post-condition: A `BorderStyleColor` with `BorderStyle::Hidden` and `AbsoluteColor::TRANSPARENT_BLACK` is returned.
    pub(crate) fn hidden() -> Self {
        Self::new(BorderStyle::Hidden, AbsoluteColor::TRANSPARENT_BLACK)
    }
}

impl Default for BorderStyleColor {
    /// Returns the default `BorderStyleColor` (style `None`, color `TRANSPARENT_BLACK`).
    fn default() -> Self {
        Self::new(BorderStyle::None, AbsoluteColor::TRANSPARENT_BLACK)
    }
}

/// `ComputedValuesExt` provides extension methods for Stylo's `ComputedValues`
/// to extract layout-relevant information.
pub(crate) trait ComputedValuesExt {
    /// Returns the physical box offsets (top, right, bottom, left) as `LengthPercentageOrAuto`.
    fn physical_box_offsets(&self) -> PhysicalSides<LengthPercentageOrAuto<'_>>;
    /// Returns the logical box offsets (block-start, inline-end, block-end, inline-start) as `LengthPercentageOrAuto`.
    fn box_offsets(&self, writing_mode: WritingMode) -> LogicalSides<LengthPercentageOrAuto<'_>>;
    /// Returns the specified box size (width and height) in logical dimensions.
    fn box_size(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalVec2<Size<LengthPercentage>>;
    /// Returns the specified minimum box size (min-width and min-height) in logical dimensions.
    fn min_box_size(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalVec2<Size<LengthPercentage>>;
    /// Returns the specified maximum box size (max-width and max-height) in logical dimensions.
    fn max_box_size(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalVec2<Size<LengthPercentage>>;
    /// Computes the content box size given an outer box size and PBM.
    fn content_box_size_for_box_size(
        &self,
        box_size: LogicalVec2<Size<Au>>,
        pbm: &PaddingBorderMargin,
    ) -> LogicalVec2<Size<Au>>;
    /// Computes the content min box size given an outer min box size and PBM.
    fn content_min_box_size_for_min_size(
        &self,
        box_size: LogicalVec2<Size<Au>>,
        pbm: &PaddingBorderMargin,
    ) -> LogicalVec2<Size<Au>>;
    /// Computes the content max box size given an outer max box size and PBM.
    fn content_max_box_size_for_max_size(
        &self,
        box_size: LogicalVec2<Size<Au>>,
        pbm: &PaddingBorderMargin,
    ) -> LogicalVec2<Size<Au>>;
    /// Returns the resolved border style and color for each logical side.
    fn border_style_color(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalSides<BorderStyleColor>;
    /// Returns the physical margin values as `LengthPercentageOrAuto`.
    fn physical_margin(&self) -> PhysicalSides<LengthPercentageOrAuto<'_>>;
    /// Returns the logical margin values as `LengthPercentageOrAuto`.
    fn margin(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalSides<LengthPercentageOrAuto<'_>>;
    /// Checks if the element is transformable according to CSS Transforms specification.
    fn is_transformable(&self, fragment_flags: FragmentFlags) -> bool;
    /// Checks if the element has an active transform or perspective property.
    fn has_transform_or_perspective(&self, fragment_flags: FragmentFlags) -> bool;
    /// Checks if the `z-index` property applies to this element.
    fn z_index_applies(&self, fragment_flags: FragmentFlags) -> bool;
    /// Returns the effective `z-index` of the element.
    fn effective_z_index(&self, fragment_flags: FragmentFlags) -> i32;
    /// Returns the effective `overflow` property for both axes.
    fn effective_overflow(&self, fragment_flags: FragmentFlags) -> AxesOverflow;
    /// Checks if the element establishes a new block formatting context.
    fn establishes_block_formatting_context(&self, fragment_flags: FragmentFlags) -> bool;
    /// Checks if the element establishes a scroll container.
    fn establishes_scroll_container(&self, fragment_flags: FragmentFlags) -> bool;
    /// Checks if the element establishes a stacking context.
    fn establishes_stacking_context(&self, fragment_flags: FragmentFlags) -> bool;
    /// Checks if the element establishes a containing block for absolutely positioned descendants.
    fn establishes_containing_block_for_absolute_descendants(
        &self,
        fragment_flags: FragmentFlags,
    ) -> bool;
    /// Checks if the element establishes a containing block for all descendants (including fixed).
    fn establishes_containing_block_for_all_descendants(
        &self,
        fragment_flags: FragmentFlags,
    ) -> bool;
    /// Resolves the preferred aspect ratio for the element.
    fn preferred_aspect_ratio(
        &self,
        natural_aspect_ratio: Option<CSSFloat>,
        padding_border_sums: &LogicalVec2<Au>,
    ) -> Option<AspectRatio>;
    /// Checks if the background of the element is transparent.
    fn background_is_transparent(&self) -> bool;
    /// Generates appropriate WebRender `PrimitiveFlags` based on the style.
    fn get_webrender_primitive_flags(&self) -> wr::PrimitiveFlags;
    /// Returns bidi control characters to inject for the element's text content.
    fn bidi_control_chars(&self) -> (&'static str, &'static str);
    /// Resolves the `align-self` property based on auto and normal values.
    fn resolve_align_self(
        &self,
        resolved_auto_value: AlignItems,
        resolved_normal_value: AlignItems,
    ) -> AlignItems;
    /// Checks if the element's positioning depends on block constraints due to relative/sticky positioning.
    fn depends_on_block_constraints_due_to_relative_positioning(
        &self,
        writing_mode: WritingMode,
    ) -> bool;
    /// Checks if the element is an inline box.
    fn is_inline_box(&self, fragment_flags: FragmentFlags) -> bool;
}

impl ComputedValuesExt for ComputedValues {
    /// Returns the physical box offsets (top, right, bottom, left) as `LengthPercentageOrAuto`.
    fn physical_box_offsets(&self) -> PhysicalSides<LengthPercentageOrAuto<'_>> {
        fn convert(inset: &Inset) -> LengthPercentageOrAuto<'_> {
            match inset {
                Inset::LengthPercentage(v) => LengthPercentageOrAuto::LengthPercentage(v),
                Inset::Auto => LengthPercentageOrAuto::Auto,
                Inset::AnchorFunction(_) => unreachable!("anchor() should be disabled"),
                Inset::AnchorSizeFunction(_) => unreachable!("anchor-size() should be disabled"),
            }
        }
        let position = self.get_position();
        PhysicalSides::new(
            convert(&position.top),
            convert(&position.right),
            convert(&position.bottom),
            convert(&position.left),
        )
    }

    /// Returns the logical box offsets (block-start, inline-end, block-end, inline-start) as `LengthPercentageOrAuto`.
    fn box_offsets(&self, writing_mode: WritingMode) -> LogicalSides<LengthPercentageOrAuto<'_>> {
        LogicalSides::from_physical(&self.physical_box_offsets(), writing_mode)
    }

    /// Returns the specified box size (width and height) in logical dimensions.
    fn box_size(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalVec2<Size<LengthPercentage>> {
        let position = self.get_position();
        LogicalVec2::from_physical_size(
            &PhysicalSize::new(
                position.clone_width().into(),
                position.clone_height().into(),
            ),
            containing_block_writing_mode,
        )
    }

    /// Returns the specified minimum box size (min-width and min-height) in logical dimensions.
    fn min_box_size(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalVec2<Size<LengthPercentage>> {
        let position = self.get_position();
        LogicalVec2::from_physical_size(
            &PhysicalSize::new(
                position.clone_min_width().into(),
                position.clone_min_height().into(),
            ),
            containing_block_writing_mode,
        )
    }

    /// Returns the specified maximum box size (max-width and max-height) in logical dimensions.
    fn max_box_size(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalVec2<Size<LengthPercentage>> {
        let position = self.get_position();
        LogicalVec2::from_physical_size(
            &PhysicalSize::new(
                position.clone_max_width().into(),
                position.clone_max_height().into(),
            ),
            containing_block_writing_mode,
        )
    }

    /// Computes the content box size given an outer box size and `PaddingBorderMargin`.
    ///
    /// # Arguments
    /// * `box_size` - The outer box size.
    /// * `pbm` - The padding, border, and margin values.
    ///
    /// Post-condition: A `LogicalVec2<Size<Au>>` representing the content box size is returned.
    fn content_box_size_for_box_size(
        &self,
        box_size: LogicalVec2<Size<Au>>,
        pbm: &PaddingBorderMargin,
    ) -> LogicalVec2<Size<Au>> {
        match self.get_position().box_sizing {
            BoxSizing::ContentBox => box_size,
            // These may be negative, but will later be clamped by `min-width`/`min-height`
            // which is clamped to zero.
            BoxSizing::BorderBox => box_size.map_inline_and_block_sizes(
                |value| value - pbm.padding_border_sums.inline,
                |value| value - pbm.padding_border_sums.block,
            ),
        }
    }

    /// Computes the content min box size given an outer min box size and `PaddingBorderMargin`.
    ///
    /// # Arguments
    /// * `min_box_size` - The outer min box size.
    /// * `pbm` - The padding, border, and margin values.
    ///
    /// Post-condition: A `LogicalVec2<Size<Au>>` representing the content min box size is returned.
    fn content_min_box_size_for_min_size(
        &self,
        min_box_size: LogicalVec2<Size<Au>>,
        pbm: &PaddingBorderMargin,
    ) -> LogicalVec2<Size<Au>> {
        match self.get_position().box_sizing {
            BoxSizing::ContentBox => min_box_size,
            // Clamp to zero to make sure the used size components are non-negative
            BoxSizing::BorderBox => min_box_size.map_inline_and_block_sizes(
                |value| Au::zero().max(value - pbm.padding_border_sums.inline),
                |value| Au::zero().max(value - pbm.padding_border_sums.block),
            ),
        }
    }

    /// Computes the content max box size given an outer max box size and `PaddingBorderMargin`.
    ///
    /// # Arguments
    /// * `max_box_size` - The outer max box size.
    /// * `pbm` - The padding, border, and margin values.
    ///
    /// Post-condition: A `LogicalVec2<Size<Au>>` representing the content max box size is returned.
    fn content_max_box_size_for_max_size(
        &self,
        max_box_size: LogicalVec2<Size<Au>>,
        pbm: &PaddingBorderMargin,
    ) -> LogicalVec2<Size<Au>> {
        match self.get_position().box_sizing {
            BoxSizing::ContentBox => max_box_size,
            // This may be negative, but will later be clamped by `min-width`
            // which itself is clamped to zero.
            BoxSizing::BorderBox => max_box_size.map_inline_and_block_sizes(
                |value| value - pbm.padding_border_sums.inline,
                |value| value - pbm.padding_border_sums.block,
            ),
        }
    }

    /// Returns the resolved border style and color for each logical side.
    ///
    /// # Arguments
    /// * `containing_block_writing_mode` - The writing mode of the containing block.
    ///
    /// Post-condition: A `LogicalSides<BorderStyleColor>` is returned.
    fn border_style_color(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalSides<BorderStyleColor> {
        let current_color = self.get_inherited_text().clone_color();
        LogicalSides::from_physical(
            &BorderStyleColor::from_border(self.get_border(), &current_color),
            containing_block_writing_mode,
        )
    }

    /// Returns the physical margin values as `LengthPercentageOrAuto`.
    fn physical_margin(&self) -> PhysicalSides<LengthPercentageOrAuto<'_>> {
        fn convert(inset: &Margin) -> LengthPercentageOrAuto<'_> {
            match inset {
                Margin::LengthPercentage(v) => LengthPercentageOrAuto::LengthPercentage(v),
                Margin::Auto => LengthPercentageOrAuto::Auto,
                Margin::AnchorSizeFunction(_) => unreachable!("anchor-size() should be disabled"),
            }
        }
        let margin = self.get_margin();
        PhysicalSides::new(
            convert(&margin.margin_top),
            convert(&margin.margin_right),
            convert(&margin.margin_bottom),
            convert(&margin.margin_left),
        )
    }

    /// Returns the logical margin values as `LengthPercentageOrAuto`.
    ///
    /// # Arguments
    /// * `containing_block_writing_mode` - The writing mode of the containing block.
    ///
    /// Post-condition: A `LogicalSides<LengthPercentageOrAuto>` is returned.
    fn margin(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalSides<LengthPercentageOrAuto<'_>> {
        LogicalSides::from_physical(&self.physical_margin(), containing_block_writing_mode)
    }

    /// Checks if the element is an inline box.
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: Returns `true` if the display is `inline-flow` and it's not a
    /// replaced element or text control, `false` otherwise.
    fn is_inline_box(&self, fragment_flags: FragmentFlags) -> bool {
        self.get_box().display.is_inline_flow() &&
            !fragment_flags
                .intersects(FragmentFlags::IS_REPLACED | FragmentFlags::IS_TEXT_CONTROL)
    }

    /// Returns true if this is a transformable element.
    ///
    /// Post-condition: A boolean indicating if the element is transformable is returned.
    /// @see <https://drafts.csswg.org/css-transforms/#transformable-element>
    fn is_transformable(&self, fragment_flags: FragmentFlags) -> bool {
        // "A transformable element is an element in one of these categories:
        //   * all elements whose layout is governed by the CSS box model except for
        //     non-replaced inline boxes, table-column boxes, and table-column-group
        //     boxes,
        //   * all SVG paint server elements, the clipPath element  and SVG renderable
        //     elements with the exception of any descendant element of text content
        //     elements."
        // <https://drafts.csswg.org/css-transforms/#transformable-element>
        // TODO: check for all cases listed in the above spec.
        !self.is_inline_box(fragment_flags)
    }

    /// Returns true if this style has a transform, or perspective property set and
    /// it applies to this element.
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: A boolean indicating if the element has active transform or perspective properties.
    fn has_transform_or_perspective(&self, fragment_flags: FragmentFlags) -> bool {
        self.is_transformable(fragment_flags) &&
            (!self.get_box().transform.0.is_empty() ||
                self.get_box().scale != GenericScale::None ||
                self.get_box().rotate != GenericRotate::None ||
                self.get_box().translate != GenericTranslate::None ||
                self.get_box().perspective != Perspective::None)
    }

    /// Whether the `z-index` property applies to this fragment.
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: Returns `true` if `z-index` applies (positioned, flex/grid item), `false` otherwise.
    fn z_index_applies(&self, fragment_flags: FragmentFlags) -> bool {
        // As per CSS 2 § 9.9.1, `z-index` applies to positioned elements.
        // <http://www.w3.org/TR/CSS2/visuren.html#z-index>
        if self.get_box().position != ComputedPosition::Static {
            return true;
        }
        // More modern specs also apply it to flex and grid items.
        // - From <https://www.w3.org/TR/css-flexbox-1/#painting>:
        //   > Flex items paint exactly the same as inline blocks [CSS2], except that order-modified
        //   > document order is used in place of raw document order, and z-index values other than auto
        //   > create a stacking context even if position is static (behaving exactly as if position
        //   > were relative).
        // - From <https://drafts.csswg.org/css-flexbox/#painting>:
        //   > The painting order of grid items is exactly the same as inline blocks [CSS2], except that
        //   > order-modified document order is used in place of raw document order, and z-index values
        //   > other than auto create a stacking context even if position is static (behaving exactly
        //   > as if position were relative).
        fragment_flags.contains(FragmentFlags::IS_FLEX_OR_GRID_ITEM)
    }

    /// Get the effective z-index of this fragment. Z-indices only apply to positioned elements
    /// per CSS 2 9.9.1 (<http://www.w3.org/TR/CSS2/visuren.html#z-index>), so this value may differ
    /// from the value specified in the style.
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: The effective `z-index` as an `i32` is returned.
    fn effective_z_index(&self, fragment_flags: FragmentFlags) -> i32 {
        if self.z_index_applies(fragment_flags) {
            self.get_position().z_index.integer_or(0)
        } else {
            0
        }
    }

    /// Get the effective overflow of this box. The property only applies to block containers,
    /// flex containers, and grid containers. And some box types only accept a few values.
    /// <https://www.w3.org/TR/css-overflow-3/#overflow-control>
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: An `AxesOverflow` struct is returned, specifying the effective overflow
    /// for both X and Y axes, considering replaced elements and table types.
    fn effective_overflow(&self, fragment_flags: FragmentFlags) -> AxesOverflow {
        let style_box = self.get_box();
        let mut overflow_x = style_box.overflow_x;
        let mut overflow_y = style_box.overflow_y;

        // From <https://www.w3.org/TR/css-overflow-4/#overflow-control>:
        // "On replaced elements, the used values of all computed values other than visible is clip."
        // Block Logic: Replaced elements always clip if overflow is not visible.
        if fragment_flags.contains(FragmentFlags::IS_REPLACED) {
            if overflow_x != Overflow::Visible {
                overflow_x = Overflow::Clip;
            }
            if overflow_y != Overflow::Visible {
                overflow_y = Overflow::Clip;
            }
            return AxesOverflow {
                x: overflow_x,
                y: overflow_y,
            };
        }

        // Block Logic: Determines if overflow should be ignored based on display type.
        let ignores_overflow = match style_box.display.inside() {
            stylo::DisplayInside::Table => {
                // According to <https://drafts.csswg.org/css-tables/#global-style-overrides>,
                // - overflow applies to table-wrapper boxes and not to table grid boxes.
                //   That's what Blink and WebKit do, however Firefox matches a CSSWG resolution that says
                //   the opposite: <https://lists.w3.org/Archives/Public/www-style/2012Aug/0298.html>
                //   Due to the way that we implement table-wrapper boxes, it's easier to align with Firefox.
                // - Tables ignore overflow values different than visible, clip and hidden.
                //   This affects both axes, to ensure they have the same scrollability.
                !matches!(self.pseudo(), Some(PseudoElement::ServoTableGrid)) ||
                    matches!(overflow_x, Overflow::Auto | Overflow::Scroll) ||
                    matches!(overflow_y, Overflow::Auto | Overflow::Scroll)
            },
            stylo::DisplayInside::TableColumn |
            stylo::DisplayInside::TableColumnGroup |
            stylo::DisplayInside::TableRow |
            stylo::DisplayInside::TableRowGroup |
            stylo::DisplayInside::TableHeaderGroup |
            stylo::DisplayInside::TableFooterGroup => {
                // <https://drafts.csswg.org/css-tables/#global-style-overrides>
                // Table-track and table-track-group boxes ignore overflow.
                true
            },
            _ => false,
        };

        // Block Logic: Returns visible overflow if ignored, otherwise computed overflow.
        if ignores_overflow {
            AxesOverflow {
                x: Overflow::Visible,
                y: Overflow::Visible,
            }
        } else {
            AxesOverflow {
                x: overflow_x,
                y: overflow_y,
            }
        }
    }

    /// Return true if this style is a normal block and establishes
    /// a new block formatting context.
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: Returns `true` if a new block formatting context is established, `false` otherwise.
    fn establishes_block_formatting_context(&self, fragment_flags: FragmentFlags) -> bool {
        if self.establishes_scroll_container(fragment_flags) {
            return true;
        }

        if self.get_column().is_multicol() {
            return true;
        }

        if self.get_column().column_span == ColumnSpan::All {
            return true;
        }

        // Per <https://drafts.csswg.org/css-align/#distribution-block>:
        // Block containers with an `align-content` value that is not `normal` should
        // form an independent block formatting context. This should really only happen
        // for block containers, but we do not support subgrid containers yet which is the
        // only other case.
        // Block Logic: Checks `align-content` for block formatting context establishment.
        if self.get_position().align_content.0.primary() != AlignFlags::NORMAL {
            return true;
        }

        // TODO: We need to handle CSS Contain here.
        false
    }

    /// Whether or not the `overflow` value of this style establishes a scroll container.
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: Returns `true` if the effective overflow is scrollable, `false` otherwise.
    fn establishes_scroll_container(&self, fragment_flags: FragmentFlags) -> bool {
        // Checking one axis suffices, because the computed value ensures that
        // either both axes are scrollable, or none is scrollable.
        self.effective_overflow(fragment_flags).x.is_scrollable()
    }

    /// Returns true if this fragment establishes a new stacking context and false otherwise.
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: Returns `true` if a new stacking context is established, `false` otherwise.
    fn establishes_stacking_context(&self, fragment_flags: FragmentFlags) -> bool {
        // From <https://www.w3.org/TR/css-will-change/#valdef-will-change-custom-ident>:
        // > If any non-initial value of a property would create a stacking context on the element,
        // > specifying that property in will-change must create a stacking context on the element.
        let will_change_bits = self.clone_will_change().bits;
        // Block Logic: Checks for `will-change` properties that create a stacking context.
        if will_change_bits
            .intersects(WillChangeBits::STACKING_CONTEXT_UNCONDITIONAL | WillChangeBits::OPACITY)
        {
            return true;
        }

        // From <https://www.w3.org/TR/CSS2/visuren.html#z-index>, values different than `auto`
        // make the box establish a stacking context.
        // Block Logic: `z-index` values other than `auto` create a stacking context.
        if self.z_index_applies(fragment_flags) &&
            (!self.get_position().z_index.is_auto() ||
                will_change_bits.intersects(WillChangeBits::Z_INDEX))
        {
            return true;
        }

        // Fixed position and sticky position always create stacking contexts.
        // Note `will-change: position` is handled above by `STACKING_CONTEXT_UNCONDITIONAL`.
        // Block Logic: Fixed and sticky positions always create stacking contexts.
        if matches!(
            self.get_box().position,
            ComputedPosition::Fixed | ComputedPosition::Sticky
        ) {
            return true;
        }

        // From <https://www.w3.org/TR/css-transforms-1/#transform-rendering>
        // > For elements whose layout is governed by the CSS box model, any value other than
        // > `none` for the `transform` property results in the creation of a stacking context.
        // From <https://www.w3.org/TR/css-transforms-2/#transform-style-property>
        // > A computed value of `preserve-3d` for `transform-style` on a transformable element
        // > establishes both a stacking context and a containing block for all descendants.
        // From <https://www.w3.org/TR/css-transforms-2/#perspective-property>
        // > any value other than none establishes a stacking context.
        // TODO: handle individual transform properties (`translate`, `scale` and `rotate`).
        // <https://www.w3.org/TR/css-transforms-2/#individual-transforms>
        // Block Logic: Checks for transform, transform-style, perspective, and related `will-change` properties.
        if self.is_transformable(fragment_flags) &&
            (!self.get_box().transform.0.is_empty() ||
                self.get_box().transform_style == ComputedTransformStyle::Preserve3d ||
                self.get_box().perspective != Perspective::None ||
                will_change_bits
                    .intersects(WillChangeBits::TRANSFORM | WillChangeBits::PERSPECTIVE))
        {
            return true;
        }

        // From <https://www.w3.org/TR/css-color-3/#transparency>
        // > implementations must create a new stacking context for any element with opacity less than 1.
        // Note `will-change: opacity` is handled above by `WillChangeBits::OPACITY`.
        // Block Logic: Opacity less than 1 creates a stacking context.
        let effects = self.get_effects();
        if effects.opacity != 1.0 {
            return true;
        }

        // From <https://www.w3.org/TR/filter-effects-1/#FilterProperty>
        // > A computed value of other than `none` results in the creation of a stacking context
        // Note `will-change: filter` is handled above by `STACKING_CONTEXT_UNCONDITIONAL`.
        // Block Logic: Non-empty filter creates a stacking context.
        if !effects.filter.0.is_empty() {
            return true;
        }

        // From <https://www.w3.org/TR/compositing-1/#mix-blend-mode>
        // > Applying a blendmode other than `normal` to the element must establish a new stacking context
        // Note `will-change: mix-blend-mode` is handled above by `STACKING_CONTEXT_UNCONDITIONAL`.
        // Block Logic: Non-normal mix-blend-mode creates a stacking context.
        if effects.mix_blend_mode != ComputedMixBlendMode::Normal {
            return true;
        }

        // From <https://www.w3.org/TR/css-masking-1/#the-clip-path>
        // > A computed value of other than `none` results in the creation of a stacking context.
        // Note `will-change: clip-path` is handled above by `STACKING_CONTEXT_UNCONDITIONAL`.
        // Block Logic: Non-none clip-path creates a stacking context.
        if self.get_svg().clip_path != ClipPath::None {
            return true;
        }

        // From <https://www.w3.org/TR/compositing-1/#isolation>
        // > For CSS, setting `isolation` to `isolate` will turn the element into a stacking context.
        // Note `will-change: isolation` is handled above by `STACKING_CONTEXT_UNCONDITIONAL`.
        // Block Logic: `isolation: isolate` creates a stacking context.
        if self.get_box().isolation == ComputedIsolation::Isolate {
            return true;
        }

        // TODO: We need to handle CSS Contain here.
        false
    }

    /// Returns true if this style establishes a containing block for absolute
    /// descendants (`position: absolute`). If this style happens to establish a
    /// containing block for “all descendants” (ie including `position: fixed`
    /// descendants) this method will return true, but a true return value does
    /// not imply that the style establishes a containing block for all descendants.
    /// Use `establishes_containing_block_for_all_descendants()` instead.
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: Returns `true` if a containing block for absolute descendants is established, `false` otherwise.
    fn establishes_containing_block_for_absolute_descendants(
        &self,
        fragment_flags: FragmentFlags,
    ) -> bool {
        if self.establishes_containing_block_for_all_descendants(fragment_flags) {
            return true;
        }

        // From <https://www.w3.org/TR/css-will-change/#valdef-will-change-custom-ident>:
        // > If any non-initial value of a property would cause the element to
        // > generate a containing block for absolutely positioned elements, specifying that property in
        // > will-change must cause the element to generate a containing block for absolutely positioned elements.
        // Block Logic: `will-change: position` creates a containing block for absolute descendants.
        if self
            .clone_will_change()
            .bits
            .intersects(WillChangeBits::POSITION)
        {
            return true;
        }

        self.clone_position() != ComputedPosition::Static
    }

    /// Returns true if this style establishes a containing block for
    /// all descendants, including fixed descendants (`position: fixed`).
    /// Note that this also implies that it establishes a containing block
    /// for absolute descendants (`position: absolute`).
    ///
    /// # Arguments
    /// * `fragment_flags` - Flags associated with the fragment.
    ///
    /// Post-condition: Returns `true` if a containing block for all descendants is established, `false` otherwise.
    fn establishes_containing_block_for_all_descendants(
        &self,
        fragment_flags: FragmentFlags,
    ) -> bool {
        if self.has_transform_or_perspective(fragment_flags) {
            return true;
        }

        if !self.get_effects().filter.0.is_empty() {
            return true;
        }

        // See <https://drafts.csswg.org/css-transforms-2/#transform-style-property>.
        // Block Logic: `transform-style: preserve-3d` on a transformable element creates a containing block for all descendants.
        if self.is_transformable(fragment_flags) &&
            self.get_box().transform_style == ComputedTransformStyle::Preserve3d
        {
            return true;
        }
        // From <https://www.w3.org/TR/css-will-change/#valdef-will-change-custom-ident>:
        // > If any non-initial value of a property would cause the element to generate a
        // > containing block for fixed positioned elements, specifying that property in will-change
        // > must cause the element to generate a containing block for fixed positioned elements.
        let will_change_bits = self.clone_will_change().bits;
        // Block Logic: Checks `will-change` properties that create a containing block for fixed descendants.
        if will_change_bits.intersects(WillChangeBits::FIXPOS_CB_NON_SVG) ||
            (will_change_bits
                .intersects(WillChangeBits::TRANSFORM | WillChangeBits::PERSPECTIVE) &&
                self.is_transformable(fragment_flags))
        {
            return true;
        }

        // TODO: We need to handle CSS Contain here.
        false
    }

    /// Resolve the preferred aspect ratio according to the given natural aspect
    /// ratio and the `aspect-ratio` property.
    /// @see <https://drafts.csswg.org/css-sizing-4/#aspect-ratio>
    ///
    /// # Arguments
    /// * `natural_aspect_ratio` - The natural aspect ratio of the element, if any.
    /// * `padding_border_sums` - The sum of padding and border for the element.
    ///
    /// Post-condition: An `Option<AspectRatio>` is returned, representing the resolved preferred aspect ratio.
    fn preferred_aspect_ratio(
        &self,
        natural_aspect_ratio: Option<CSSFloat>,
        padding_border_sums: &LogicalVec2<Au>,
    ) -> Option<AspectRatio> {
        let GenericAspectRatio {
            auto,
            ratio: mut preferred_ratio,
        } = self.clone_aspect_ratio();

        // For all cases where a ratio is specified:
        // "If the <ratio> is degenerate, the property instead behaves as auto."
        // Block Logic: Degenerate ratios (e.g., 0/0) behave as `auto`.
        if matches!(preferred_ratio, PreferredRatio::Ratio(ratio) if ratio.is_degenerate()) {
            preferred_ratio = PreferredRatio::None;
        }

        // Block Logic: Resolves aspect ratio based on `auto`, specified ratio, and `box-sizing`.
        match (auto, preferred_ratio) {
            // The value `auto`. Either the ratio was not specified, or was
            // degenerate and set to PreferredRatio::None above.
            //
            // "Replaced elements with a natural aspect ratio use that aspect
            // ratio; otherwise the box has no preferred aspect ratio. Size
            // calculations involving the aspect ratio work with the content box
            // dimensions always."
            (_, PreferredRatio::None) => natural_aspect_ratio.map(AspectRatio::from_content_ratio),
            // "If both auto and a <ratio> are specified together, the preferred
            // aspect ratio is the specified ratio of width / height unless it
            // is a replaced element with a natural aspect ratio, in which case
            // that aspect ratio is used instead. In all cases, size
            // calculations involving the aspect ratio work with the content box
            // dimensions always."
            (true, PreferredRatio::Ratio(preferred_ratio)) => {
                Some(AspectRatio::from_content_ratio(
                    natural_aspect_ratio
                        .unwrap_or_else(|| (preferred_ratio.0).0 / (preferred_ratio.1).0),
                ))
            },

            // "The box’s preferred aspect ratio is the specified ratio of width
            // / height. Size calculations involving the aspect ratio work with
            // the dimensions of the box specified by box-sizing."
            (false, PreferredRatio::Ratio(preferred_ratio)) => {
                // If the `box-sizing` is `border-box`, use the padding and
                // border when calculating the aspect ratio.
                // Block Logic: Adjusts aspect ratio calculation based on `box-sizing` for `border-box`.
                let box_sizing_adjustment = match self.clone_box_sizing() {
                    BoxSizing::ContentBox => LogicalVec2::zero(),
                    BoxSizing::BorderBox => *padding_border_sums,
                };
                Some(AspectRatio {
                    i_over_b: (preferred_ratio.0).0 / (preferred_ratio.1).0,
                    box_sizing_adjustment,
                })
            },
        }
    }

    /// Whether or not this style specifies a non-transparent background.
    ///
    /// Post-condition: Returns `true` if the background color is transparent and no
    /// background images are present, `false` otherwise.
    fn background_is_transparent(&self) -> bool {
        let background = self.get_background();
        let color = self.resolve_color(&background.background_color);
        color.alpha == 0.0 &&
            background
                .background_image
                .0
                .iter()
                .all(|layer| matches!(layer, ComputedImageLayer::None))
    }

    /// Generate appropriate WebRender `PrimitiveFlags` that should be used
    /// for display items generated by the `Fragment` which owns this style.
    ///
    /// Post-condition: WebRender `PrimitiveFlags` are returned based on `backface-visibility`.
    fn get_webrender_primitive_flags(&self) -> wr::PrimitiveFlags {
        match self.get_box().backface_visibility {
            BackfaceVisiblity::Visible => wr::PrimitiveFlags::default(),
            BackfaceVisiblity::Hidden => wr::PrimitiveFlags::empty(),
        }
    }

    /// If the 'unicode-bidi' property has a value other than 'normal', return the bidi control codes
    /// to inject before and after the text content of the element.
    /// See the table in <http://dev.w3.org/csswg/css-writing-modes/#unicode-bidi>.
    ///
    /// Post-condition: A tuple `(&'static str, &'static str)` representing the bidi control
    /// characters (start and end) is returned.
    fn bidi_control_chars(&self) -> (&'static str, &'static str) {
        match (
            self.get_text().unicode_bidi,
            self.get_inherited_box().direction,
        ) {
            (UnicodeBidi::Normal, _) => ("", ""),
            (UnicodeBidi::Embed, Direction::Ltr) => ("\u{202a}", "\u{202c}"),
            (UnicodeBidi::Embed, Direction::Rtl) => ("\u{202b}", "\u{202c}"),
            (UnicodeBidi::Isolate, Direction::Ltr) => ("\u{2066}", "\u{2069}"),
            (UnicodeBidi::Isolate, Direction::Rtl) => ("\u{2067}", "\u{2069}"),
            (UnicodeBidi::BidiOverride, Direction::Ltr) => ("\u{202d}", "\u{202c}"),
            (UnicodeBidi::BidiOverride, Direction::Rtl) => ("\u{202e}", "\u{202c}"),
            (UnicodeBidi::IsolateOverride, Direction::Ltr) => {
                ("\u{2068}\u{202d}", "\u{202c}\u{2069}")
            },
            (UnicodeBidi::IsolateOverride, Direction::Rtl) => {
                ("\u{2068}\u{202e}", "\u{202c}\u{2069}")
            },
            (UnicodeBidi::Plaintext, _) => ("\u{2068}", "\u{2069}"),
        }
    }

    /// Resolves the `align-self` property, accounting for `auto` and `normal` values.
    ///
    /// # Arguments
    /// * `resolved_auto_value` - The value to use if `align-self` is `auto`.
    /// * `resolved_normal_value` - The value to use if `align-self` is `normal`.
    ///
    /// Post-condition: The resolved `AlignItems` value is returned.
    fn resolve_align_self(
        &self,
        resolved_auto_value: AlignItems,
        resolved_normal_value: AlignItems,
    ) -> AlignItems {
        match self.clone_align_self().0.0 {
            AlignFlags::AUTO => resolved_auto_value,
            AlignFlags::NORMAL => resolved_normal_value,
            value => AlignItems(value),
        }
    }

    /// Checks if the element's positioning depends on block constraints due to
    /// `position: relative` or `position: sticky` with percentage-based offsets.
    ///
    /// # Arguments
    /// * `writing_mode` - The writing mode of the element.
    ///
    /// Post-condition: Returns `true` if block constraints are depended upon due to relative/sticky positioning, `false` otherwise.
    fn depends_on_block_constraints_due_to_relative_positioning(
        &self,
        writing_mode: WritingMode,
    ) -> bool {
        // Block Logic: Only applies to `relative` or `sticky` positioned elements.
        if !matches!(
            self.get_box().position,
            ComputedPosition::Relative | ComputedPosition::Sticky
        ) {
            return false;
        }
        let box_offsets = self.box_offsets(writing_mode);
        // Functional Utility: Checks if a `LengthPercentageOrAuto` offset contains a percentage.
        let has_percentage = |offset: LengthPercentageOrAuto<'_>| {
            offset
                .non_auto()
                .is_some_and(LengthPercentage::has_percentage)
        };
        has_percentage(box_offsets.block_start) || has_percentage(box_offsets.block_end)
    }
}

/// `LayoutStyle` encapsulates the computed layout style of an element,
/// with a special variant for table elements.
pub(crate) enum LayoutStyle<'a> {
    /// Default layout style based on `ComputedValues`.
    Default(&'a ComputedValues),
    /// Layout style specifically for table elements, which may have unique properties.
    Table(TableLayoutStyle<'a>),
}

impl LayoutStyle<'_> {
    /// Returns a reference to the underlying `ComputedValues` of the layout style.
    #[inline]
    pub(crate) fn style(&self) -> &ComputedValues {
        match self {
            Self::Default(style) => style,
            Self::Table(table) => table.style(),
        }
    }

    /// Checks if the layout style corresponds to a table element.
    #[inline]
    pub(crate) fn is_table(&self) -> bool {
        matches!(self, Self::Table(_))
    }

    /// Computes the content box sizes along with padding, border, and margin (PBM) for an element.
    ///
    /// This function handles the resolution of various CSS sizing properties, including `min-width`,
    /// `max-width`, and `auto` values, taking into account cyclic percentage contributions and
    /// whether the element establishes a block formatting context.
    ///
    /// # Arguments
    /// * `containing_block` - Information about the element's containing block.
    ///
    /// Post-condition: A `ContentBoxSizesAndPBM` struct is returned, containing the computed
    /// content box sizes, PBM values, and dependency on block constraints.
    pub(crate) fn content_box_sizes_and_padding_border_margin(
        &self,
        containing_block: &IndefiniteContainingBlock,
    ) -> ContentBoxSizesAndPBM {
        // <https://drafts.csswg.org/css-sizing-3/#cyclic-percentage-contribution>
        // If max size properties or preferred size properties are set to a value containing
        // indefinite percentages, we treat the entire value as the initial value of the property.
        // However, for min size properties, as well as for margins and paddings,
        // we instead resolve indefinite percentages against zero.
        let containing_block_size_or_zero =
            containing_block.size.map(|value| value.unwrap_or_default());
        let writing_mode = containing_block.writing_mode;
        let pbm = self.padding_border_margin_with_writing_mode_and_containing_block_inline_size(
            writing_mode,
            containing_block_size_or_zero.inline,
        );
        let style = self.style();
        let box_size = style.box_size(writing_mode);
        let min_size = style.min_box_size(writing_mode);
        let max_size = style.max_box_size(writing_mode);
        let preferred_size_computes_to_auto = box_size.map(|size| size.is_initial());

        // Functional Utility: Checks if a `Size<LengthPercentage>` depends on block constraints.
        let depends_on_block_constraints = |size: &Size<LengthPercentage>| {
            match size {
                // fit-content is like clamp(min-content, stretch, max-content), but currently
                // min-content and max-content have the same behavior in the block axis,
                // so there is no dependency on block constraints.
                // TODO: for flex and grid layout, min-content and max-content should be different.
                // TODO: We are assuming that Size::Initial doesn't stretch. However, it may actually
                // stretch flex and grid items depending on the CSS Align properties, in that case
                // the caller needs to take care of it.
                Size::Stretch => true,
                Size::Numeric(length_percentage) => length_percentage.has_percentage(),
                _ => false,
            }
        };
        let depends_on_block_constraints = depends_on_block_constraints(&box_size.block) ||
            depends_on_block_constraints(&min_size.block) ||
            depends_on_block_constraints(&max_size.block) ||
            style.depends_on_block_constraints_due_to_relative_positioning(writing_mode);

        let box_size = box_size.map_with(&containing_block.size, |size, basis| {
            size.resolve_percentages_for_preferred(*basis)
        });
        let content_box_size = style.content_box_size_for_box_size(box_size, &pbm);
        let min_size = min_size.percentages_relative_to_basis(&containing_block_size_or_zero);
        let content_min_box_size = style.content_min_box_size_for_min_size(min_size, &pbm);
        let max_size = max_size.map_with(&containing_block.size, |size, basis| {
            size.resolve_percentages_for_max(*basis)
        });
        let content_max_box_size = style.content_max_box_size_for_max_size(max_size, &pbm);
        ContentBoxSizesAndPBM {
            content_box_sizes: LogicalVec2 {
                block: Sizes::new(
                    content_box_size.block,
                    content_min_box_size.block,
                    content_max_box_size.block,
                ),
                inline: Sizes::new(
                    content_box_size.inline,
                    content_min_box_size.inline,
                    content_max_box_size.inline,
                ),
            },
            pbm,
            depends_on_block_constraints,
            preferred_size_computes_to_auto,
        }
    }

    /// Computes padding, border, and margin (`PBM`) values given a `ContainingBlock`.
    ///
    /// # Arguments
    /// * `containing_block` - The `ContainingBlock` of the element.
    ///
    /// Post-condition: A `PaddingBorderMargin` struct is returned.
    pub(crate) fn padding_border_margin(
        &self,
        containing_block: &ContainingBlock,
    ) -> PaddingBorderMargin {
        self.padding_border_margin_with_writing_mode_and_containing_block_inline_size(
            containing_block.style.writing_mode,
            containing_block.size.inline,
        )
    }

    /// Computes padding, border, and margin (`PBM`) values given a writing mode and inline size.
    ///
    /// # Arguments
    /// * `writing_mode` - The `WritingMode` to use for logical side resolution.
    /// * `containing_block_inline_size` - The inline size of the containing block.
    ///
    /// Post-condition: A `PaddingBorderMargin` struct is returned with resolved values.
    pub(crate) fn padding_border_margin_with_writing_mode_and_containing_block_inline_size(
        &self,
        writing_mode: WritingMode,
        containing_block_inline_size: Au,
    ) -> PaddingBorderMargin {
        let padding = self
            .padding(writing_mode)
            .percentages_relative_to(containing_block_inline_size);
        let style = self.style();
        let border = self.border_width(writing_mode);
        let margin = style
            .margin(writing_mode)
            .percentages_relative_to(containing_block_inline_size);
        PaddingBorderMargin {
            padding_border_sums: LogicalVec2 {
                inline: padding.inline_sum() + border.inline_sum(),
                block: padding.block_sum() + border.block_sum(),
            },
            padding,
            border,
            margin,
        }
    }

    /// Computes the padding values in logical dimensions.
    ///
    /// # Arguments
    /// * `containing_block_writing_mode` - The writing mode of the containing block.
    ///
    /// Post-condition: A `LogicalSides<LengthPercentage>` representing the padding is returned.
    fn padding(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalSides<LengthPercentage> {
        // Block Logic: Table borders in collapsed mode have zero padding.
        if matches!(self, Self::Table(table) if table.collapses_borders()) {
            // https://drafts.csswg.org/css-tables/#collapsed-style-overrides
            // > The padding of the table-root is ignored (as if it was set to 0px).
            return LogicalSides::zero();
        }
        let padding = self.style().get_padding().clone();
        LogicalSides::from_physical(
            &PhysicalSides::new(
                padding.padding_top.0,
                padding.padding_right.0,
                padding.padding_bottom.0,
                padding.padding_left.0,
            ),
            containing_block_writing_mode,
        )
    }

    /// Computes the border width values in logical dimensions.
    ///
    /// # Arguments
    /// * `containing_block_writing_mode` - The writing mode of the containing block.
    ///
    /// Post-condition: A `LogicalSides<Au>` representing the border widths is returned.
    fn border_width(
        &self,
        containing_block_writing_mode: WritingMode,
    ) -> LogicalSides<Au> {
        let border_width = match self {
            // For tables in collapsed-borders mode we halve the border widths, because
            // > in this model, the width of the table includes half the table border.
            // https://www.w3.org/TR/CSS22/tables.html#collapsing-borders
            Self::Table(table) if table.collapses_borders() => table
                .halved_collapsed_border_widths()
                .to_physical(self.style().writing_mode),
            _ => {
                let border = self.style().get_border();
                PhysicalSides::new(
                    border.border_top_width,
                    border.border_right_width,
                    border.border_bottom_width,
                    border.border_left_width,
                )
            },
        };
        LogicalSides::from_physical(&border_width, containing_block_writing_mode)
    }
}

impl From<stylo::Display> for Display {
    /// Converts a Stylo `stylo::Display` enum into a Layout 2020 `Display` enum.
    ///
    /// This implementation handles the mapping of Stylo's packed display values
    /// to the more structured `Display` enum used for layout calculations,
    /// particularly for table-related internal display types.
    ///
    /// # Arguments
    /// * `packed` - The Stylo `stylo::Display` value to convert.
    ///
    /// Post-condition: A `Display` enum representing the converted display type is returned.
    fn from(packed: stylo::Display) -> Self {
        let outside = packed.outside();
        let inside = packed.inside();

        let outside = match outside {
            stylo::DisplayOutside::Block => DisplayOutside::Block,
            stylo::DisplayOutside::Inline => DisplayOutside::Inline,
            stylo::DisplayOutside::TableCaption => {
                return Display::GeneratingBox(DisplayGeneratingBox::LayoutInternal(
                    DisplayLayoutInternal::TableCaption,
                ));
            },
            stylo::DisplayOutside::InternalTable => {
                // Block Logic: Maps internal table display types to `DisplayLayoutInternal`.
                let internal = match inside {
                    stylo::DisplayInside::TableRowGroup => DisplayLayoutInternal::TableRowGroup,
                    stylo::DisplayInside::TableColumn => DisplayLayoutInternal::TableColumn,
                    stylo::DisplayInside::TableColumnGroup => {
                        DisplayLayoutInternal::TableColumnGroup
                    },
                    stylo::DisplayInside::TableHeaderGroup => {
                        DisplayLayoutInternal::TableHeaderGroup
                    },
                    stylo::DisplayInside::TableFooterGroup => {
                        DisplayLayoutInternal::TableFooterGroup
                    },
                    stylo::DisplayInside::TableRow => DisplayLayoutInternal::TableRow,
                    stylo::DisplayInside::TableCell => DisplayLayoutInternal::TableCell,
                    _ => unreachable!("Non-internal DisplayInside found"),
                };
                return Display::GeneratingBox(DisplayGeneratingBox::LayoutInternal(internal));
            },
            // This should not be a value of DisplayInside, but oh well
            // special-case display: contents because we still want it to work despite the early return
            stylo::DisplayOutside::None if inside == stylo::DisplayInside::Contents => {
                return Display::Contents;
            },
            stylo::DisplayOutside::None => return Display::None,
        };

        let inside = match packed.inside() {
            stylo::DisplayInside::Flow => DisplayInside::Flow {
                is_list_item: packed.is_list_item(),
            },
            stylo::DisplayInside::FlowRoot => DisplayInside::FlowRoot {
                is_list_item: packed.is_list_item(),
            },
            stylo::DisplayInside::Flex => DisplayInside::Flex,
            stylo::DisplayInside::Grid => DisplayInside::Grid,

            // These should not be values of DisplayInside, but oh well
            stylo::DisplayInside::None => return Display::None,
            stylo::DisplayInside::Contents => return Display::Contents,

            stylo::DisplayInside::Table => DisplayInside::Table,
            stylo::DisplayInside::TableRowGroup |
            stylo::DisplayInside::TableColumn |
            stylo::DisplayInside::TableColumnGroup |
            stylo::DisplayInside::TableHeaderGroup |
            stylo::DisplayInside::TableFooterGroup |
            stylo::DisplayInside::TableRow |
            stylo::DisplayInside::TableCell => unreachable!("Internal DisplayInside found"),
        };
        Display::GeneratingBox(DisplayGeneratingBox::OutsideInside { outside, inside })
    }
}

/// `Clamp` trait provides methods for clamping a value between extremums.
pub(crate) trait Clamp: Sized {
    /// Clamps `self` to be less than or equal to `max`, if `max` is `Some`.
    fn clamp_below_max(self, max: Option<Self>) -> Self;
    /// Clamps `self` to be between `min` and `max` (inclusive).
    fn clamp_between_extremums(self, min: Self, max: Option<Self>) -> Self;
}

impl Clamp for Au {
    /// Clamps `self` to be less than or equal to `max`, if `max` is `Some`.
    ///
    /// # Arguments
    /// * `max` - An `Option<Self>` representing the maximum value.
    ///
    /// Post-condition: Returns `self` or `max`, whichever is smaller.
    fn clamp_below_max(self, max: Option<Self>) -> Self {
        match max {
            None => self,
            Some(max) => self.min(max),
        }
    }

    /// Clamps `self` to be between `min` and `max` (inclusive).
    ///
    /// # Arguments
    /// * `min` - The minimum value.
    /// * `max` - An `Option<Self>` representing the maximum value.
    ///
    /// Post-condition: Returns `self` clamped within the specified range.
    fn clamp_between_extremums(self, min: Self, max: Option<Self>) -> Self {
        self.clamp_below_max(max).max(min)
    }
}

/// `TransformExt` trait provides extension methods for `LayoutTransform`.
pub(crate) trait TransformExt {
    /// Changes the basis of the transform by applying a translation before and after the transform.
    ///
    /// This is used to effectively shift the origin of the transformation.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` - The translation values for each axis.
    ///
    /// Post-condition: A new `LayoutTransform` with its basis changed is returned.
    /// @see <https://drafts.csswg.org/css-transforms/#transformation-matrix-computation>
    fn change_basis(&self, x: f32, y: f32, z: f32) -> Self;
}

impl TransformExt for LayoutTransform {
    /// <https://drafts.csswg.org/css-transforms/#transformation-matrix-computation>
    fn change_basis(&self, x: f32, y: f32, z: f32) -> Self {
        let pre_translation = Self::translation(x, y, z);
        let post_translation = Self::translation(-x, -y, -z);
        post_translation.then(self).then(&pre_translation)
    }
}
