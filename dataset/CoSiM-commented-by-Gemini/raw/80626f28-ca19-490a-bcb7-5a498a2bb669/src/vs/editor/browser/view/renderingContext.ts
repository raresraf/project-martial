/**
 * @file renderingContext.ts
 * @brief Defines the rendering context for the editor's view.
 * @details This file contains classes and interfaces that provide information about the
 * viewport, layout, and decorations, which are essential for rendering the editor's content.
 * It abstracts away the details of the view's layout and data, providing a clean API for
 * rendering components to use.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Position } from '../../common/core/position.js';
import { Range } from '../../common/core/range.js';
import { ViewportData } from '../../common/viewLayout/viewLinesViewportData.js';
import { IViewLayout } from '../../common/viewModel.js';
import { ViewModelDecoration } from '../../common/viewModel/viewModelDecoration.js';

/**
 * @interface IViewLines
 * @brief An interface for querying visible ranges of lines.
 */
export interface IViewLines {
	linesVisibleRangesForRange(range: Range, includeNewLines: boolean): LineVisibleRanges[] | null;
	visibleRangeForPosition(position: Position): HorizontalPosition | null;
}

/**
 * @class RestrictedRenderingContext
 * @brief A base class for the rendering context, providing restricted access to view layout information.
 */
export abstract class RestrictedRenderingContext {
	_restrictedRenderingContextBrand: void = undefined;

	public readonly viewportData: ViewportData;

	public readonly scrollWidth: number;
	public readonly scrollHeight: number;

	public readonly visibleRange: Range;
	public readonly bigNumbersDelta: number;

	public readonly scrollTop: number;
	public readonly scrollLeft: number;

	public readonly viewportWidth: number;
	public readonly viewportHeight: number;

	private readonly _viewLayout: IViewLayout;

	constructor(viewLayout: IViewLayout, viewportData: ViewportData) {
		this._viewLayout = viewLayout;
		this.viewportData = viewportData;

		this.scrollWidth = this._viewLayout.getScrollWidth();
		this.scrollHeight = this._viewLayout.getScrollHeight();

		this.visibleRange = this.viewportData.visibleRange;
		this.bigNumbersDelta = this.viewportData.bigNumbersDelta;

		const vInfo = this._viewLayout.getCurrentViewport();
		this.scrollTop = vInfo.top;
		this.scrollLeft = vInfo.left;
		this.viewportWidth = vInfo.width;
		this.viewportHeight = vInfo.height;
	}

	/**
	 * @brief Converts an absolute top position to a scrolled top position.
	 * @param absoluteTop The absolute top position.
	 * @returns The top position relative to the scroll top.
	 */
	public getScrolledTopFromAbsoluteTop(absoluteTop: number): number {
		return absoluteTop - this.scrollTop;
	}

	/**
	 * @brief Gets the vertical offset for a given line number.
	 * @param lineNumber The line number.
	 * @param includeViewZones Whether to include view zones in the calculation.
	 * @returns The vertical offset.
	 */
	public getVerticalOffsetForLineNumber(lineNumber: number, includeViewZones?: boolean): number {
		return this._viewLayout.getVerticalOffsetForLineNumber(lineNumber, includeViewZones);
	}

	/**
	 * @brief Gets the vertical offset after a given line number.
	 * @param lineNumber The line number.
	 * @param includeViewZones Whether to include view zones in the calculation.
	 * @returns The vertical offset after the line.
	 */
	public getVerticalOffsetAfterLineNumber(lineNumber: number, includeViewZones?: boolean): number {
		return this._viewLayout.getVerticalOffsetAfterLineNumber(lineNumber, includeViewZones);
	}

	/**
	 * @brief Gets the line height for a given line number.
	 * @param lineNumber The line number.
	 * @returns The line height.
	 */
	public getLineHeightForLineNumber(lineNumber: number): number {
		return this._viewLayout.getLineHeightForLineNumber(lineNumber);
	}

	/**
	 * @brief Gets the decorations in the current viewport.
	 * @returns An array of view model decorations.
	 */
	public getDecorationsInViewport(): ViewModelDecoration[] {
		return this.viewportData.getDecorationsInViewport();
	}

}

/**
 * @class RenderingContext
 * @brief The main rendering context, extending the restricted context with access to view lines.
 */
export class RenderingContext extends RestrictedRenderingContext {
	_renderingContextBrand: void = undefined;

	private readonly _viewLines: IViewLines;
	private readonly _viewLinesGpu?: IViewLines;

	constructor(viewLayout: IViewLayout, viewportData: ViewportData, viewLines: IViewLines, viewLinesGpu?: IViewLines) {
		super(viewLayout, viewportData);
		this._viewLines = viewLines;
		this._viewLinesGpu = viewLinesGpu;
	}

	/**
	 * @brief Gets the visible ranges for a given range of lines.
	 * @param range The range of lines.
	 * @param includeNewLines Whether to include new lines in the ranges.
	 * @returns An array of line visible ranges, or null.
	 */
	public linesVisibleRangesForRange(range: Range, includeNewLines: boolean): LineVisibleRanges[] | null {
		const domRanges = this._viewLines.linesVisibleRangesForRange(range, includeNewLines);
		if (!this._viewLinesGpu) {
			return domRanges ?? null;
		}
		const gpuRanges = this._viewLinesGpu.linesVisibleRangesForRange(range, includeNewLines);
		if (!domRanges) {
			return gpuRanges;
		}
		if (!gpuRanges) {
			return domRanges;
		}
		return domRanges.concat(gpuRanges).sort((a, b) => a.lineNumber - b.lineNumber);
	}

	/**
	 * @brief Gets the visible range for a given position.
	 * @param position The position.
	 * @returns A horizontal position, or null.
	 */
	public visibleRangeForPosition(position: Position): HorizontalPosition | null {
		return this._viewLines.visibleRangeForPosition(position) ?? this._viewLinesGpu?.visibleRangeForPosition(position) ?? null;
	}
}

/**
 * @class LineVisibleRanges
 * @brief Represents the visible horizontal ranges for a single line.
 */
export class LineVisibleRanges {
	/**
	 * @brief Returns the element with the smallest `lineNumber`.
	 * @param ranges An array of LineVisibleRanges.
	 * @returns The first LineVisibleRanges object, or null.
	 */
	public static firstLine(ranges: LineVisibleRanges[] | null): LineVisibleRanges | null {
		if (!ranges) {
			return null;
		}
		let result: LineVisibleRanges | null = null;
		for (const range of ranges) {
			if (!result || range.lineNumber < result.lineNumber) {
				result = range;
			}
		}
		return result;
	}

	/**
	 * @brief Returns the element with the largest `lineNumber`.
	 * @param ranges An array of LineVisibleRanges.
	 * @returns The last LineVisibleRanges object, or null.
	 */
	public static lastLine(ranges: LineVisibleRanges[] | null): LineVisibleRanges | null {
		if (!ranges) {
			return null;
		}
		let result: LineVisibleRanges | null = null;
		for (const range of ranges) {
			if (!result || range.lineNumber > result.lineNumber) {
				result = range;
			}
		}
		return result;
	}

	constructor(
		public readonly outsideRenderedLine: boolean,
		public readonly lineNumber: number,
		public readonly ranges: HorizontalRange[],
		/**
		 * Indicates if the requested range does not end in this line, but continues on the next line.
		 */
		public readonly continuesOnNextLine: boolean,
	) { }
}

/**
 * @class HorizontalRange
 * @brief Represents a horizontal range with a left position and a width.
 */
export class HorizontalRange {
	_horizontalRangeBrand: void = undefined;

	public left: number;
	public width: number;

	public static from(ranges: FloatHorizontalRange[]): HorizontalRange[] {
		const result = new Array(ranges.length);
		for (let i = 0, len = ranges.length; i < len; i++) {
			const range = ranges[i];
			result[i] = new HorizontalRange(range.left, range.width);
		}
		return result;
	}

	constructor(left: number, width: number) {
		this.left = Math.round(left);
		this.width = Math.round(width);
	}

	public toString(): string {
		return `[${this.left},${this.width}]`;
	}
}

/**
 * @class FloatHorizontalRange
 * @brief Represents a horizontal range with floating-point left position and width.
 */
export class FloatHorizontalRange {
	_floatHorizontalRangeBrand: void = undefined;

	public left: number;
	public width: number;

	constructor(left: number, width: number) {
		this.left = left;
		this.width = width;
	}

	public toString(): string {
		return `[${this.left},${this.width}]`;
	}

	public static compare(a: FloatHorizontalRange, b: FloatHorizontalRange): number {
		return a.left - b.left;
	}
}

/**
 * @class HorizontalPosition
 * @brief Represents a horizontal position.
 */
export class HorizontalPosition {
	public outsideRenderedLine: boolean;
	/**
	 * Math.round(this.originalLeft)
	 */
	public left: number;
	public originalLeft: number;

	constructor(outsideRenderedLine: boolean, left: number) {
		this.outsideRenderedLine = outsideRenderedLine;
		this.originalLeft = left;
		this.left = Math.round(this.originalLeft);
	}
}

/**
 * @class VisibleRanges
 * @brief Represents the visible ranges on a line.
 */
export class VisibleRanges {
	constructor(
		public readonly outsideRenderedLine: boolean,
		public readonly ranges: FloatHorizontalRange[]
	) {
	}
}
