/**
 * @file decorations.ts
 * @brief Implements the decorations overlay for the editor view.
 * @details This file contains the `DecorationsOverlay` class, which is responsible for rendering
 * decorations (like highlights, squiggly lines, etc.) on top of the text in the editor. It is
 * a dynamic view overlay that reacts to various editor events to efficiently update the
 * rendered decorations.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import './decorations.css';
import { DynamicViewOverlay } from '../../view/dynamicViewOverlay.js';
import { HorizontalRange, RenderingContext } from '../../view/renderingContext.js';
import { EditorOption } from '../../../common/config/editorOptions.js';
import { Range } from '../../../common/core/range.js';
import * as viewEvents from '../../../common/viewEvents.js';
import { ViewContext } from '../../../common/viewModel/viewContext.js';
import { ViewModelDecoration } from '../../../common/viewModel/viewModelDecoration.js';

/**
 * @class DecorationsOverlay
 * @brief A view overlay that renders editor decorations.
 */
export class DecorationsOverlay extends DynamicViewOverlay {

	private readonly _context: ViewContext;
	private _typicalHalfwidthCharacterWidth: number;
	private _renderResult: string[] | null;

	constructor(context: ViewContext) {
		super();
		this._context = context;
		const options = this._context.configuration.options;
		this._typicalHalfwidthCharacterWidth = options.get(EditorOption.fontInfo).typicalHalfwidthCharacterWidth;
		this._renderResult = null;

		this._context.addEventHandler(this);
	}

	public override dispose(): void {
		this._context.removeEventHandler(this);
		this._renderResult = null;
		super.dispose();
	}

	// --- begin event handlers
	/**
	 * @section Event Handlers
	 * @brief These methods are called in response to various editor events and determine
	 * whether the decorations need to be re-rendered.
	 */

	public override onConfigurationChanged(e: viewEvents.ViewConfigurationChangedEvent): boolean {
		const options = this._context.configuration.options;
		this._typicalHalfwidthCharacterWidth = options.get(EditorOption.fontInfo).typicalHalfwidthCharacterWidth;
		return true;
	}
	public override onDecorationsChanged(e: viewEvents.ViewDecorationsChangedEvent): boolean {
		return true;
	}
	public override onFlushed(e: viewEvents.ViewFlushedEvent): boolean {
		return true;
	}
	public override onLinesChanged(e: viewEvents.ViewLinesChangedEvent): boolean {
		return true;
	}
	public override onLinesDeleted(e: viewEvents.ViewLinesDeletedEvent): boolean {
		return true;
	}
	public override onLinesInserted(e: viewEvents.ViewLinesInsertedEvent): boolean {
		return true;
	}
	public override onScrollChanged(e: viewEvents.ViewScrollChangedEvent): boolean {
		return e.scrollTopChanged || e.scrollWidthChanged;
	}
	public override onZonesChanged(e: viewEvents.ViewZonesChangedEvent): boolean {
		return true;
	}
	// --- end event handlers

	/**
	 * @brief Prepares the decorations for rendering.
	 * @details This method is called before rendering. It gets the decorations in the
	 * viewport, sorts them, and generates the HTML for each line.
	 * @param ctx The rendering context.
	 */
	public prepareRender(ctx: RenderingContext): void {
		const _decorations = ctx.getDecorationsInViewport();

		// Keep only decorations with `className`
		let decorations: ViewModelDecoration[] = [];
		let decorationsLen = 0;
		for (let i = 0, len = _decorations.length; i < len; i++) {
			const d = _decorations[i];
			if (d.options.className) {
				decorations[decorationsLen++] = d;
			}
		}

		// Sort decorations for consistent render output
		decorations = decorations.sort((a, b) => {
			if (a.options.zIndex! < b.options.zIndex!) {
				return -1;
			}
			if (a.options.zIndex! > b.options.zIndex!) {
				return 1;
			}
			const aClassName = a.options.className!;
			const bClassName = b.options.className!;

			if (aClassName < bClassName) {
				return -1;
			}
			if (aClassName > bClassName) {
				return 1;
			}

			return Range.compareRangesUsingStarts(a.range, b.range);
		});

		const visibleStartLineNumber = ctx.visibleRange.startLineNumber;
		const visibleEndLineNumber = ctx.visibleRange.endLineNumber;
		const output: string[] = [];
		for (let lineNumber = visibleStartLineNumber; lineNumber <= visibleEndLineNumber; lineNumber++) {
			const lineIndex = lineNumber - visibleStartLineNumber;
			output[lineIndex] = '';
		}

		// Render first whole line decorations and then regular decorations
		this._renderWholeLineDecorations(ctx, decorations, output);
		this._renderNormalDecorations(ctx, decorations, output);
		this._renderResult = output;
	}

	/**
	 * @brief Renders decorations that span the entire line.
	 * @param ctx The rendering context.
	 * @param decorations The decorations to render.
	 * @param output The output array to which the HTML is written.
	 */
	private _renderWholeLineDecorations(ctx: RenderingContext, decorations: ViewModelDecoration[], output: string[]): void {
		const visibleStartLineNumber = ctx.visibleRange.startLineNumber;
		const visibleEndLineNumber = ctx.visibleRange.endLineNumber;

		for (let i = 0, lenI = decorations.length; i < lenI; i++) {
			const d = decorations[i];

			if (!d.options.isWholeLine) {
				continue;
			}

			const decorationOutput = (
				'<div class="cdr '
				+ d.options.className
				+ '" style="left:0;width:100%;"></div>'
			);

			const startLineNumber = Math.max(d.range.startLineNumber, visibleStartLineNumber);
			const endLineNumber = Math.min(d.range.endLineNumber, visibleEndLineNumber);
			for (let j = startLineNumber; j <= endLineNumber; j++) {
				const lineIndex = j - visibleStartLineNumber;
				output[lineIndex] += decorationOutput;
			}
		}
	}

	/**
	 * @brief Renders normal (in-line) decorations.
	 * @details This method optimizes rendering by merging decorations with the same class name
	 * that are adjacent.
	 * @param ctx The rendering context.
	 * @param decorations The decorations to render.
	 * @param output The output array to which the HTML is written.
	 */
	private _renderNormalDecorations(ctx: RenderingContext, decorations: ViewModelDecoration[], output: string[]): void {
		const visibleStartLineNumber = ctx.visibleRange.startLineNumber;

		let prevClassName: string | null = null;
		let prevShowIfCollapsed: boolean = false;
		let prevRange: Range | null = null;
		let prevShouldFillLineOnLineBreak: boolean = false;

		for (let i = 0, lenI = decorations.length; i < lenI; i++) {
			const d = decorations[i];

			if (d.options.isWholeLine) {
				continue;
			}

			const className = d.options.className!;
			const showIfCollapsed = Boolean(d.options.showIfCollapsed);

			let range = d.range;
			if (showIfCollapsed && range.endColumn === 1 && range.endLineNumber !== range.startLineNumber) {
				range = new Range(range.startLineNumber, range.startColumn, range.endLineNumber - 1, this._context.viewModel.getLineMaxColumn(range.endLineNumber - 1));
			}

			if (prevClassName === className && prevShowIfCollapsed === showIfCollapsed && Range.areIntersectingOrTouching(prevRange!, range)) {
				// merge into previous decoration
				prevRange = Range.plusRange(prevRange!, range);
				continue;
			}

			// flush previous decoration
			if (prevClassName !== null) {
				this._renderNormalDecoration(ctx, prevRange!, prevClassName, prevShouldFillLineOnLineBreak, prevShowIfCollapsed, visibleStartLineNumber, output);
			}

			prevClassName = className;
			prevShowIfCollapsed = showIfCollapsed;
			prevRange = range;
			prevShouldFillLineOnLineBreak = d.options.shouldFillLineOnLineBreak ?? false;
		}

		if (prevClassName !== null) {
			this._renderNormalDecoration(ctx, prevRange!, prevClassName, prevShouldFillLineOnLineBreak, prevShowIfCollapsed, visibleStartLineNumber, output);
		}
	}

	/**
	 * @brief Renders a single normal decoration.
	 * @param ctx The rendering context.
	 * @param range The range of the decoration.
	 * @param className The CSS class name for the decoration.
	 * @param shouldFillLineOnLineBreak Whether to fill the line on a line break.
	 * @param showIfCollapsed Whether to show the decoration if it's collapsed.
	 * @param visibleStartLineNumber The start line number of the viewport.
	 * @param output The output array to which the HTML is written.
	 */
	private _renderNormalDecoration(ctx: RenderingContext, range: Range, className: string, shouldFillLineOnLineBreak: boolean, showIfCollapsed: boolean, visibleStartLineNumber: number, output: string[]): void {
		const linesVisibleRanges = ctx.linesVisibleRangesForRange(range, /*TODO@Alex*/className === 'findMatch');
		if (!linesVisibleRanges) {
			return;
		}

		for (let j = 0, lenJ = linesVisibleRanges.length; j < lenJ; j++) {
			const lineVisibleRanges = linesVisibleRanges[j];
			if (lineVisibleRanges.outsideRenderedLine) {
				continue;
			}
			const lineIndex = lineVisibleRanges.lineNumber - visibleStartLineNumber;

			if (showIfCollapsed && lineVisibleRanges.ranges.length === 1) {
				const singleVisibleRange = lineVisibleRanges.ranges[0];
				if (singleVisibleRange.width < this._typicalHalfwidthCharacterWidth) {
					// collapsed/very small range case => make the decoration visible by expanding its width
					// expand its size on both sides (both to the left and to the right, keeping it centered)
					const center = Math.round(singleVisibleRange.left + singleVisibleRange.width / 2);
					const left = Math.max(0, Math.round(center - this._typicalHalfwidthCharacterWidth / 2));
					lineVisibleRanges.ranges[0] = new HorizontalRange(left, this._typicalHalfwidthCharacterWidth);
				}
			}

			for (let k = 0, lenK = lineVisibleRanges.ranges.length; k < lenK; k++) {
				const expandToLeft = shouldFillLineOnLineBreak && lineVisibleRanges.continuesOnNextLine && lenK === 1;
				const visibleRange = lineVisibleRanges.ranges[k];
				const decorationOutput = (
					'<div class="cdr '
					+ className
					+ '" style="left:'
					+ String(visibleRange.left)
					+ 'px;width:'
					+ (expandToLeft ?
						'100%;' :
						(String(visibleRange.width) + 'px;')
					)
					+ '"></div>'
				);
				output[lineIndex] += decorationOutput;
			}
		}
	}

	/**
	 * @brief Renders the decorations for a specific line.
	 * @param startLineNumber The start line number of the viewport.
	 * @param lineNumber The line number to render.
	 * @returns The HTML string for the decorations on the given line.
	 */
	public render(startLineNumber: number, lineNumber: number): string {
		if (!this._renderResult) {
			return '';
		}
		const lineIndex = lineNumber - startLineNumber;
		if (lineIndex < 0 || lineIndex >= this._renderResult.length) {
			return '';
		}
		return this._renderResult[lineIndex];
	}
}
