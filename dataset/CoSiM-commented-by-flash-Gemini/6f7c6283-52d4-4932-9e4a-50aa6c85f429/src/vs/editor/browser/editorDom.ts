/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file editorDom.ts
 * @brief DOM and coordinate system utilities for the code editor.
 * 
 * Functional Intent: Provides a unified abstraction for handling screen coordinates 
 * and dynamic styling within the editor. It manages complex transformations 
 * between page-relative, client-relative, and editor-relative coordinate systems, 
 * correctly accounting for CSS transforms (e.g. scale). Additionally, it implements 
 * a reference-counted dynamic CSS rule engine to prevent memory leaks during 
 * frequent UI updates.
 * 
 * Domain: Production Systems, IDE Development, Browser UI Logic.
 */

import * as dom from '../../base/browser/dom.js';
import * as domStylesheetsJs from '../../base/browser/domStylesheets.js';
import { GlobalPointerMoveMonitor } from '../../base/browser/globalPointerMoveMonitor.js';
import { StandardMouseEvent } from '../../base/browser/mouseEvent.js';
import { RunOnceScheduler } from '../../base/common/async.js';
import { Disposable, DisposableStore, IDisposable } from '../../base/common/lifecycle.js';
import { ICodeEditor } from './editorBrowser.js';
import { asCssVariable } from '../../platform/theme/common/colorRegistry.js';
import { ThemeColor } from '../../base/common/themables.js';

/**
 * @class PageCoordinates
 * @brief Representation of coordinates relative to the entire document.
 */
export class PageCoordinates {
	_pageCoordinatesBrand: void = undefined;

	constructor(
		public readonly x: number,
		public readonly y: number
	) { }

	/**
	 * @brief Translates document-level coordinates to viewport-relative (client) coordinates.
	 */
	public toClientCoordinates(targetWindow: Window): ClientCoordinates {
		return new ClientCoordinates(this.x - targetWindow.scrollX, this.y - targetWindow.scrollY);
	}
}

/**
 * @class ClientCoordinates
 * @brief Coordinates relative to the application's client viewport.
 *
 * Logic: Ignores document-level scrolling to provide stable offsets within the 
 * currently visible area.
 */
export class ClientCoordinates {
	_clientCoordinatesBrand: void = undefined;

	constructor(
		public readonly clientX: number,
		public readonly clientY: number
	) { }

	public toPageCoordinates(targetWindow: Window): PageCoordinates {
		return new PageCoordinates(this.clientX + targetWindow.scrollX, this.clientY + targetWindow.scrollY);
	}
}

/**
 * @class EditorPagePosition
 * @brief Encapsulates the physical geometry of the editor element on the page.
 */
export class EditorPagePosition {
	_editorPagePositionBrand: void = undefined;

	constructor(
		public readonly x: number,
		public readonly y: number,
		public readonly width: number,
		public readonly height: number
	) { }
}

/**
 * @class CoordinatesRelativeToEditor
 * @brief Normalised coordinates relative to the editor's top-left origin.
 * 
 * Logic: These coordinates are corrected for CSS scaling, making them safe 
 * for use with internal layout engines and pixel-perfect hit testing.
 */
export class CoordinatesRelativeToEditor {
	_positionRelativeToEditorBrand: void = undefined;

	constructor(
		public readonly x: number,
		public readonly y: number
	) { }
}

export function createEditorPagePosition(editorViewDomNode: HTMLElement): EditorPagePosition {
	const editorPos = dom.getDomNodePagePosition(editorViewDomNode);
	return new EditorPagePosition(editorPos.left, editorPos.top, editorPos.width, editorPos.height);
}

/**
 * createCoordinatesRelativeToEditor - Solves for inverse CSS scaling.
 * 
 * Algorithm: Scale-aware coordinate projection.
 * Logic: Detects discrepancies between bounding client rect (scaled) and 
 * offset dimensions (unscaled) to compute the current zoom/transform level 
 * and applies the inverse to input coordinates.
 */
export function createCoordinatesRelativeToEditor(editorViewDomNode: HTMLElement, editorPagePosition: EditorPagePosition, pos: PageCoordinates) {
	const scaleX = editorPagePosition.width / editorViewDomNode.offsetWidth;
	const scaleY = editorPagePosition.height / editorViewDomNode.offsetHeight;

	// Optimization: Inverse transformation ensures mouse events map correctly to 
	// internal grid cells regardless of parent scale.
	const relativeX = (pos.x - editorPagePosition.x) / scaleX;
	const relativeY = (pos.y - editorPagePosition.y) / scaleY;
	return new CoordinatesRelativeToEditor(relativeX, relativeY);
}

/**
 * @class EditorMouseEvent
 * @brief Extension of StandardMouseEvent enriched with scale-corrected editor offsets.
 */
export class EditorMouseEvent extends StandardMouseEvent {
	_editorMouseEventBrand: void = undefined;

	public readonly isFromPointerCapture: boolean;

	public readonly pos: PageCoordinates;

	public readonly editorPos: EditorPagePosition;

	public readonly relativePos: CoordinatesRelativeToEditor;

	constructor(e: MouseEvent, isFromPointerCapture: boolean, editorViewDomNode: HTMLElement) {
		super(dom.getWindow(editorViewDomNode), e);
		this.isFromPointerCapture = isFromPointerCapture;
		this.pos = new PageCoordinates(this.posx, this.posy);
		this.editorPos = createEditorPagePosition(editorViewDomNode);
		this.relativePos = createCoordinatesRelativeToEditor(editorViewDomNode, this.editorPos, this.pos);
	}
}

/**
 * @class EditorMouseEventFactory
 * @brief Unified event dispatcher for editor-aware mouse interactions.
 */
export class EditorMouseEventFactory {

	private readonly _editorViewDomNode: HTMLElement;

	constructor(editorViewDomNode: HTMLElement) {
		this._editorViewDomNode = editorViewDomNode;
	}

	private _create(e: MouseEvent): EditorMouseEvent {
		return new EditorMouseEvent(e, false, this._editorViewDomNode);
	}

	public onContextMenu(target: HTMLElement, callback: (e: EditorMouseEvent) => void): IDisposable {
		return dom.addDisposableListener(target, 'contextmenu', (e: MouseEvent) => {
			callback(this._create(e));
		});
	}

	public onMouseUp(target: HTMLElement, callback: (e: EditorMouseEvent) => void): IDisposable {
		return dom.addDisposableListener(target, 'mouseup', (e: MouseEvent) => {
			callback(this._create(e));
		});
	}

	public onMouseDown(target: HTMLElement, callback: (e: EditorMouseEvent) => void): IDisposable {
		return dom.addDisposableListener(target, dom.EventType.MOUSE_DOWN, (e: MouseEvent) => {
			callback(this._create(e));
		});
	}

	public onPointerDown(target: HTMLElement, callback: (e: EditorMouseEvent, pointerId: number) => void): IDisposable {
		return dom.addDisposableListener(target, dom.EventType.POINTER_DOWN, (e: PointerEvent) => {
			callback(this._create(e), e.pointerId);
		});
	}

	public onMouseLeave(target: HTMLElement, callback: (e: EditorMouseEvent) => void): IDisposable {
		return dom.addDisposableListener(target, dom.EventType.MOUSE_LEAVE, (e: MouseEvent) => {
			callback(this._create(e));
		});
	}

	public onMouseMove(target: HTMLElement, callback: (e: EditorMouseEvent) => void): IDisposable {
		return dom.addDisposableListener(target, 'mousemove', (e) => callback(this._create(e)));
	}
}

/**
 * @class EditorPointerEventFactory
 * @brief Specialised factory for Pointer API events, maintaining editor-relative context.
 */
export class EditorPointerEventFactory {

	private readonly _editorViewDomNode: HTMLElement;

	constructor(editorViewDomNode: HTMLElement) {
		this._editorViewDomNode = editorViewDomNode;
	}

	private _create(e: MouseEvent): EditorMouseEvent {
		return new EditorMouseEvent(e, false, this._editorViewDomNode);
	}

	public onPointerUp(target: HTMLElement, callback: (e: EditorMouseEvent) => void): IDisposable {
		return dom.addDisposableListener(target, 'pointerup', (e: MouseEvent) => {
			callback(this._create(e));
		});
	}

	public onPointerDown(target: HTMLElement, callback: (e: EditorMouseEvent, pointerId: number) => void): IDisposable {
		return dom.addDisposableListener(target, dom.EventType.POINTER_DOWN, (e: PointerEvent) => {
			callback(this._create(e), e.pointerId);
		});
	}

	public onPointerLeave(target: HTMLElement, callback: (e: EditorMouseEvent) => void): IDisposable {
		return dom.addDisposableListener(target, dom.EventType.POINTER_LEAVE, (e: MouseEvent) => {
			callback(this._create(e));
		});
	}

	public onPointerMove(target: HTMLElement, callback: (e: EditorMouseEvent) => void): IDisposable {
		return dom.addDisposableListener(target, 'pointermove', (e) => callback(this._create(e)));
	}
}

/**
 * @class GlobalEditorPointerMoveMonitor
 * @brief Lifecycle monitor for document-wide pointer tracking with editor-specific event normalization.
 */
export class GlobalEditorPointerMoveMonitor extends Disposable {

	private readonly _editorViewDomNode: HTMLElement;
	private readonly _globalPointerMoveMonitor: GlobalPointerMoveMonitor;
	private _keydownListener: IDisposable | null;

	constructor(editorViewDomNode: HTMLElement) {
		super();
		this._editorViewDomNode = editorViewDomNode;
		this._globalPointerMoveMonitor = this._register(new GlobalPointerMoveMonitor());
		this._keydownListener = null;
	}

	/**
	 * startMonitoring - Observes all pointer movements until stopped or a key is pressed.
	 */
	public startMonitoring(
		initialElement: Element,
		pointerId: number,
		initialButtons: number,
		pointerMoveCallback: (e: EditorMouseEvent) => void,
		onStopCallback: (browserEvent?: PointerEvent | KeyboardEvent) => void
	): void {

		// Block Logic: Interruption guard.
		// Invariant: Cancels monitoring if any non-modifier key is pressed to release capture state.
		this._keydownListener = dom.addStandardDisposableListener(<any>initialElement.ownerDocument, 'keydown', (e) => {
			const chord = e.toKeyCodeChord();
			if (chord.isModifierKey()) {
				return;
			}
			this._globalPointerMoveMonitor.stopMonitoring(true, e.browserEvent);
		}, true);

		this._globalPointerMoveMonitor.startMonitoring(
			initialElement,
			pointerId,
			initialButtons,
			(e) => {
				// Functional Intent: Normalizes global movements to editor-relative coordinates.
				pointerMoveCallback(new EditorMouseEvent(e, true, this._editorViewDomNode));
			},
			(e) => {
				this._keydownListener!.dispose();
				onStopCallback(e);
			}
		);
	}

	public stopMonitoring(): void {
		this._globalPointerMoveMonitor.stopMonitoring(true);
	}
}


/**
 * @class DynamicCssRules
 * @brief Reference-counted manager for dynamically generated CSS stylesheets.
 * 
 * Algorithm: Content-addressed style reuse.
 * Logic:
 * 1. Computes a hash key from CSS properties.
 * 2. Deduplicates identical rules across multiple subscribers.
 * 3. Schedules garbage collection to purge unused rules after a cooldown period.
*/
export class DynamicCssRules {
	private static _idPool = 0;
	private readonly _instanceId = ++DynamicCssRules._idPool;
	private _counter = 0;
	private readonly _rules = new Map<string, RefCountedCssRule>();

	// Optimization: Debounced cleanup to prevent churn during rapid layout shifts.
	private readonly _garbageCollectionScheduler = new RunOnceScheduler(() => this.garbageCollect(), 1000);

	constructor(private readonly _editor: ICodeEditor) {
	}

	/**
	 * createClassNameRef - Returns a reference-counted CSS class name.
	 */
	public createClassNameRef(options: CssProperties): ClassNameReference {
		const rule = this.getOrCreateRule(options);
		rule.increaseRefCount();

		return {
			className: rule.className,
			dispose: () => {
				rule.decreaseRefCount();
				this._garbageCollectionScheduler.schedule();
			}
		};
	}

	private getOrCreateRule(properties: CssProperties): RefCountedCssRule {
		const key = this.computeUniqueKey(properties);
		let existingRule = this._rules.get(key);
		if (!existingRule) {
			const counter = this._counter++;
			existingRule = new RefCountedCssRule(key, `dyn-rule-${this._instanceId}-${counter}`,
				dom.isInShadowDOM(this._editor.getContainerDomNode())
					? this._editor.getContainerDomNode()
					: undefined,
				properties
			);
			this._rules.set(key, existingRule);
		}
		return existingRule;
	}

	private computeUniqueKey(properties: CssProperties): string {
		return JSON.stringify(properties);
	}

	private garbageCollect() {
		/**
		 * Block Logic: Purge sweep.
		 * Invariant: Deletes rules from the DOM only when their reference count hits zero.
		 */
		for (const rule of this._rules.values()) {
			if (!rule.hasReferences()) {
				this._rules.delete(rule.key);
				rule.dispose();
			}
		}
	}
}

export interface ClassNameReference extends IDisposable {
	className: string;
}

export interface CssProperties {
	border?: string;
	borderColor?: string | ThemeColor;
	borderRadius?: string;
	fontStyle?: string;
	fontWeight?: string;
	fontSize?: string;
	fontFamily?: string;
	unicodeBidi?: string;
	textDecoration?: string;
	color?: string | ThemeColor;
	backgroundColor?: string | ThemeColor;
	opacity?: string;
	verticalAlign?: string;
	cursor?: string;
	margin?: string;
	padding?: string;
	width?: string;
	height?: string;
	display?: string;
}

/**
 * @class RefCountedCssRule
 * @brief Managed lifecycle for an individual DOM stylesheet.
 */
class RefCountedCssRule {
	private _referenceCount: number = 0;
	private _styleElement: HTMLStyleElement | undefined;
	private readonly _styleElementDisposables: DisposableStore;

	constructor(
		public readonly key: string,
		public readonly className: string,
		_containerElement: HTMLElement | undefined,
		public readonly properties: CssProperties,
	) {
		this._styleElementDisposables = new DisposableStore();
		this._styleElement = domStylesheetsJs.createStyleSheet(_containerElement, undefined, this._styleElementDisposables);
		this._styleElement.textContent = this.getCssText(this.className, this.properties);
	}

	/**
	 * getCssText - Serializes a property bag into a valid CSS string.
	 */
	private getCssText(className: string, properties: CssProperties): string {
		let str = `.${className} {`;
		for (const prop in properties) {
			const value = (properties as any)[prop] as string | ThemeColor;
			let cssValue;
			if (typeof value === 'object') {
				// Logic: Translates platform ThemeColors to CSS variables.
				cssValue = asCssVariable(value.id);
			} else {
				cssValue = value;
			}

			const cssPropName = camelToDashes(prop);
			str += `\n\t${cssPropName}: ${cssValue};`;
		}
		str += `\n}`;
		return str;
	}

	public dispose(): void {
		this._styleElementDisposables.dispose();
		this._styleElement = undefined;
	}

	public increaseRefCount(): void {
		this._referenceCount++;
	}

	public decreaseRefCount(): void {
		this._referenceCount--;
	}

	public hasReferences(): boolean {
		return this._referenceCount > 0;
	}
}

/**
 * camelToDashes - Utility for CSS property name normalization.
 */
function camelToDashes(str: string): string {
	return str.replace(/(^[A-Z])/, ([first]) => first.toLowerCase())
		.replace(/([A-Z])/g, ([letter]) => `-${letter.toLowerCase()}`);
}
