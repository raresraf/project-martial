/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file editorDom.ts
 * @brief DOM and Coordinate utilities for the VS Code editor.
 * 
 * Functional Intent: Provides high-level abstractions for handling mouse and pointer 
 * events within the editor. It specializes in coordinate transformations that 
 * account for CSS scale transforms, ensuring that internal editor metrics remain 
 * consistent regardless of UI scaling. Additionally, it manages a reference-counted 
 * dynamic CSS injection system to prevent stylesheet pollution during rapid UI updates.
 * 
 * Domain: Production Systems, UI Frameworks, Browser Interoperability.
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
 * @brief Represents a position relative to the entire document.
 */
export class PageCoordinates {
	_pageCoordinatesBrand: void = undefined;

	constructor(
		public readonly x: number,
		public readonly y: number
	) { }

	public toClientCoordinates(targetWindow: Window): ClientCoordinates {
		return new ClientCoordinates(this.x - targetWindow.scrollX, this.y - targetWindow.scrollY);
	}
}

/**
 * @class ClientCoordinates
 * @brief Represents a position relative to the document's visible viewport.
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
 * @brief Geometric bounds of the editor container within the page context.
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
 * @brief Normalized coordinates for internal editor arithmetic.
 * 
 * Logic: Transformed relative to the editor's top-left corner, adjusting for 
 * any scale transformations detected in the DOM layout.
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
 * createCoordinatesRelativeToEditor - Normalizes raw page coordinates to editor-local space.
 * 
 * Algorithm: Scale-aware coordinate translation.
 * Logic: Computes the effective X/Y scaling factor by comparing the bounding 
 * client rect (physical pixels) against offset dimensions (layout pixels), 
 * then applies the inverse transformation to the delta.
 */
export function createCoordinatesRelativeToEditor(editorViewDomNode: HTMLElement, editorPagePosition: EditorPagePosition, pos: PageCoordinates) {
	const scaleX = editorPagePosition.width / editorViewDomNode.offsetWidth;
	const scaleY = editorPagePosition.height / editorViewDomNode.offsetHeight;

	// Invariant: Returns coordinates that map correctly to the editor's internal buffer even if the UI is zoomed or scaled.
	const relativeX = (pos.x - editorPagePosition.x) / scaleX;
	const relativeY = (pos.y - editorPagePosition.y) / scaleY;
	return new CoordinatesRelativeToEditor(relativeX, relativeY);
}

/**
 * @class EditorMouseEvent
 * @brief Specialized mouse event wrapper containing multi-coordinate context.
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
 * @brief Convenience builder for binding DOM events to EditorMouseEvent handlers.
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
		return dom.addDisposableListener(target, dom.EventType.CONTEXT_MENU, (e: MouseEvent) => {
			callback(this._create(e));
		});
	}

	public onMouseUp(target: HTMLElement, callback: (e: EditorMouseEvent) => void): IDisposable {
		return dom.addDisposableListener(target, dom.EventType.MOUSE_UP, (e: MouseEvent) => {
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
		return dom.addDisposableListener(target, dom.EventType.MOUSE_MOVE, (e) => callback(this._create(e)));
	}
}

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
 * @brief Orchestrates document-level pointer tracking with cancellation logic.
 * 
 * Logic: Tracks movement across the entire document until a keypress or release 
 * occurs. Blocks non-modifier keys to prevent state conflicts during drag operations.
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

	public startMonitoring(
		initialElement: Element,
		pointerId: number,
		initialButtons: number,
		pointerMoveCallback: (e: EditorMouseEvent) => void,
		onStopCallback: (browserEvent?: PointerEvent | KeyboardEvent) => void
	): void {

		// Block Logic: Cancellation guard.
		// Invariant: Terminate monitoring if any non-modifier key is pressed.
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
 * @brief Reference-counted manager for dynamically generated CSS classes.
 * 
 * Functional Intent: Minimizes stylesheet updates by caching and reusing 
 * rules with identical properties. Implements a delayed garbage collection 
 * cycle to avoid performance jitter during rapid decoration changes.
 */
export class DynamicCssRules {
	private static _idPool = 0;
	private readonly _instanceId = ++DynamicCssRules._idPool;
	private _counter = 0;
	private readonly _rules = new Map<string, RefCountedCssRule>();

	// Logic: Debounced cleanup to allow rule reuse within a 1000ms window.
	private readonly _garbageCollectionScheduler = new RunOnceScheduler(() => this.garbageCollect(), 1000);

	constructor(private readonly _editor: ICodeEditor) {
	}

	/**
	 * createClassNameRef - Requests a CSS class for a set of properties.
	 */
	public createClassNameRef(options: CssProperties): ClassNameReference {
		const rule = this.getOrCreateRule(options);
		rule.increaseRefCount();

		return {
			className: rule.className,
			dispose: () => {
				// Functional Utility: Automatic cleanup when the consumer is disposed.
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
		 * Block Logic: Rule eviction sweep.
		 * Invariant: Deletes only rules that have zero active consumers.
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
 * @brief Encapsulates a single CSS rule with lifecycle tracking.
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
	 * getCssText - Serializes TS properties into valid CSS text.
	 */
	private getCssText(className: string, properties: CssProperties): string {
		let str = `.${className} {`;
		for (const prop in properties) {
			const value = (properties as any)[prop] as string | ThemeColor;
			let cssValue;
			// Logic: Resolve ThemeColor references to CSS variables.
			if (typeof value === 'object') {
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
 * camelToDashes - Converts JS property names to CSS standard kebab-case.
 */
function camelToDashes(str: string): string {
	return str.replace(/(^[A-Z])/, ([first]) => first.toLowerCase())
		.replace(/([A-Z])/g, ([letter]) => `-${letter.toLowerCase()}`);
}
