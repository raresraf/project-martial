/**
 * @file This file contains a collection of DOM-related utilities and classes specifically
 * tailored for the VS Code editor's browser rendering. It provides abstractions for
 * handling coordinate systems, mouse events, and dynamic CSS styling, ensuring
 * consistent and correct behavior across different parts of the editor.
 *
 * Production Systems:
 * The abstractions in this file are critical for the correct rendering and interaction
 * model of the editor. For example:
 * - `CoordinatesRelativeToEditor` correctly handles mouse input when the editor is scaled (zoomed),
 *   preventing bugs in cursor positioning and text selection.
 * - `DynamicCssRules` provides an efficient way to manage thousands of style rules for syntax
 *   highlighting and decorations without polluting the DOM or causing performance issues.
 * - `EditorMouseEvent` standardizes mouse event data, making event handling logic more robust
 *   and easier to maintain.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

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
 * Coordinates relative to the whole document (e.g., `event.pageX` and `event.pageY`).
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
 * Coordinates within the application's client area (i.e., the origin is the top-left
 * of the viewport, independent of page scroll).
 * For example, `event.clientX` and `event.clientY`.
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
 * The position and dimensions of the editor's DOM node, relative to the entire page.
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
 * Coordinates relative to the top-left corner of the editor's DOM node.
 * **NOTE**: This position is calculated by transforming page coordinates to be
 * relative to the editor, crucially taking any CSS `transform: scale()` effects
 * into account. This makes them safe to use for internal editor calculations.
 * **NOTE**: These coordinates can be negative if the mouse is outside the editor.
 */
export class CoordinatesRelativeToEditor {
	_positionRelativeToEditorBrand: void = undefined;

	constructor(
		public readonly x: number,
		public readonly y: number
	) { }
}

/**
 * Creates an `EditorPagePosition` for the editor's view DOM node.
 */
export function createEditorPagePosition(editorViewDomNode: HTMLElement): EditorPagePosition {
	const editorPos = dom.getDomNodePagePosition(editorViewDomNode);
	return new EditorPagePosition(editorPos.left, editorPos.top, editorPos.width, editorPos.height);
}

/**
 * Creates coordinates relative to the editor's view DOM node.
 * @param editorViewDomNode The editor's view DOM node.
 * @param editorPagePosition The editor's page position.
 * @param pos The page coordinates to transform.
 */
export function createCoordinatesRelativeToEditor(editorViewDomNode: HTMLElement, editorPagePosition: EditorPagePosition, pos: PageCoordinates) {
	// Intent: This function is critical for features like editor zoom. It detects if the editor's
	// rendered size (from `getBoundingClientRect`) differs from its layout size (`offsetWidth`).
	// This difference indicates a CSS `transform: scale()` is active. By calculating the scale
	// factor, we can correctly translate mouse coordinates into the editor's internal, unscaled
	// coordinate system.
	const scaleX = editorPagePosition.width / editorViewDomNode.offsetWidth;
	const scaleY = editorPagePosition.height / editorViewDomNode.offsetHeight;

	// Adjust mouse offsets to reverse the scaling transformation.
	const relativeX = (pos.x - editorPagePosition.x) / scaleX;
	const relativeY = (pos.y - editorPagePosition.y) / scaleY;
	return new CoordinatesRelativeToEditor(relativeX, relativeY);
}

/**
 * A standardized wrapper for mouse events within the editor.
 * It provides coordinates in multiple, well-defined systems.
 */
export class EditorMouseEvent extends StandardMouseEvent {
	_editorMouseEventBrand: void = undefined;

	/**
	 * If the event is a result of using `setPointerCapture`, the `event.target`
	 * does not necessarily reflect the position in the editor.
	 */
	public readonly isFromPointerCapture: boolean;

	/**
	 * Coordinates relative to the whole document (page).
	 */
	public readonly pos: PageCoordinates;

	/**
	 * The editor's position and dimensions relative to the whole document (page).
	 */
	public readonly editorPos: EditorPagePosition;

	/**
	 * Coordinates relative to the top-left of the editor, adjusted for CSS scaling.
	 * This is the preferred coordinate system for most internal editor logic.
	 */
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
 * A factory for creating standardized `EditorMouseEvent`s from DOM mouse events.
 * This simplifies event listener setup.
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

/**
 * A factory for creating standardized `EditorMouseEvent`s from DOM pointer events.
 * This simplifies event listener setup for pointer-based interactions.
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
 * A utility that monitors pointer movement events globally across the document.
 * This is essential for interactions like drag-and-drop or resizing widgets,
 * where the pointer may leave the initial element's bounds. It uses `setPointerCapture`.
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
	 * Starts monitoring pointer movements.
	 * @param initialElement The element that initiated the pointer capture.
	 * @param pointerId The ID of the pointer to monitor.
	 * @param initialButtons The initial mouse buttons state.
	 * @param pointerMoveCallback A callback for each pointer move event.
	 * @param onStopCallback A callback for when monitoring stops.
	 */
	public startMonitoring(
		initialElement: Element,
		pointerId: number,
		initialButtons: number,
		pointerMoveCallback: (e: EditorMouseEvent) => void,
		onStopCallback: (browserEvent?: PointerEvent | KeyboardEvent) => void
	): void {

		// Block Logic: Installs a capturing keydown listener to stop monitoring on most key presses.
		// This provides a way for the user to cancel a drag operation with the keyboard.
		this._keydownListener = dom.addStandardDisposableListener(<any>initialElement.ownerDocument, 'keydown', (e) => {
			const chord = e.toKeyCodeChord();
			// Pre-condition: Allow modifier keys to be pressed without stopping the drag.
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
 * A helper to create and manage dynamic CSS rules, bound to a class name.
 *
 * Algorithm:
 * This class uses reference counting to manage the lifecycle of CSS rules.
 * 1. When a style is requested via `createClassNameRef`, its properties are serialized
 *    into a unique key.
 * 2. If a rule for this key already exists, its reference count is incremented.
 * 3. If not, a new `RefCountedCssRule` is created, which injects a `<style>` element
 *    into the DOM, and its reference count is set to 1.
 * 4. When a consumer disposes of its `ClassNameReference`, the rule's reference count
 *    is decremented.
 * 5. A `RunOnceScheduler` schedules a garbage collection pass. This delay allows
 *    a rule to be quickly reused if it's disposed and then requested again in short order.
 * 6. The garbage collector removes any rules with a reference count of zero.
 */
export class DynamicCssRules {
	private static _idPool = 0;
	private readonly _instanceId = ++DynamicCssRules._idPool;
	private _counter = 0;
	private readonly _rules = new Map<string, RefCountedCssRule>();

	// A scheduler for delayed garbage collection of unused CSS rules.
	private readonly _garbageCollectionScheduler = new RunOnceScheduler(() => this.garbageCollect(), 1000);

	constructor(private readonly _editor: ICodeEditor) {
	}

	/**
	 * Creates a reference to a CSS class name for the given properties.
	 * @param options The CSS properties for the rule.
	 * @returns A disposable reference to the generated class name.
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
		for (const rule of this._rules.values()) {
			if (!rule.hasReferences()) {
				this._rules.delete(rule.key);
				rule.dispose();
			}
		}
	}
}

/**
 * A disposable reference to a dynamically generated CSS class name.
 */
export interface ClassNameReference extends IDisposable {
	className: string;
}

/**
 * A collection of CSS properties that can be applied to a dynamic rule.
 */
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
 * An internal class that represents a single, reference-counted CSS rule
 * and manages its corresponding `<style>` element in the DOM.
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

	private getCssText(className: string, properties: CssProperties): string {
		let str = `.${className} {`;
		for (const prop in properties) {
			const value = (properties as any)[prop] as string | ThemeColor;
			let cssValue;
			// Pre-condition: Check if the value is a `ThemeColor` object.
			if (typeof value === 'object') {
				// If so, convert it to a CSS variable reference (e.g., `var(--vscode-editor-foreground)`).
				cssValue = asCssVariable(value.id);
			} else {
				cssValue = value;
			}

			const cssPropName = camelToDashes(prop);
			str += `
	${cssPropName}: ${cssValue};`;
		}
		str += `
}`;
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
 * Converts a camelCase string to a dash-separated string (e.g., "backgroundColor" -> "background-color").
 */
function camelToDashes(str: string): string {
	return str.replace(/(^[A-Z])/, ([first]) => first.toLowerCase())
		.replace(/([A-Z])/g, ([letter]) => `-${letter.toLowerCase()}`);
}
