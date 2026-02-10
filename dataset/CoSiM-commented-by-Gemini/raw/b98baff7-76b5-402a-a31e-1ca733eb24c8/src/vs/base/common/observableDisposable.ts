/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Emitter } from './event.js';
import { Disposable, IDisposable } from './lifecycle.js';

/**
 * A disposable object that enhances the standard Disposable pattern by making its
 * lifecycle observable.
 *
 * Architectural Role: In a system with complex object relationships, this class
 * allows objects to be safely garbage-collected by managing resource ownership,
 * while also allowing other components to react to an object's disposal without
 * being tightly coupled to it. The public `disposed` property and `onDispose`
 * event provide a standardized way to query and subscribe to an object's lifecycle.
 */
export abstract class ObservableDisposable extends Disposable {
	/**
	 * An internal event emitter that fires when the object is disposed.
	 */
	private readonly _onDispose = this._register(new Emitter<void>());

	/**
	 * An event that fires once when this object has been disposed.
	 *
	 * Functional Utility: This allows other objects to subscribe to the disposal
	 * event. A key feature is that if a listener is attached *after* the object
	 * has already been disposed, the listener is executed immediately.
	 *
	 * @param callback The callback function to be executed on disposal.
	 */
	public onDispose(callback: () => void): IDisposable {
		// Pre-condition: If the object is already disposed, we must honor the
		// contract and execute the callback immediately.
		if (this.disposed) {
			callback();
			return this;
		}

		return this._onDispose.event(callback);
	}

	/**
	 * A convenience method for adding disposable objects that will be disposed
	 * along with this object. Supports a fluent, chainable programming style.
	 * @param disposables The disposable objects to add.
	 */
	public addDisposable(...disposables: IDisposable[]): this {
		for (const disposable of disposables) {
			this._register(disposable);
		}
		return this;
	}

	/**
	 * The internal flag tracking the disposed state of this object.
	 */
	private _disposed = false;

	/**
	 * Public getter for the disposed state of this object.
	 * @returns `true` if the object has been disposed, `false` otherwise.
	 */
	public get disposed(): boolean {
		return this._disposed;
	}

	/**
	 * Disposes of this object and all registered disposables.
	 *
	 * This method is idempotent; subsequent calls will have no effect.
	 */
	public override dispose(): void {
		// Invariant: Ensure disposal logic runs only once.
		if (this.disposed) {
			return;
		}
		this._disposed = true;

		// Block Logic: The disposal sequence is critical.
		// 1. Fire the `onDispose` event to notify all listeners.
		this._onDispose.fire();
		// 2. Call `super.dispose()` to dispose all registered IDisposable objects.
		super.dispose();
	}

	/**
	 * Throws an error if the object has already been disposed.
	 *
	 * Functional Utility: This is a defensive programming tool to prevent "use-after-free"
	 * bugs by asserting that an object is still valid before performing an operation.
	 * The `asserts this is TNotDisposed<this>` signature provides a hint to the
	 * TypeScript compiler for type narrowing.
	 *
	 * @throws If the object has been disposed.
	 * @param error The error message or Error object to throw.
	 */
	public assertNotDisposed(
		error: string | Error,
	): asserts this is TNotDisposed<this> {
		assertNotDisposed(this, error);
	}
}

/**
 * A mapped type that represents a non-disposed version of an object `TObject`,
 * effectively narrowing its `disposed` property to `false`.
 */
type TNotDisposed<TObject extends { disposed: boolean }> = TObject & { disposed: false };

/**
 * Asserts that the provided `object` has not been disposed.
 *
 * Functional Utility: This function uses a TypeScript `asserts` clause. If this
 * function is called and does not throw an error, the TypeScript compiler will
 * "narrow" the type of the `object` in the calling scope to `TNotDisposed<TObject>`,
 * confirming to the type system that `object.disposed` is `false`. This improves
 * static analysis and helps catch potential bugs at compile time.
 *
 * @throws If `object.disposed` is `true`.
 * @param object The object to check.
 * @param error The error message or Error object to throw.
 */
export function assertNotDisposed<TObject extends { disposed: boolean }>(
	object: TObject,
	error: string | Error,
): asserts object is TNotDisposed<TObject> {
	if (!object.disposed) {
		return;
	}

	const errorToThrow = typeof error === 'string'
		? new Error(error)
		: error;

	throw errorToThrow;
}