/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file observableDisposable.ts
 * @brief Provides a disposable base class that allows other parts of the system
 * to observe its disposal, enhancing resource management in an event-driven
 * architecture.
 */

import { Emitter } from './event.js';
import { Disposable, IDisposable } from './lifecycle.js';

/**
 * @class ObservableDisposable
 * @extends Disposable
 * @description An abstract base class for creating disposable objects whose
 * disposal can be observed. This is useful for components that need to release
 * resources or unregister listeners when another object they depend on is disposed.
 * It encapsulates the logic for tracking the disposed state and notifying listeners.
 */
export abstract class ObservableDisposable extends Disposable {
	/**
	 * @property _onDispose
	 * @private
	 * @description The underlying event emitter for the `onDispose` event.
	 */
	private readonly _onDispose = this._register(new Emitter<void>());

	/**
	 * @method onDispose
	 * @description Registers a callback to be executed when this object is disposed.
	 *
	 * Functional Utility: This provides a reliable mechanism for dependent objects
	 * to clean up their own resources. If the object is already disposed, the
	 * callback is executed asynchronously in a `setTimeout` to ensure consistent
	 * and predictable behavior, preventing race conditions where a listener might
	 * be added after disposal.
	 *
	 * @param callback The function to be called on disposal.
	 * @returns An `IDisposable` to unregister the callback.
	 */
	public onDispose(callback: () => void): IDisposable {
		// if already disposed, execute the callback immediately
		if (this.disposed) {
			const timeoutHandle = setTimeout(callback);

			return {
				dispose: () => {
					clearTimeout(timeoutHandle);
				},
			};
		}

		return this._onDispose.event(callback);
	}

	/**
	 * @method addDisposable
	 * @description A convenience method to register multiple `IDisposable` objects
	 * that will be disposed of when this object is disposed.
	 */
	public addDisposable(...disposables: IDisposable[]): this {
		for (const disposable of disposables) {
			this._register(disposable);
		}

		return this;
	}

	/**
	 * @property _disposed
	 * @private
	 * @description Internal flag to track the disposed state.
	 */
	private _disposed = false;

	/**
	 * @property disposed
	 * @description Public getter to check if the object has been disposed.
	 */
	public get disposed(): boolean {
		return this._disposed;
	}

	/**
	 * @method dispose
	 * @description Overrides the base `dispose` method to fire the `_onDispose`
	 * event before completing the disposal process.
	 */
	public override dispose(): void {
		if (this.disposed) {
			return;
		}
		this._disposed = true;

		this._onDispose.fire();
		super.dispose();
	}

	/**
	 * @method assertNotDisposed
	 * @description Asserts that the object has not been disposed.
	 *
	 * Functional Utility: This is a defensive programming tool to prevent
	 * "use-after-free" errors by ensuring that methods are not called on an
	 * object that has already released its resources.
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
 * @typedef TNotDisposed
 * @description A TypeScript utility type that refines the type of an object `TObject`
 * to one where its `disposed` property is statically known to be `false`. This is
 * used in assertion functions for type narrowing.
 */
type TNotDisposed<TObject extends { disposed: boolean }> = TObject & { disposed: false };

/**
 * @function assertNotDisposed
 * @description A standalone assertion function that validates an object has not been disposed.
 *
 * Functional Utility: Leverages TypeScript's `asserts` keyword for control-flow
 * based type analysis. If this function returns successfully, the TypeScript compiler
 * will narrow the type of the `object` to `TNotDisposed<TObject>`, meaning its
 * `disposed` property is treated as `false` in subsequent code paths.
 *
 * @throws If `object.disposed` is `true`.
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
