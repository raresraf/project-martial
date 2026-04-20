/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @cea35623-2d7a-455c-b137-eed88e258d03/src/vs/base/common/observableDisposable.ts
 * @brief Extension of the Disposable pattern with observable lifecycle state.
 * 
 * Functional Intent: Enhances standard resource management by providing an 
 * explicit, queryable 'disposed' state and an event emitter for disposal 
 * notifications. It facilitates reactive cleanup and provides runtime guards 
 * to prevent illegal operations on terminated objects.
 * 
 * Domain: Platform Architecture, Resource Lifecycle, Reactive UI.
 */

import { Emitter } from './event.js';
import { Disposable, IDisposable } from './lifecycle.js';

/**
 * @brief Abstract base for objects with observable termination.
 * 
 * Functional Utility: Tracks its own lifecycle state and broadcasts disposal 
 * events to external observers.
 */
export abstract class ObservableDisposable extends Disposable {
	/**
	 * @private
	 * Internal dispatcher for lifecycle events.
	 */
	private readonly _onDispose = this._register(new Emitter<void>());

	/**
	 * onDispose - Subscribes to the object's termination.
	 * @param callback Logic to execute when the object is disposed.
	 * 
	 * Block Logic: Immediate or deferred notification.
	 * Logic: If the object is already terminal, schedules the callback for the 
	 * next event loop cycle (async consistency) and returns a cancellation handle. 
	 * Otherwise, registers the callback for future execution when 'dispose' is invoked.
	 */
	public onDispose(callback: () => void): IDisposable {
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
	 * @brief Bulk registration utility for child resources.
	 */
	public addDisposable(...disposables: IDisposable[]): this {
		for (const disposable of disposables) {
			this._register(disposable);
		}

		return this;
	}

	private _disposed = false;

	/**
	 * @brief Public predicate for checking the object's lifecycle status.
	 */
	public get disposed(): boolean {
		return this._disposed;
	}

	/**
	 * dispose - Idempotent resource cleanup.
	 * 
	 * Block Logic: Finalization sequence.
	 * Logic: 
	 * 1. Checks for existing termination (idempotency guard).
	 * 2. Mark as terminal to prevent race conditions during cleanup.
	 * 3. Broadcasts the disposal event to all listeners.
	 * 4. Delegates to the base Disposable for recursive cleanup of registered children.
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
	 * @brief Integrated type guard and runtime invariant check.
	 */
	public assertNotDisposed(
		error: string | Error,
	): asserts this is TNotDisposed<this> {
		assertNotDisposed(this, error);
	}
}

/**
 * @brief Utility type representing a guaranteed active instance.
 */
type TNotDisposed<TObject extends { disposed: boolean }> = TObject & { disposed: false };

/**
 * assertNotDisposed - Validation utility for disposal state.
 * @throws Error if the target object has been disposed.
 * 
 * Block Logic: Invariant enforcement.
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
