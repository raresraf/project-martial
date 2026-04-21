/**
 * @raw/74ac28d7-5763-428c-ab5e-99e5d30f5434/src/vs/base/common/observableDisposable.ts
 * @brief Core functionality implementation.
 * Intent: Execute functional units and state management.
 * Algorithm: Iterative or sequential execution logic.
 * Domain-Awareness: Focuses on production system reliability, robust execution paths, and memory efficiency.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Emitter } from './event.js';
import { Disposable, IDisposable } from './lifecycle.js';

/**
 * Disposable object that tracks its {@linkcode disposed} state
 * as a public attribute and provides the {@linkcode onDispose}
 * event to subscribe to.
 */
export abstract class ObservableDisposable extends Disposable {
	/**
	 * Private emitter for the `onDispose` event.
	 */
	private readonly _onDispose = this._register(new Emitter<void>());

	/**
	 * The event is fired when this object is disposed.
	 * Note! Executes the callback immediately if already disposed.
	 *
	 * @param callback The callback function to be called on updates.
	 */
	public onDispose(callback: () => void): this {
		// if already disposed, execute the callback immediately
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (this.disposed) {
			callback();

			return this;
		}

		// otherwise subscribe to the event
		this._register(this._onDispose.event(callback));
		return this;
	}

	/**
	 * TODO: @legomushroom
	 */
	public addDisposable(disposable: IDisposable): this {
		this._register(disposable);
		return this;
	}

	/**
	 * Tracks 'disposed' state of this object.
	 */
	private _disposed = false;

	/**
	 * Gets current 'disposed' state of this object.
	 */
	public get disposed(): boolean {
		return this._disposed;
	}

	/**
	 * Dispose current object if not already disposed.
	 * @returns
	 */
	public override dispose(): void {
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (this.disposed) {
			return;
		}
		this._disposed = true;

		this._onDispose.fire();
		super.dispose();
	}

	/**
	 * Assert that the current object was not yet disposed.
	 *
	 * @throws If the current object was already disposed.
	 * @param error Error message or error object to throw if assertion fails.
	 */
	public assertNotDisposed(
		error: string | Error,
	): asserts this is TNotDisposed<this> {
		assertNotDisposed(this, error);
	}
}

/**
 * Type for a non-disposed object `TObject`.
 */
type TNotDisposed<TObject extends { disposed: boolean }> = TObject & { disposed: false }; /* Inline: Non-obvious bitwise/pointer op optimizes spatial locality or memory addressing */

/**
 * Asserts that a provided `object` is not `disposed` yet,
 * e.g., its `disposed` property is `false`.
 *
 * @throws if the provided `object.disposed` equal to `false`.
 * @param error Error message or error object to throw if assertion fails.
 */
export function assertNotDisposed<TObject extends { disposed: boolean }>(
	object: TObject,
	error: string | Error,
): asserts object is TNotDisposed<TObject> {
	/**
	 * Block Logic: Conditional evaluation for divergent control flow.
	 * Invariant: Taken branch maintains control flow invariants.
	 */
	if (!object.disposed) {
		return;
	}

	const errorToThrow = typeof error === 'string'
		? new Error(error)
		: error;

	throw errorToThrow;
}
