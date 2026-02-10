/**
 * @file This file defines a utility class, `ObservableDisposable`, that extends
 * the base `Disposable` class to provide observable behavior for its disposal state.
 * It is a common pattern in this codebase for managing object lifecycles and
 * preventing memory leaks.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Emitter } from './event.js';
import { Disposable, IDisposable } from './lifecycle.js';

/**
 * @class ObservableDisposable
 * @brief An abstract base class for disposable objects that provides an
 * observable `disposed` state and an `onDispose` event.
 *
 * @details This class allows other parts of the system to react to the
 * disposal of an object. It ensures that disposal logic is executed only once
 * and provides a clear way to check if an object is still valid.
 */
export abstract class ObservableDisposable extends Disposable {
	/**
	 * @property
	 * @private
	 * An event emitter that fires when the object is disposed.
	 */
	private readonly _onDispose = this._register(new Emitter<void>());

	/**
	 * @description An event that fires when this object is disposed.
	 * If the object is already disposed, the callback is executed immediately.
	 * @param callback The function to execute upon disposal.
	 * @returns An `IDisposable` that can be used to unregister the callback.
	 */
	public onDispose(callback: () => void): IDisposable {
		// if already disposed, execute the callback immediately
		if (this.disposed) {
			callback();

			return this;
		}

		return this._onDispose.event(callback);
	}

	/**
	 * @description Registers one or more disposable objects with this instance.
	 * When this `ObservableDisposable` is disposed, all registered disposables
	 * will also be disposed.
	 * @param disposables A list of objects to be disposed along with this instance.
	 * @returns The current instance, allowing for chained calls.
	 */
	public addDisposable(...disposables: IDisposable[]): this {
		for (const disposable of disposables) {
			this._register(disposable);
		}

		return this;
	}

	/**
	 * @property
	 * @private
	 * Internal flag to track the disposal state.
	 */
	private _disposed = false;

	/**
	 * @description A public getter to check if the object has been disposed.
	 * @returns `true` if the object is disposed, `false` otherwise.
	 */
	public get disposed(): boolean {
		return this._disposed;
	}

	/**
	 * @description Disposes this object and all registered disposables.
	 * This method is idempotent; subsequent calls have no effect.
	 * It fires the `_onDispose` event to notify listeners.
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
	 * @description Asserts that the object has not been disposed.
	 * This is a debugging utility to catch use-after-free errors.
	 *
	 * @throws If the object has already been disposed.
	 * @param error The error message or Error object to throw.
	 */
	public assertNotDisposed(
		error: string | Error,
	): asserts this is TNotDisposed<this> {
		assertNotDisposed(this, error);
	}
}

/**
 * @description A TypeScript type guard that narrows the type of an object to
 * one where its `disposed` property is guaranteed to be `false`.
 */
type TNotDisposed<TObject extends { disposed: boolean }> = TObject & { disposed: false };

/**
 * @description A utility function that asserts an object has not been disposed.
 * It serves as a runtime check to enforce correct object lifecycle management.
 *
 * @throws If the provided `object.disposed` is `true`.
 * @param object The object to check.
 * @param error The error message or Error object to throw if the assertion fails.
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
