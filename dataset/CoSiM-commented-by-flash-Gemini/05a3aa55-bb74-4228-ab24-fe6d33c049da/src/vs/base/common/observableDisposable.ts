/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Emitter } from './event.js';
import { Disposable, IDisposable } from './lifecycle.js';

/**
 * @module vs/base/common/observableDisposable
 * @brief Provides a base class for disposable objects that can be observed for their disposal state.
 *
 * This module introduces the `ObservableDisposable` class, which extends the
 * standard `Disposable` pattern by allowing external observers to react to
 * an object's disposal. It also includes an assertion utility to ensure
 * objects are not used after being disposed.
 *
 * Domain: Software Engineering, Lifecycle Management, Event-Driven Programming.
 */

/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Emitter } from './event.js';
import { Disposable, IDisposable } from './lifecycle.js';

/**
 * @class ObservableDisposable
 * @augments Disposable
 * @brief A disposable object that provides observable disposal state and event notifications.
 *
 * This abstract class extends the basic `Disposable` functionality by tracking
 * its disposal status publicly and providing an `onDispose` event. This allows
 * other components to react when an instance of `ObservableDisposable` is
 * deallocated or no longer needed.
 */
export abstract class ObservableDisposable extends Disposable {
	/**
	 * @private
	 * @brief Internal Emitter for the `onDispose` event.
	 * This emitter is used to notify subscribers when the object is disposed.
	 * It is automatically registered for disposal with the parent `Disposable` class.
	 */
	private readonly _onDispose = this._register(new Emitter<void>());

	/**
	 * @method onDispose
	 * @brief Subscribes a callback function to be executed when the object is disposed.
	 *
	 * If the object has already been disposed, the callback is executed immediately.
	 * This ensures that a subscriber always receives the disposal notification,
	 * regardless of when they subscribe relative to the disposal event.
	 *
	 * @param callback The callback function to be called upon disposal.
	 * @returns An `IDisposable` that can be used to unsubscribe the callback.
	 */
	public onDispose(callback: () => void): IDisposable {
		// Block Logic: Checks if the object is already disposed.
		// If true, the callback is invoked immediately to ensure no missed events.
		if (this.disposed) {
			callback();

			// Functional Utility: Returns 'this' as an IDisposable when already disposed,
			// indicating that no further action is needed for unsubscription in this case.
			return this;
		}

		// Functional Utility: Registers the callback with the internal Emitter for future disposal events.
		return this._onDispose.event(callback);
	}

	/**
	 * @method addDisposable
	 * @brief Registers one or more disposables to be disposed alongside this object.
	 *
	 * This method provides a convenient way to manage the lifecycle of child
	 * disposable objects by associating them with the parent `ObservableDisposable`
	 * instance. When the parent is disposed, all registered children will also be disposed.
	 *
	 * @param disposables A rest parameter of `IDisposable` objects to be registered.
	 * @returns The current `ObservableDisposable` instance for method chaining.
	 */
	public addDisposable(...disposables: IDisposable[]): this {
		for (const disposable of disposables) {
			this._register(disposable);
		}

		return this;
	}

	/**
	 * @private
	 * @property _disposed
	 * @brief Internal flag to track the disposal state of the object.
	 * `true` indicates the object has been disposed; `false` otherwise.
	 */
	private _disposed = false;

	/**
	 * @property disposed
	 * @brief Public getter to retrieve the current disposal state of the object.
	 * @returns `true` if the object has been disposed, `false` otherwise.
	 */
	public get disposed(): boolean {
		return this._disposed;
	}

	/**
	 * @method dispose
	 * @override
	 * @brief Disposes the current object and notifies all `onDispose` subscribers.
	 *
	 * This method implements the `IDisposable` interface. It sets the internal
	 * `_disposed` flag to `true`, fires the `_onDispose` event, and then calls
	 * the `dispose` method of the superclass to handle further cleanup.
	 *
	 * Pre-condition: The object must not have been disposed already.
	 * Post-condition: The object's `disposed` property will be `true`, and all
	 * registered callbacks via `onDispose` will have been executed.
	 * @returns void
	 */
	public override dispose(): void {
		// Block Logic: Prevents redundant disposal calls.
		// If the object is already disposed, the method exits early.
		if (this.disposed) {
			return;
		}
		// Functional Utility: Marks the object as disposed.
		this._disposed = true;

		// Functional Utility: Notifies all subscribers that the object has been disposed.
		this._onDispose.fire();
		// Functional Utility: Calls the dispose method of the superclass to ensure proper cleanup chain.
		super.dispose();
	}

	/**
	 * @method assertNotDisposed
	 * @brief Asserts that the current object has not yet been disposed.
	 *
	 * This method is a runtime check to ensure that an object is still in
	 * a valid, non-disposed state before performing operations on it.
	 * If the object has been disposed, it throws an error.
	 *
	 * @throws {Error} If the current object was already disposed.
	 * @param error Error message string or an Error object to throw if the assertion fails.
	 * @returns asserts this is TNotDisposed<this>
	 */
	public assertNotDisposed(
		error: string | Error,
	): asserts this is TNotDisposed<this> {
		// Functional Utility: Delegates the assertion logic to a standalone function for reusability.
		assertNotDisposed(this, error);
	}
}

/**
 * @typedef TNotDisposed
 * @template TObject The type of the object being asserted.
 * @brief Type assertion for an object that has a `disposed` property and is currently not disposed.
 *
 * This type is used with assertion functions (like `assertNotDisposed`) to
 * inform the TypeScript compiler that, after the assertion, the `disposed`
 * property of the `TObject` will definitively be `false`.
 */
type TNotDisposed<TObject extends { disposed: boolean }> = TObject & { disposed: false };

/**
 * @function assertNotDisposed
 * @template TObject The type of the object being asserted.
 * @brief Asserts that a provided object is not yet disposed.
 *
 * This utility function performs a runtime check. If the `disposed` property
 * of the `object` is `true`, it throws an error, indicating an attempt to
 * use an object after its lifecycle has ended. This helps prevent bugs
 * related to stale object references.
 *
 * @throws {Error} If `object.disposed` is `true`.
 * @param object The object to check for disposal status.
 * @param error An error message string or an `Error` object to be thrown if the assertion fails.
 * @returns {asserts object is TNotDisposed<TObject>} A TypeScript assertion that the object is not disposed.
 */
export function assertNotDisposed<TObject extends { disposed: boolean }>(
	object: TObject,
	error: string | Error,
): asserts object is TNotDisposed<TObject> {
	// Block Logic: Checks if the object is already disposed.
	// If the object is not disposed, the function returns silently.
	if (!object.disposed) {
		return;
	}

	// Block Logic: Throws an appropriate error if the object is disposed.
	// This prevents further execution with a stale object.
	const errorToThrow = typeof error === 'string'
		? new Error(error)
		: error;

	throw errorToThrow;
}
