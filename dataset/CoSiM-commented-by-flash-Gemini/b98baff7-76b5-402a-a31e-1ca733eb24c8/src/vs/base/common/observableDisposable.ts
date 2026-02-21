/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file observableDisposable.ts
 * @brief Provides an abstract class for disposable objects with an observable disposal state and event.
 *
 * This module introduces `ObservableDisposable`, an extension of the `Disposable` class
 * that tracks its disposal status (`disposed`) as a public property and provides an
 * `onDispose` event. This allows external components to react to the disposal of an
 * object, and also includes an assertion utility to check if an object has been disposed.
 */

import { Emitter } from './event.js';
import { Disposable, IDisposable } from './lifecycle.js';

/**
 * @abstract
 * @class ObservableDisposable
 * @extends Disposable
 * @brief Abstract base class for disposable objects that track their disposal state
 * as a public attribute and provide an event to subscribe to disposal.
 *
 * This class enhances the basic `Disposable` pattern by making the disposal status
 * observable, which is useful for managing resources and reacting to object lifecycle
 * events in complex application architectures.
 */
export abstract class ObservableDisposable extends Disposable {
	/**
	 * @private
	 * @readonly
	 * @property _onDispose
	 * @brief Private emitter for the `onDispose` event.
	 * Functional Utility: Manages the subscription and firing of dispose events.
	 */
	private readonly _onDispose = this._register(new Emitter<void>());

	/**
	 * @method onDispose
	 * @brief The event is fired when this object is disposed.
	 * Note! Executes the callback immediately if already disposed.
	 *
	 * Functional Utility: Provides a mechanism for external code to react
	 * to the object's disposal, ensuring proper cleanup or state management.
	 *
	 * @param callback The callback function to be called on disposal.
	 * @returns An `IDisposable` instance that can be used to unsubscribe from the event.
	 */
	public onDispose(callback: () => void): IDisposable {
		// Block Logic: If the object is already disposed, execute the callback immediately
		// to ensure that consumers always receive the disposal notification.
		if (this.disposed) {
			callback();

			return this; // Return itself as a no-op disposable since callback was already executed.
		}

		// Functional Utility: Registers the callback with the internal Emitter for future disposal events.
		return this._onDispose.event(callback);
	}

	/**
	 * @method addDisposable
	 * @brief Adds a disposable object to the list of disposables
	 * that will be disposed with this object.
	 *
	 * Functional Utility: Simplifies resource management by allowing related disposable
	 * objects to be automatically disposed when this object is disposed.
	 *
	 * @param disposables One or more `IDisposable` objects to be added.
	 * @returns The current `ObservableDisposable` instance for chaining.
	 */
	public addDisposable(...disposables: IDisposable[]): this {
		// Block Logic: Iterates through the provided disposables and registers each one
		// with the base `Disposable` class's registration mechanism.
		for (const disposable of disposables) {
			this._register(disposable);
		}

		return this;
	}

	/**
	 * @private
	 * @property _disposed
	 * @brief Tracks 'disposed' state of this object internally.
	 * Invariant: True if the object has been disposed, false otherwise.
	 */
	private _disposed = false;

	/**
	 * @method disposed
	 * @brief Gets current 'disposed' state of this object.
	 *
	 * @returns `true` if the object has been disposed, `false` otherwise.
	 */
	public get disposed(): boolean {
		return this._disposed;
	}

	/**
	 * @method dispose
	 * @brief Disposes the current object if not already disposed.
	 *
	 * Functional Utility: Marks the object as disposed, fires the `onDispose` event,
	 * and calls the superclass's `dispose` method to handle other registered disposables.
	 * Ensures that disposal logic is executed only once.
	 */
	public override dispose(): void {
		// Block Logic: Prevents redundant disposal operations if already disposed.
		if (this.disposed) {
			return;
		}
		this._disposed = true; // Mark the object as disposed.

		this._onDispose.fire(); // Notify all subscribers of the disposal event.
		super.dispose(); // Call the dispose method of the base `Disposable` class.
	}

	/**
	 * @method assertNotDisposed
	 * @brief Assert that the current object was not yet disposed.
	 *
	 * Functional Utility: Provides a runtime check for ensuring that operations are not
	 * performed on an already disposed object, aiding in debugging and maintaining
	 * object lifecycle integrity.
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
 * @typedef TNotDisposed
 * @brief Type for a non-disposed object `TObject`.
 *
 * This utility type can be used for type assertions to narrow down the type
 * of an `ObservableDisposable` instance to one that is guaranteed not to be disposed.
 */
type TNotDisposed<TObject extends { disposed: boolean }> = TObject & { disposed: false };

/**
 * @function assertNotDisposed
 * @brief Asserts that a provided `object` is not `disposed` yet,
 * e.g., its `disposed` property is `false`.
 *
 * Functional Utility: A standalone helper function to perform disposal assertions,
 * which can be used on any object that conforms to the `{ disposed: boolean }` interface.
 *
 * @throws if the provided `object.disposed` is `true`.
 * @param object The object to check for disposal status.
 * @param error Error message or error object to throw if assertion fails.
 */
export function assertNotDisposed<TObject extends { disposed: boolean }>(
	object: TObject,
	error: string | Error,
): asserts object is TNotDisposed<TObject> {
	// Block Logic: Checks if the object is disposed. If not, it means the assertion passes.
	if (!object.disposed) {
		return;
	}

	// Block Logic: Constructs and throws an error if the assertion fails (object is disposed).
	const errorToThrow = typeof error === 'string'
		? new Error(error)
		: error;

	throw errorToThrow;
}