/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file observableDisposable.test.ts
 * @brief Test suite for the `ObservableDisposable` class and related disposal utilities.
 *
 * This file contains unit tests to verify the functionality of `ObservableDisposable`,
 * an abstract class designed to manage the lifecycle of disposable resources.
 * It ensures proper tracking of disposal state, firing of disposal events, and
 * hierarchical disposal of associated disposable objects.
 */
import assert from 'assert';
import { spy } from 'sinon';
import { wait, waitRandom } from './testUtils.js';
import { randomInt } from '../../common/numbers.js';
import { Disposable, IDisposable } from '../../common/lifecycle.js';
import { ensureNoDisposablesAreLeakedInTestSuite } from './utils.js';
import { assertNotDisposed, ObservableDisposable } from '../../common/observableDisposable.js';

suite('ObservableDisposable', () => {
	/**
	 * @brief Test suite for the `ObservableDisposable` class.
	 *
	 * This suite verifies various aspects of the `ObservableDisposable` abstract class,
	 * including its disposal state tracking, the `onDispose` event, and the `addDisposable`
	 * functionality for managing a hierarchy of disposable objects.
	 */
	const disposables = ensureNoDisposablesAreLeakedInTestSuite();

	test('• tracks `disposed` state', () => {
		/**
		 * @brief Test case: Verifies that `ObservableDisposable` correctly tracks its disposal state.
		 *
		 * This test creates an anonymous class extending `ObservableDisposable` and asserts
		 * that the `disposed` property accurately reflects the object's lifecycle
		 * (not disposed initially, then disposed after calling `dispose()`).
		 */
		// Since ObservableDisposable is an abstract class, we create an anonymous concrete class
		// that extends it to instantiate an object for testing.
		const object = new class extends ObservableDisposable { }();
		disposables.add(object); // Add to suite's disposables to ensure proper cleanup.

		// Assertions to confirm class inheritance and initial state.
		assert(
			object instanceof ObservableDisposable,
			'Object must be instance of ObservableDisposable.',
		);

		assert(
			object instanceof Disposable,
			'Object must be instance of Disposable.',
		);

		assert(
			object.disposed === false,
			'Object must not be disposed yet.',
		);

		object.dispose(); // Trigger disposal.

		assert(
			object.disposed,
			'Object must be disposed.',
		);
	});

	suite('• onDispose()', () => {
		/**
		 * @brief Test suite for the `onDispose()` method of `ObservableDisposable`.
		 *
		 * This suite verifies that the `onDispose()` event fires correctly when an
		 * `ObservableDisposable` object is disposed, and handles scenarios
		 * like multiple disposal calls and immediate callback execution for
		 * already disposed objects.
		 */
		test('• fires the event on dispose', async () => {
			/**
			 * @brief Test case: Verifies that the `onDispose` callback is fired when the object is disposed.
			 *
			 * This test checks that a registered `onDispose` callback is executed exactly once
			 * when `dispose()` is called on the `ObservableDisposable` object, even if `dispose()`
			 * is called multiple times.
			 */
			// Since ObservableDisposable is an abstract class, we create an anonymous concrete class
			// that extends it to instantiate an object for testing.
			const object = new class extends ObservableDisposable { }();
			disposables.add(object); // Add to suite's disposables for proper cleanup.

			assert(
				object.disposed === false,
				'Object must not be disposed yet.',
			);

			// Create a sinon spy to monitor the onDispose callback.
			const onDisposeSpy = spy(() => { });
			// Register the spy as a dispose callback.
			disposables.add(object.onDispose(onDisposeSpy));

			// Assert that the callback has not been called yet.
			assert(
				onDisposeSpy.notCalled,
				'`onDispose` callback must not be called yet.',
			);

			// Wait for a random short period to ensure no premature firing.
			await waitRandom(10);

			// Assert again that the callback has not been called.
			assert(
				onDisposeSpy.notCalled,
				'`onDispose` callback must not be called yet.',
			);

			// Dispose the object and wait for a microtask tick for event processing.
			object.dispose();
			await wait(1); // Small wait to allow async event to propagate.

			// Validate that the object is disposed and the callback was called exactly once.
			assert(
				object.disposed,
				'Object must be disposed.',
			);

			assert(
				onDisposeSpy.calledOnce,
				'`onDispose` callback must be called.',
			);

			// Validate that the callback is not called again on subsequent dispose calls.
			object.dispose(); // Call dispose again.
			object.dispose(); // Call dispose a third time.
			await waitRandom(10); // Wait again.
			object.dispose(); // Call dispose a fourth time.

			assert(
				onDisposeSpy.calledOnce,
				'`onDispose` callback must not be called again.',
			);

			assert(
				object.disposed,
				'Object must be disposed.',
			);
		});

		test('• executes callback immediately if already disposed', async () => {
			/**
			 * @brief Test case: Verifies that `onDispose` callbacks registered after an object is disposed are executed immediately.
			 *
			 * This test ensures that if an `ObservableDisposable` object is already disposed,
			 * any subsequent `onDispose` callbacks added to it are invoked synchronously
			 * or almost synchronously.
			 */
			// Since ObservableDisposable is an abstract class, we create an anonymous concrete class
			// that extends it to instantiate an object for testing.
			const object = new class extends ObservableDisposable { }();
			disposables.add(object); // Add to suite's disposables for proper cleanup.

			// Dispose the object first.
			object.dispose();
			await wait(1); // Small wait to allow async event to propagate fully.

			const onDisposeSpy = spy(() => { }); // Create a spy for the callback.
			// Register a callback after the object has already been disposed.
			disposables.add(object.onDispose(onDisposeSpy));

			assert(
				onDisposeSpy.calledOnce,
				'`onDispose` callback must be called immediately.',
			);

			// Wait for a random period to ensure no unexpected calls.
			await waitRandom(10, 5);

			// Register the same callback again.
			disposables.add(object.onDispose(onDisposeSpy));

			assert(
				onDisposeSpy.calledTwice,
				'`onDispose` callback must be called immediately the second time.',
			);

			// Call dispose again (should have no further effect on already called callbacks).
			object.dispose();
			await wait(1);

			assert(
				onDisposeSpy.calledTwice,
				'`onDispose` callback must not be called again on dispose.',
			);
		});
	});

	suite('• addDisposable()', () => {
		/**
		 * @brief Test suite for the `addDisposable()` method of `ObservableDisposable`.
		 *
		 * This suite verifies that disposable objects added to an `ObservableDisposable`
		 * instance are correctly disposed when the `ObservableDisposable` itself is disposed.
		 * It covers scenarios involving single-level additions and hierarchical disposal.
		 */
		test('• disposes provided object with itself', async () => {
			/**
			 * @brief Test case: Verifies that `addDisposable` correctly disposes added objects when the parent is disposed.
			 *
			 * This test defines a simple `TestDisposable` class, adds multiple instances of it
			 * to an `ObservableDisposable` object, and then asserts that all added disposables
			 * are disposed when the parent `ObservableDisposable` is disposed.
			 */
			// A simple implementation of IDisposable for testing purposes.
			class TestDisposable implements IDisposable {
				private _disposed = false;
				public get disposed() {
					return this._disposed;
				}

				public dispose(): void {
					this._disposed = true;
				}
			}

			// Since ObservableDisposable is an abstract class, we create an anonymous concrete class
			// that extends it to instantiate an object for testing.
			const object = new class extends ObservableDisposable { }();
			disposables.add(object); // Add to suite's disposables for proper cleanup.

			assert(
				object.disposed === false,
				'Object must not be disposed yet.',
			);

			// Generate a random number of TestDisposable objects.
			const disposableObjects = [];
			for (let i = 0; i < randomInt(20, 10); i++) {
				disposableObjects.push(new TestDisposable());
			}

			// Sanity check: ensure all newly created disposable objects are initially not disposed.
			for (const disposable of disposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			// Add the generated disposable objects to the ObservableDisposable parent.
			object.addDisposable(...disposableObjects);

			// Sanity check: ensure adding them does not prematurely dispose them.
			for (const disposable of disposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			object.dispose(); // Dispose the parent ObservableDisposable.

			// Finally, validate that all disposable objects added to the parent are now disposed.
			const allDisposed = disposableObjects.reduce((acc, disposable) => {
				return acc && disposable.disposed;
			}, true);

			assert(
				allDisposed === true,
				'Disposable object must be disposed now.',
			);
		});

		test('• disposes the entire tree of disposables', async () => {
			/**
			 * @brief Test case: Verifies that disposing a parent `ObservableDisposable`
			 * also disposes all its recursively added disposable children.
			 *
			 * This test generates a tree-like structure of `TestDisposable` objects
			 * (which extend `ObservableDisposable`), adds them to a root `ObservableDisposable`,
			 * and then asserts that disposing the root correctly disposes all
			 * objects in the entire hierarchy.
			 */
			class TestDisposable extends ObservableDisposable { }

			/**
			 * @brief Helper function to recursively generate a tree of disposable objects.
			 *
			 * @param count The number of disposable objects to generate at the current level.
			 * @param parent The parent `TestDisposable` to which newly created disposables will be added.
			 * @returns An array containing all generated `TestDisposable` objects (flat list).
			 */
			const disposableObjects = (
				count: number = randomInt(20, 10),
				parent: TestDisposable | null = null,
			): TestDisposable[] => {
				assert(
					count > 0,
					'Count must be greater than 0.',
				);

				const allDisposables = [];
				for (let i = 0; i < count; i++) {
					const disposableObject = new TestDisposable();
					allDisposables.push(disposableObject);
					if (parent !== null) {
						parent.addDisposable(disposableObject); // Add child to parent's disposables.
					}

					// Generate child disposable objects recursively to create a tree structure.
					const countMax = count / 2;
					const countMin = count / 5;

					if (countMin < 1) { // Base case for recursion.
						return allDisposables;
					}

					const childDisposables = disposableObjects(
						randomInt(countMax, countMin),
						disposableObject, // Current disposable becomes the parent for the next level.
					);
					allDisposables.push(...childDisposables); // Add children to the flat list.
				}

				return allDisposables;
			};

			// Since ObservableDisposable is an abstract class, we create an anonymous concrete class
			// that extends it to instantiate the root object for testing.
			const object = new class extends ObservableDisposable { }();
			disposables.add(object); // Add to suite's disposables for proper cleanup.

			assert(
				object.disposed === false,
				'Object must not be disposed yet.',
			);

			// Generate a random number for the initial count of disposables.
			const disposablesCount = randomInt(20, 10);
			// Generate the hierarchical structure of disposable objects.
			const allDisposableObjects = disposableObjects(disposablesCount, object);

			// Assert that there are indeed nested disposable objects for the test to be valid.
			assert(
				allDisposableObjects.length > disposablesCount,
				'Must have some of the nested disposable objects for this test to be valid.',
			);

			// Sanity check: ensure all generated disposable objects are initially not disposed.
			for (const disposable of allDisposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			object.dispose(); // Dispose the root ObservableDisposable, which should trigger hierarchical disposal.

			// Finally, validate that all objects in the entire hierarchy are now disposed.
			const allDisposed = allDisposableObjects.reduce((acc, disposable) => {
				return acc && disposable.disposed;
			}, true);

			assert(
				allDisposed === true,
				'Disposable object must be disposed now.',
			);
		});
	});

	suite('• asserts', () => {
		/**
		 * @brief Test suite for the assertion methods related to `ObservableDisposable`.
		 *
		 * This suite verifies the behavior of `assertNotDisposed` (both as a method
		 * and a standalone function) to ensure it correctly throws an error when an
		 * `ObservableDisposable` object has been disposed, and does not throw otherwise.
		 */
		test('• not disposed (method)', async () => {
			/**
			 * @brief Test case: Verifies the `assertNotDisposed` method on an `ObservableDisposable` instance.
			 *
			 * This test checks that `object.assertNotDisposed()` does not throw an error
			 * when the object is not disposed, and correctly throws an error after the
			 * object has been disposed.
			 */
			// Since ObservableDisposable is an abstract class, we create an anonymous concrete class
			// that extends it to instantiate an object for testing.
			const object: ObservableDisposable = new class extends ObservableDisposable { }();
			disposables.add(object); // Add to suite's disposables for proper cleanup.

			// Assert that `assertNotDisposed` does not throw when the object is not disposed.
			assert.doesNotThrow(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

			await waitRandom(10); // Wait for a random period.

			// Assert again that `assertNotDisposed` does not throw.
			assert.doesNotThrow(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

			// Dispose the object and wait for microtask queue to clear.
			object.dispose();
			await wait(1);

			// Assert that `assertNotDisposed` now throws an error after disposal.
			assert.throws(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

			await waitRandom(10); // Wait for a random period.

			// Assert again that it still throws after disposal.
			assert.throws(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});
		});

		test('• not disposed (function)', async () => {
			/**
			 * @brief Test case: Verifies the standalone `assertNotDisposed` function.
			 *
			 * This test checks that the `assertNotDisposed` function (imported externally)
			 * does not throw an error when the provided `ObservableDisposable` object
			 * is not disposed, and correctly throws an error after the object has been disposed.
			 */
			// Since ObservableDisposable is an abstract class, we create an anonymous concrete class
			// that extends it to instantiate an object for testing.
			const object: ObservableDisposable = new class extends ObservableDisposable { }();
			disposables.add(object); // Add to suite's disposables for proper cleanup.

			// Assert that `assertNotDisposed` does not throw when the object is not disposed.
			assert.doesNotThrow(() => {
				assertNotDisposed(
					object,
					'Object must not be disposed.',
				);
			});

			await waitRandom(10); // Wait for a random period.

			// Assert again that `assertNotDisposed` does not throw.
			assert.doesNotThrow(() => {
				assertNotDisposed(
					object,
					'Object must not be disposed.',
				);
			});

			// Dispose the object and wait for microtask queue to clear.
			object.dispose();
			await wait(1);

			// Assert that `assertNotDisposed` now throws an error after disposal.
			assert.throws(() => {
				assertNotDisposed(
					object,
					'Object must not be disposed.',
				);
			});

			await waitRandom(10); // Wait for a random period.

			// Assert again that it still throws after disposal.
			assert.throws(() => {
				assertNotDisposed(
					object,
					'Object must not be disposed.',
				);
			});
		});
	});
});
