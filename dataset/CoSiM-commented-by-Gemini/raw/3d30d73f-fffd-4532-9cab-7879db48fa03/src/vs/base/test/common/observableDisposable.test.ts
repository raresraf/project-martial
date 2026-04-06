/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file This file contains unit tests for the `ObservableDisposable` class, which is a
 * foundational component for managing resource lifecycle and observing disposal events.
 */

import assert from 'assert';
import { spy } from 'sinon';
import { wait, waitRandom } from './testUtils.js';
import { randomInt } from '../../common/numbers.js';
import { Disposable, IDisposable } from '../../common/lifecycle.js';
import { ensureNoDisposablesAreLeakedInTestSuite } from './utils.js';
import { assertNotDisposed, ObservableDisposable } from '../../common/observableDisposable.js';

suite('ObservableDisposable', () => {
	// A test utility that ensures any disposable created in this suite is
	// automatically disposed after the tests are run, preventing leaks in the test runner.
	const disposables = ensureNoDisposablesAreLeakedInTestSuite();

	/**
	 * This test verifies the basic functionality of the `disposed` property.
	 * An `ObservableDisposable` should correctly track its own lifecycle state.
	 */
	test('• tracks `disposed` state', () => {
		// this is an abstract class, so we have to create
		// an anonymous class that extends it
		const object = new class extends ObservableDisposable { }();
		disposables.add(object);

		assert(
			object instanceof ObservableDisposable,
			'Object must be instance of ObservableDisposable.',
		);

		assert(
			object instanceof Disposable,
			'Object must be instance of Disposable.',
		);

		// Pre-condition: Check that the object is not disposed upon creation.
		assert(
			object.disposed === false,
			'Object must not be disposed yet.',
		);

		object.dispose();

		// Assert that the `disposed` property is true after `dispose()` is called.
		assert(
			object.disposed,
			'Object must be disposed.',
		);
	});

	/**
	 * This suite tests the behavior of the `onDispose` event handler, which allows
	 * code to react to the disposal of the object.
	 */
	suite('• onDispose()', () => {
		/**
		 * This test ensures that the `onDispose` event is fired exactly once when
		 * the object is disposed, and not before or on subsequent dispose calls.
		 */
		test('• fires the event on dispose', async () => {
			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object = new class extends ObservableDisposable { }();
			disposables.add(object);

			assert(
				object.disposed === false,
				'Object must not be disposed yet.',
			);

			const onDisposeSpy = spy(() => { });
			disposables.add(object.onDispose(onDisposeSpy));

			// Pre-condition: The callback should not be called synchronously.
			assert(
				onDisposeSpy.notCalled,
				'`onDispose` callback must not be called yet.',
			);

			await waitRandom(10);

			assert(
				onDisposeSpy.notCalled,
				'`onDispose` callback must not be called yet.',
			);

			// dispose object and wait for the event to be fired/received
			object.dispose();
			await wait(1);

			/**
			 * Validate that the callback was called.
			 */

			assert(
				object.disposed,
				'Object must be disposed.',
			);

			assert(
				onDisposeSpy.calledOnce,
				'`onDispose` callback must be called.',
			);

			/**
			 * Validate that the callback is not called again.
			 */
			// Invariant: Subsequent calls to dispose should have no effect.
			object.dispose();
			object.dispose();
			await waitRandom(10);
			object.dispose();

			assert(
				onDisposeSpy.calledOnce,
				'`onDispose` callback must not be called again.',
			);

			assert(
				object.disposed,
				'Object must be disposed.',
			);
		});

		/**
		 * This test verifies a critical behavior for robust cleanup: if a listener
		 * is attached to `onDispose` *after* the object is already disposed, the
		 * listener should be executed immediately.
		 */
		test('• executes callback immediately if already disposed', async () => {
			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object = new class extends ObservableDisposable { }();
			disposables.add(object);

			// dispose object and wait for the event to be fired/received
			object.dispose();
			await wait(1);

			const onDisposeSpy = spy(() => { });
			disposables.add(object.onDispose(onDisposeSpy));

			assert(
				onDisposeSpy.calledOnce,
				'`onDispose` callback must be called immediately.',
			);

			await waitRandom(10, 5);

			// Attaching another listener should also result in immediate execution.
			disposables.add(object.onDispose(onDisposeSpy));

			assert(
				onDisposeSpy.calledTwice,
				'`onDispose` callback must be called immediately the second time.',
			);

			// dispose object and wait for the event to be fired/received
			object.dispose();
			await wait(1);

			assert(
				onDisposeSpy.calledTwice,
				'`onDispose` callback must not be called again on dispose.',
			);
		});
	});

	/**
	 * This suite tests the `addDisposable` method, which is used to build
	 * chains or trees of disposable objects.
	 */
	suite('• addDisposable()', () => {
		/**
		 * This test ensures that when a parent `ObservableDisposable` is disposed,
		 * all child disposables added via `addDisposable` are also disposed.
		 */
		test('• disposes provided object with itself', async () => {
			class TestDisposable implements IDisposable {
				private _disposed = false;
				public get disposed() {
					return this._disposed;
				}

				public dispose(): void {
					this._disposed = true;
				}
			}

			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object = new class extends ObservableDisposable { }();
			disposables.add(object);

			assert(
				object.disposed === false,
				'Object must not be disposed yet.',
			);

			const disposableObjects = [];
			for (let i = 0; i < randomInt(20, 10); i++) {
				disposableObjects.push(new TestDisposable());
			}

			// a sanity check for the initial state of the objects
			for (const disposable of disposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			object.addDisposable(...disposableObjects);

			// a sanity check after the 'addDisposable' call
			for (const disposable of disposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			// Action: Dispose the parent object.
			object.dispose();

			// finally validate that all child objects are also disposed.
			const allDisposed = disposableObjects.reduce((acc, disposable) => {
				return acc && disposable.disposed;
			}, true);

			assert(
				allDisposed === true,
				'Disposable object must be disposed now.',
			);
		});

		/**
		 * This test verifies that disposal cascades down through an entire nested
		 * tree of disposable objects, ensuring that complex object graphs are
		 * cleaned up correctly.
		 */
		test('• disposes the entire tree of disposables', async () => {
			class TestDisposable extends ObservableDisposable { }

			/**
			 * Helper function to generate a random tree of disposable objects.
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
						parent.addDisposable(disposableObject);
					}

					// generate child disposable objects recursively
					// to create a tree structure
					const countMax = count / 2;
					const countMin = count / 5;

					if (countMin < 1) {
						return allDisposables;
					}

					const childDisposables = disposableObjects(
						randomInt(countMax, countMin),
						disposableObject,
					);
					allDisposables.push(...childDisposables);
				}

				return allDisposables;
			};

			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object = new class extends ObservableDisposable { }();
			disposables.add(object);

			assert(
				object.disposed === false,
				'Object must not be disposed yet.',
			);

			// Build the tree of disposables attached to the root object.
			const disposablesCount = randomInt(20, 10);
			const allDisposableObjects = disposableObjects(disposablesCount, object);

			assert(
				allDisposableObjects.length > disposablesCount,
				'Must have some of the nested disposable objects for this test to be valid.',
			);

			// a sanity check for the initial state of the objects
			for (const disposable of allDisposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			// Action: Dispose the root of the tree.
			object.dispose();

			// finally validate that all objects in the entire tree are disposed.
			const allDisposed = allDisposableObjects.reduce((acc, disposable) => {
				return acc && disposable.disposed;
			}, true);

			assert(
				allDisposed === true,
				'Disposable object must be disposed now.',
			);
		});
	});

	/**
	 * This suite tests the assertion helpers that guard against operations on
	 * already-disposed objects.
	 */
	suite('• asserts', () => {
		/**
		 * Tests the `assertNotDisposed()` instance method.
		 */
		test('• not disposed (method)', async () => {
			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object: ObservableDisposable = new class extends ObservableDisposable { }();
			disposables.add(object);

			// Should not throw when the object is alive.
			assert.doesNotThrow(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

			await waitRandom(10);

			assert.doesNotThrow(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

			// dispose object and wait for the event to be fired/received
			object.dispose();
			await wait(1);

			// Should throw an error after the object has been disposed.
			assert.throws(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

			await waitRandom(10);

			assert.throws(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});
		});

		/**
		 * Tests the `assertNotDisposed()` standalone utility function.
		 */
		test('• not disposed (function)', async () => {
			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object: ObservableDisposable = new class extends ObservableDisposable { }();
			disposables.add(object);

			// Should not throw when the object is alive.
			assert.doesNotThrow(() => {
				assertNotDisposed(
					object,
					'Object must not be disposed.',
				);
			});

			await waitRandom(10);

			assert.doesNotThrow(() => {
				assertNotDisposed(
					object,
					'Object must not be disposed.',
				);
			});

			// dispose object and wait for the event to be fired/received
			object.dispose();
			await wait(1);

			// Should throw an error after the object has been disposed.
			assert.throws(() => {
				assertNotDisposed(
					object,
					'Object must not be disposed.',
				);
			});

			await waitRandom(10);

			assert.throws(() => {
				assertNotDisposed(
					object,
					'Object must not be disposed.',
				);
			});
		});
	});
});
