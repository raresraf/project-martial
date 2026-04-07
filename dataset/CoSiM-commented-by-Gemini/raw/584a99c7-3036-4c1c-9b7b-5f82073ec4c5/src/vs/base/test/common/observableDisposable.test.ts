/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
import assert from 'assert';
import { spy } from 'sinon';
import { wait, waitRandom } from './testUtils.js';
import { randomInt } from '../../common/numbers.js';
import { Disposable, IDisposable } from '../../common/lifecycle.js';
import { ensureNoDisposablesAreLeakedInTestSuite } from './utils.js';
import { assertNotDisposed, ObservableDisposable } from '../../common/observableDisposable.js';

/**
 * @file Test suite for the `ObservableDisposable` class.
 * @description This suite verifies the functionality of `ObservableDisposable`, ensuring it correctly
 * tracks its disposed state, fires `onDispose` events, and manages hierarchies of disposable objects.
 */
suite('ObservableDisposable', () => {
	const disposables = ensureNoDisposablesAreLeakedInTestSuite();

	/**
	 * @test Verifies that the `disposed` property is correctly tracked.
	 * @description An instance of `ObservableDisposable` should initially have `disposed` as `false`,
	 * and it should be set to `true` after `dispose()` is called.
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

		assert(
			object.disposed === false,
			'Object must not be disposed yet.',
		);

		object.dispose();

		assert(
			object.disposed,
			'Object must be disposed.',
		);
	});

	/**
	 * @suite Tests for the `onDispose()` method.
	 */
	suite('• onDispose()', () => {
		/**
		 * @test Ensures the `onDispose` event fires exactly once when the object is disposed.
		 * @description It should not fire before disposal and should not fire on subsequent calls to `dispose()`.
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

			object.dispose();
			object.dispose();
			await waitRandom(10, 5);
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
		 * @test Verifies that the `onDispose` callback is executed immediately if the object is already disposed.
		 * @description This ensures that listeners are notified of disposal even if they are attached after the fact.
		 */
		test('• executes callback immediately if already disposed', async () => {
			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object = new class extends ObservableDisposable { }();
			disposables.add(object);

			// dispose object and wait for the event to be fired/received
			object.dispose();
			await wait(10);

			const onDisposeSpy = spy();
			disposables.add(object.onDispose(onDisposeSpy));

			await wait(10);

			assert(
				onDisposeSpy.calledOnce,
				'`onDispose` callback must be called immediately.',
			);

			await waitRandom(10, 5);

			disposables.add(object.onDispose(onDisposeSpy));

			await wait(10);

			assert(
				onDisposeSpy.calledTwice,
				'`onDispose` callback must be called immediately the second time.',
			);

			// dispose object and wait for the event to be fired/received
			object.dispose();
			await wait(10);

			assert(
				onDisposeSpy.calledTwice,
				'`onDispose` callback must not be called again on dispose.',
			);
		});
	});

	/**
	 * @suite Tests for the `addDisposable()` method.
	 */
	suite('• addDisposable()', () => {
		/**
		 * @test Checks that disposables added via `addDisposable` are disposed along with the parent object.
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

			object.dispose();

			// finally validate that all objects are disposed
			const allDisposed = disposableObjects.reduce((acc, disposable) => {
				return acc && disposable.disposed;
			}, true);

			assert(
				allDisposed === true,
				'Disposable object must be disposed now.',
			);
		});

		/**
		 * @test Validates that a nested tree of disposables is fully disposed when the root is disposed.
		 * @description This ensures that complex object hierarchies are cleaned up correctly.
		 */
		test('• disposes the entire tree of disposables', async () => {
			class TestDisposable extends ObservableDisposable { }

			/**
			 * Generate a tree of disposable objects.
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

			object.dispose();

			// finally validate that all objects are disposed
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
	 * @suite Tests for assertion helpers related to the disposal state.
	 */
	suite('• asserts', () => {
		/**
		 * @test Verifies that the `assertNotDisposed` method on the instance works as expected.
		 * @description It should not throw when the object is not disposed and should throw after it has been disposed.
		 */
		test('• not disposed (method)', async () => {
			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object: ObservableDisposable = new class extends ObservableDisposable { }();
			disposables.add(object);

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

			assert.throws(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

			await waitRandom(10);

			assert.throws(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});
		});

		/**
		 * @test Verifies that the standalone `assertNotDisposed` function works correctly.
		 * @description It should not throw for an undisposed object and should throw for a disposed one.
		 */
		test('• not disposed (function)', async () => {
			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object: ObservableDisposable = new class extends ObservableDisposable { }();
			disposables.add(object);

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
