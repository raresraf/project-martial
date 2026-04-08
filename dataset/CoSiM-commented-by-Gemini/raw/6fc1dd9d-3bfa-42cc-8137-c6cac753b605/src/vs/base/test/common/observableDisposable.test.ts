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
 * @suite Test suite for the ObservableDisposable class.
 * @description This suite verifies the core functionalities of the `ObservableDisposable` class,
 * which is an abstract class designed to be extended. It combines the `Disposable` pattern
 * with an observable `onDispose` event, allowing for robust and predictable resource management.
 * The tests cover state tracking, event emission, and the cascading disposal of child disposables.
 */
suite('ObservableDisposable', () => {
	const disposables = ensureNoDisposablesAreLeakedInTestSuite();

	/**
	 * @test Verifies that the `disposed` property is correctly tracked.
	 * @description This test ensures that an instance of `ObservableDisposable` starts with
	 * `disposed` as false and correctly sets it to true after `dispose()` is called.
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
	 * @suite Tests for the `onDispose` method.
	 * @description This suite validates the observable behavior of the `ObservableDisposable`,
	 * ensuring that callbacks are correctly fired upon disposal.
	 */
	suite('• onDispose()', () => {
		/**
		 * @test Verifies that the `onDispose` event is fired exactly once when the object is disposed.
		 * @description This test ensures that a registered callback is not called prematurely, is called
		 * once upon the first `dispose()` call, and is not called again on subsequent `dispose()` calls.
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
		 * @test Verifies that a callback is executed immediately if registered after the object has already been disposed.
		 * @description This is a critical feature for ensuring cleanup logic runs reliably, even if it's
		 * registered late in the object's lifecycle.
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

			await waitRandom(10);

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
	 * @suite Tests for the `addDisposable` method.
	 * @description This suite ensures that the `ObservableDisposable` can manage the lifecycle of other
	 * disposable objects, disposing of them automatically when it is itself disposed.
	 */
	suite('• addDisposable()', () => {
		/**
		 * @test Verifies that child disposables are disposed along with the parent.
		 * @description This test checks the fundamental contract of `addDisposable`: any object added
		 * to the parent should be disposed when the parent is disposed.
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
		 * @test Verifies that a recursively nested tree of disposables is fully disposed.
		 * @description This test builds a complex tree of `ObservableDisposable` objects and ensures
		 * that calling `dispose()` on the root object triggers a cascading disposal of all
		 * descendants, no matter how deeply they are nested.
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
	 * @suite Tests for assertion helpers.
	 * @description This suite validates the `assertNotDisposed` guard, which is used to prevent
	 * use-after-free errors by throwing an exception if an action is attempted on a disposed object.
	 */
	suite('• asserts', () => {
		/**
		 * @test Verifies the behavior of the `assertNotDisposed` instance method.
		 * @description It should not throw before `dispose()` is called and should always throw after.
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
		 * @test Verifies the behavior of the standalone `assertNotDisposed` utility function.
		 * @description It should behave identically to the instance method, throwing only after the object is disposed.
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
