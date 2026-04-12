/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @fileoverview
 * @raw/d471bcd7-7d39-4801-822b-412d24e4ed88/src/vs/base/test/common/observableDisposable.test.ts
 * @brief This file contains the test suite for the `ObservableDisposable` class.
 * The tests verify the correct behavior of the disposable pattern, including
 * state tracking, event firing on disposal, and management of a tree of
 * disposable objects.
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
	 * @purpose Ensures that no disposable objects are leaked during the test suite execution.
	 * @invariant All disposables created in the suite must be added to this container
	 * to be automatically disposed of after each test.
	 */
	const disposables = ensureNoDisposablesAreLeakedInTestSuite();

	/**
	 * @name tracks `disposed` state
	 * @brief Verifies that the `disposed` property of an `ObservableDisposable`
	 * instance is correctly updated upon calling the `dispose` method.
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

		/**
		 * @precondition The object should not be disposed initially.
		 */
		assert(
			object.disposed === false,
			'Object must not be disposed yet.',
		);

		object.dispose();

		/**
		 * @postcondition The object's `disposed` property should be true after disposal.
		 */
		assert(
			object.disposed,
			'Object must be disposed.',
		);
	});

	/**
	 * @name onDispose()
	 * @brief Test suite for the `onDispose` method, which allows registering
	 * callbacks to be executed when the object is disposed.
	 */
	suite('• onDispose()', () => {
		/**
		 * @name fires the event on dispose
		 * @brief Verifies that the `onDispose` event is fired exactly once when the
		 * object is disposed.
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

			/**
			 * @invariant The `onDispose` callback should not be called before
			 * the object is disposed.
			 */
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
			 * @purpose Validate that the callback was called after disposal.
			 * @invariant The callback should be called exactly once.
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
			 * @purpose Validate that subsequent calls to `dispose` do not trigger the
			 * callback again.
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
		 * @name executes callback immediately if already disposed
		 * @brief Verifies that if `onDispose` is called on an already disposed
		 * object, the callback is executed immediately.
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

			/**
			 * @invariant The callback should be called immediately since the object
			 * is already disposed.
			 */
			assert(
				onDisposeSpy.calledOnce,
				'`onDispose` callback must be called immediately.',
			);

			await waitRandom(10, 5);

			disposables.add(object.onDispose(onDisposeSpy));

			assert(
				onDisposeSpy.calledTwice,
				'`onDispose` callback must be called immediately the second time.',
			);

			// dispose object and wait for the event to be fired/received
			object.dispose();
			await wait(1);

			/**
			 * @invariant Calling `dispose` again should not re-trigger the callbacks.
			 */
			assert(
				onDisposeSpy.calledTwice,
				'`onDispose` callback must not be called again on dispose.',
			);
		});
	});

	/**
	 * @name addDisposable()
	 * @brief Test suite for the `addDisposable` method, which is used to register
	 * other disposable objects that should be disposed along with the parent.
	 */
	suite('• addDisposable()', () => {
		/**
		 * @name disposes provided object with itself
		 * @brief Verifies that disposable objects added via `addDisposable` are
		 * disposed when the parent object is disposed.
		 */
		test('• disposes provided object with itself', async () => {
			/**
			 * @purpose A simple disposable class for testing purposes.
			 */
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

			/**
			 * @purpose Generates a random number of disposable objects for the test.
			 */
			const disposableObjects = [];
			for (let i = 0; i < randomInt(20, 10); i++) {
				disposableObjects.push(new TestDisposable());
			}

			/**
			 * @precondition All generated disposable objects should not be disposed initially.
			 */
			// a sanity check for the initial state of the objects
			for (const disposable of disposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			object.addDisposable(...disposableObjects);

			/**
			 * @invariant After adding to the parent, the child disposables should
			 * still not be disposed.
			 */
			// a sanity check after the 'addDisposable' call
			for (const disposable of disposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			object.dispose();

			/**
			 * @postcondition After the parent is disposed, all child disposables
			 * must also be disposed.
			 */
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
		 * @name disposes the entire tree of disposables
		 * @brief Verifies that a nested tree of disposable objects is fully disposed
		 * when the root object is disposed.
		 */
		test('• disposes the entire tree of disposables', async () => {
			class TestDisposable extends ObservableDisposable { }

			/**
			 * @purpose Recursively generates a tree of disposable objects.
			 * @param count The number of disposables to create at the current level.
			 * @param parent The parent disposable to which the new disposables will be added.
			 * @returns An array containing all disposables in the generated subtree.
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

					/**
					 * @purpose Block Logic: Recursively generate child disposables to create a tree structure.
					 * The number of children is randomly determined within a range.
					 */
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

			/**
			 * @precondition All disposable objects in the tree must not be disposed initially.
			 */
			// a sanity check for the initial state of the objects
			for (const disposable of allDisposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			object.dispose();

			/**
			 * @postcondition After disposing the root, all nodes in the disposable tree
			 * must be disposed.
			 */
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
	 * @name asserts
	 * @brief Test suite for assertion helpers related to the disposed state.
	 */
	suite('• asserts', () => {
		/**
		 * @name not disposed (method)
		 * @brief Verifies that the `assertNotDisposed` method on an `ObservableDisposable`
		 * instance throws an error only after the object has been disposed.
		 */
		test('• not disposed (method)', async () => {
			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object: ObservableDisposable = new class extends ObservableDisposable { }();
			disposables.add(object);

			/**
			 * @invariant The assertion should not throw before the object is disposed.
			 */
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

			/**
			 * @invariant The assertion must throw after the object has been disposed.
			 */
			assert.throws(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

			await waitRandom(10);

			assert.throws(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});
		});

		/**
		 * @name not disposed (function)
		 * @brief Verifies that the standalone `assertNotDisposed` function
		 * throws an error only after the given object has been disposed.
		 */
		test('• not disposed (function)', async () => {
			// this is an abstract class, so we have to create
			// an anonymous class that extends it
			const object: ObservableDisposable = new class extends ObservableDisposable { }();
			disposables.add(object);

			/**
			 * @invariant The assertion should not throw before the object is disposed.
			 */
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

			/**
			 * @invariant The assertion must throw after the object has been disposed.
			 */
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
