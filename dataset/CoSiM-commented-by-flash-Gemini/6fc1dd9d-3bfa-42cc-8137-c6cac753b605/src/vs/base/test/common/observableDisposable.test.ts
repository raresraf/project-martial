/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @6fc1dd9d-3bfa-42cc-8137-c6cac753b605/src/vs/base/test/common/observableDisposable.test.ts
 * @brief Unit tests for the ObservableDisposable resource management pattern.
 * 
 * Functional Intent: Validates the state machine and event propagation of 
 * the ObservableDisposable class. It ensures that 'disposed' predicates, 
 * termination callbacks, and tree-based resource cleanup operate correctly 
 * under both synchronous and asynchronous conditions.
 * 
 * Domain: Platform Testing, Resource Lifecycle, Reactive Patterns.
 */

import assert from 'assert';
import { spy } from 'sinon';
import { wait, waitRandom } from './testUtils.js';
import { randomInt } from '../../common/numbers.js';
import { Disposable, IDisposable } from '../../common/lifecycle.js';
import { ensureNoDisposablesAreLeakedInTestSuite } from './utils.js';
import { assertNotDisposed, ObservableDisposable } from '../../common/observableDisposable.js';

suite('ObservableDisposable', () => {
	const disposables = ensureNoDisposablesAreLeakedInTestSuite();

	/**
	 * Block Logic: State tracking validation.
	 * Logic: Verifies the 'disposed' flag transitions correctly from false to true 
	 * upon invocation of the dispose() method.
	 */
	test('tracks `disposed` state', () => {
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

	suite('onDispose', () => {
		/**
		 * Block Logic: Asynchronous event propagation.
		 * Logic: Validates that listeners registered via 'onDispose' are 
		 * triggered exactly once during the object's termination cycle.
		 */
		test('fires the event on dispose', async () => {
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

			// Logic: Trigger disposal and verify signal receipt.
			object.dispose();
			await wait(1);

			assert(
				object.disposed,
				'Object must be disposed.',
			);

			assert(
				onDisposeSpy.calledOnce,
				'`onDispose` callback must be called.',
			);

			// Block Logic: Idempotency check.
			// Invariant: Multiple dispose() calls must not re-trigger lifecycle events.
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
		 * Block Logic: Post-termination subscription.
		 * Logic: Verifies that adding a listener to an already-disposed object 
		 * results in immediate execution of the callback (synchronous or next-tick).
		 */
		test('executes callback immediately if already disposed', async () => {
			const object = new class extends ObservableDisposable { }();
			disposables.add(object);

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

			object.dispose();
			await wait(1);

			assert(
				onDisposeSpy.calledTwice,
				'`onDispose` callback must not be called again on dispose.',
			);
		});
	});

	suite('addDisposable()', () => {
		/**
		 * Block Logic: Aggregate cleanup validation.
		 * Logic: Ensures that child resources registered via 'addDisposable' are 
		 * recursively terminated when the parent object is disposed.
		 */
		test('disposes provided object with itself', async () => {
			class TestDisposable implements IDisposable {
				private _disposed = false;
				public get disposed() {
					return this._disposed;
				}

				public dispose(): void {
					this._disposed = true;
				}
			}

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

			for (const disposable of disposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			object.addDisposable(...disposableObjects);

			for (const disposable of disposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			object.dispose();

			const allDisposed = disposableObjects.reduce((acc, disposable) => {
				return acc && disposable.disposed;
			}, true);

			assert(
				allDisposed === true,
				'Disposable object must be disposed now.',
			);
		});

		/**
		 * Block Logic: Deep tree traversal validation.
		 * Logic: Generates a random recursive tree of disposables and verifies 
		 * that top-level disposal propagates to every leaf node in the hierarchy.
		 */
		test('disposes the entire tree of disposables', async () => {
			class TestDisposable extends ObservableDisposable { }

			/**
			 * @brief Utility for procedural tree generation.
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

					// Recursive Logic: Probabilistic branching to create tree depth.
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

			for (const disposable of allDisposableObjects) {
				assert(
					disposable.disposed === false,
					'Disposable object must not be disposed yet.',
				);
			}

			object.dispose();

			const allDisposed = allDisposableObjects.reduce((acc, disposable) => {
				return acc && disposable.disposed;
			}, true);

			assert(
				allDisposed === true,
				'Disposable object must be disposed now.',
			);
		});
	});

	suite('asserts', () => {
		/**
		 * Block Logic: Invariant guard validation (method).
		 */
		test('not disposed (method)', async () => {
			const object: ObservableDisposable = new class extends ObservableDisposable { }();
			disposables.add(object);

			assert.doesNotThrow(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

			await waitRandom(10);

			assert.doesNotThrow(() => {
				object.assertNotDisposed('Object must not be disposed.');
			});

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
		 * Block Logic: Invariant guard validation (function).
		 */
		test('not disposed (function)', async () => {
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
