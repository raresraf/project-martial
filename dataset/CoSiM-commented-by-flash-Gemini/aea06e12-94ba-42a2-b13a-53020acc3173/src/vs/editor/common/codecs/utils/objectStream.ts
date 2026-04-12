/**
 * @file objectStream.ts
 * @brief Asynchronous stream wrapper for generator-driven object propagation.
 * @details Implements a bridge between ECMAScript Generators and VS Code's ReadableStream 
 * interface. Supports flow control (pause/resume), lifecycle management (disposal), 
 * and cancellation-aware iteration.
 * 
 * Domain: Platform Utility, Stream Processing, Resource Management.
 */

import { ITextModel } from '../../model.js';
import { VSBuffer } from '../../../../base/common/buffer.js';
import { assert, assertNever } from '../../../../base/common/assert.js';
import { CancellationToken } from '../../../../base/common/cancellation.js';
import { ObservableDisposable } from '../../../../base/common/observableDisposable.js';
import { newWriteableStream, WriteableStream, ReadableStream } from '../../../../base/common/stream.js';

/**
 * @class ObjectStream
 * @brief A high-level readable stream that emits objects produced by a Generator.
 * Functional Utility: Decouples object generation from consumption by providing 
 * an event-driven stream interface over synchronous or asynchronous data sources.
 */
export class ObjectStream<T extends object> extends ObservableDisposable implements ReadableStream<T> {
	/**
	 * State: Indicates if the stream reached its terminal state.
	 */
	private ended: boolean = false;

	/**
	 * Buffer: Internal writable sink that handles the backpressure and event emission.
	 */
	private readonly stream: WriteableStream<T>;

	/**
	 * Scheduling: Handle for the background data pumping task.
	 */
	private timeoutHandle: ReturnType<typeof setTimeout> | undefined;

	constructor(
		private readonly data: Generator<T, undefined>,
		private readonly cancellationToken?: CancellationToken,
	) {
		super();

		// Initialization: Creates a standard writable stream to act as the primary buffer.
		this.stream = newWriteableStream<T>(null);

		// Protocol: Immediate termination if the token is pre-cancelled.
		if (cancellationToken?.isCancellationRequested) {
			this.end();
			return;
		}

		// Execution: Triggers the first batch of data emission asynchronously.
		this.send(true);
	}

	/**
	 * @brief Primary data pump orchestrator.
	 * @param stopAfterFirstSend Flag to control the iterative nature of the pump.
	 * Logic: Executes a data dispatch and schedules the next iteration via microtask/timer.
	 */
	public send(
		stopAfterFirstSend: boolean = false,
	): void {
		if (this.cancellationToken?.isCancellationRequested) {
			this.end();
			return;
		}

		// Pre-condition: Operations are only valid on active streams.
		assert(
			this.ended === false,
			'Cannot send on already ended stream.',
		);

		/**
		 * Functional Utility: Asynchronous batch processing.
		 * Logic: Pumps a slice of data and then decides whether to schedule 
		 * further iterations based on the configuration and stream state.
		 */
		this.sendData()
			.then(() => {
				if (this.cancellationToken?.isCancellationRequested || this.ended) {
					this.end();
					return;
				}

				if (stopAfterFirstSend === true) {
					this.stopStream();
					return;
				}

				// Iteration: Defers the next batch to prevent blocking the event loop.
				this.timeoutHandle = setTimeout(this.send.bind(this));
			})
			.catch((error) => {
				// Exception Handling: Propagates errors to the stream and cleans up resources.
				this.stream.error(error);
				this.dispose();
			});
	}

	/**
	 * @brief Suspends the background data pumping loop.
	 */
	public stopStream(): this {
		if (this.timeoutHandle === undefined) {
			return this;
		}

		clearTimeout(this.timeoutHandle);
		this.timeoutHandle = undefined;

		return this;
	}

	/**
	 * @brief Iterates the generator to populate the internal stream buffer.
	 * @param objectsCount Bound on the number of iterations per batch to ensure fairness.
	 * Invariant: Consumes up to 'objectsCount' elements unless the source is exhausted.
	 */
	private async sendData(
		objectsCount: number = 25,
	): Promise<void> {
		while (objectsCount > 0) {
			try {
				const next = this.data.next();
				// Condition: Check for generator completion or external cancellation.
				if (next.done || this.cancellationToken?.isCancellationRequested) {
					this.end();
					return;
				}

				// Dispatch: Writes the generated value into the stream buffer.
				await this.stream.write(next.value);
				objectsCount--;
			} catch (error) {
				this.stream.error(error);
				this.dispose();
				return;
			}
		}
	}

	/**
	 * @brief Finalizes the stream state.
	 * Functional Utility: Transition to 'ended' state, stopping all background activity.
	 */
	private end(): this {
		if (this.ended) {
			return this;
		}
		this.ended = true;

		this.stopStream();
		this.stream.end();
		return this;
	}

	/**
	 * @brief Implements ReadableStream.pause().
	 */
	public pause(): void {
		this.stopStream();
		this.stream.pause();
	}

	/**
	 * @brief Implements ReadableStream.resume().
	 */
	public resume(): void {
		this.send();
		this.stream.resume();
	}

	public destroy(): void {
		this.dispose();
	}

	public removeListener(event: string, callback: (...args: any[]) => void): void {
		this.stream.removeListener(event, callback);
	}

	/**
	 * @brief Event registration proxy.
	 * Logic: Standard NodeJS-style stream event handling. Registering 'data' 
	 * automatically triggers stream emission.
	 */
	public on(event: 'data', callback: (data: T) => void): void;
	public on(event: 'error', callback: (err: Error) => void): void;
	public on(event: 'end', callback: () => void): void;
	public on(event: 'data' | 'error' | 'end', callback: (...args: any[]) => void): void {
		if (event === 'data') {
			this.stream.on(event, callback);
			// Trigger: Stream starts on first data listener registration.
			this.send();
			return;
		}

		if (event === 'error' || event === 'end') {
			this.stream.on(event, callback);
			return;
		}

		assertNever(event, `Unexpected event name '${event}'.`);
	}

	/**
	 * @brief Comprehensive resource teardown.
	 */
	public override dispose(): void {
		this.stopStream();
		this.stream.destroy();
		super.dispose();
	}

	/**
	 * @brief Factory: Creates a stream from a static array.
	 */
	public static fromArray<T extends object>(
		array: T[],
		cancellationToken?: CancellationToken,
	): ObjectStream<T> {
		return new ObjectStream(arrayToGenerator(array), cancellationToken);
	}

	/**
	 * @brief Factory: Creates a stream from a VS Code TextModel.
	 * Architecture: Provides line-by-line streaming of document content.
	 */
	public static fromTextModel(
		model: ITextModel,
		cancellationToken?: CancellationToken,
	): ObjectStream<VSBuffer> {
		return new ObjectStream(modelToGenerator(model), cancellationToken);
	}
}

/**
 * @brief Helper: Converts an Array into a Generator.
 */
export const arrayToGenerator = <T extends NonNullable<unknown>>(array: T[]): Generator<T, undefined> => {
	return (function* (): Generator<T, undefined> {
		for (const item of array) {
			yield item;
		}
	})();
};

/**
 * @brief Helper: Converts a TextModel into a line-by-line Generator.
 * Logic: Iterates through document lines and EOL sequences to reconstruct content.
 */
export const modelToGenerator = (model: ITextModel): Generator<VSBuffer, undefined> => {
	return (function* (): Generator<VSBuffer, undefined> {
		const totalLines = model.getLineCount();
		let currentLine = 1;

		while (currentLine <= totalLines) {
			// Constraint: Termination if the model is closed during iteration.
			if (model.isDisposed()) {
				return undefined;
			}

			// Yield: Line content.
			yield VSBuffer.fromString(model.getLineContent(currentLine));
			
			// Yield: Line terminator (except for the final line).
			if (currentLine !== totalLines) {
				yield VSBuffer.fromString(model.getEOL());
			}

			currentLine++;
		}
	})();
};
