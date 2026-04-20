/**
 * @aea06e12-94ba-42a2-b13a-53020acc3173/src/vs/editor/common/codecs/utils/objectStream.ts
 * @brief Asynchronous stream adapter for ECMAScript generators.
 * 
 * Functional Intent: Bridges the gap between push-based ReadableStream consumers 
 * and pull-based Generators. It manages the lifecycle of asynchronous data 
 * propagation, providing backpressure-aware buffering, pausing/resuming 
 * capabilities, and integrated cancellation support.
 * 
 * Domain: Reactive Programming, Stream Processing, VS Code Core.
 */

import { ITextModel } from '../../model.js';
import { VSBuffer } from '../../../../base/common/buffer.js';
import { assert, assertNever } from '../../../../base/common/assert.js';
import { CancellationToken } from '../../../../base/common/cancellation.js';
import { ObservableDisposable } from '../../../../base/common/observableDisposable.js';
import { newWriteableStream, WriteableStream, ReadableStream } from '../../../../base/common/stream.js';

/**
 * @brief High-level readable stream that emits objects produced by a Generator.
 * 
 * Functional Utility: Decouples object generation from consumption by providing 
 * an event-driven stream interface over synchronous or asynchronous data sources.
 */
export class ObjectStream<T extends object> extends ObservableDisposable implements ReadableStream<T> {
	private ended: boolean = false;
	private readonly stream: WriteableStream<T>;
	private timeoutHandle: ReturnType<typeof setTimeout> | undefined;

	constructor(
		private readonly data: Generator<T, undefined>,
		private readonly cancellationToken?: CancellationToken,
	) {
		super();
		this.stream = newWriteableStream<T>(null);

		// Block Logic: Early termination check.
		if (cancellationToken?.isCancellationRequested) {
			this.end();
			return;
		}

		// Initial pump: Triggers data flow on construction.
		this.send(true);
	}

	/**
	 * send - Orchestrates asynchronous data batches.
	 * @param stopAfterFirstSend If true, suspends the pump after a single successful batch.
	 * 
	 * Block Logic: Flow control and scheduling.
	 * Logic: 
	 * 1. Validates cancellation and stream state.
	 * 2. Invokes the data loader.
	 * 3. Schedules the next execution slice via microtask (setTimeout) to prevent 
	 *    long-running loops from starving the UI thread.
	 */
	public send(
		stopAfterFirstSend: boolean = false,
	): void {
		if (this.cancellationToken?.isCancellationRequested) {
			this.end();
			return;
		}

		assert(
			this.ended === false,
			'Cannot send on already ended stream.',
		);

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

				this.timeoutHandle = setTimeout(this.send.bind(this));
			})
			.catch((error) => {
				this.stream.error(error);
				this.dispose();
			});
	}

	/**
	 * @brief Suspends the background scheduling loop.
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
	 * sendData - Iterates the source generator to fill the internal buffer.
	 * @param objectsCount Maximum number of yields allowed per execution slice.
	 * 
	 * Block Logic: Internal consumption loop.
	 * Invariant: Maintains 'objectsCount' as a fairness bound to ensure 
	 * cooperative multitasking.
	 */
	private async sendData(
		objectsCount: number = 25,
	): Promise<void> {
		while (objectsCount > 0) {
			try {
				const next = this.data.next();
				// Block Logic: Source exhaustion handling.
				if (next.done || this.cancellationToken?.isCancellationRequested) {
					this.end();
					return;
				}

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
	 * @brief Finalizes the stream and shuts down the data pump.
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

	public pause(): void {
		this.stopStream();
		this.stream.pause();
	}

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
	 * on - Event listener registration with auto-trigger.
	 * Logic: Overrides standard 'on' to initiate the data pump immediately 
	 * when a 'data' listener is attached, ensuring a pull-on-demand model.
	 */
	public on(event: 'data', callback: (data: T) => void): void;
	public on(event: 'error', callback: (err: Error) => void): void;
	public on(event: 'end', callback: () => void): void;
	public on(event: 'data' | 'error' | 'end', callback: (...args: any[]) => void): void {
		if (event === 'data') {
			this.stream.on(event, callback);
			this.send();
			return;
		}

		if (event === 'error' || event === 'end') {
			this.stream.on(event, callback);
			return;
		}

		assertNever(event, `Unexpected event name '${event}'.`);
	}

	public override dispose(): void {
		this.stopStream();
		this.stream.destroy();
		super.dispose();
	}

	/**
	 * @brief Specialized factory for array-based sources.
	 */
	public static fromArray<T extends object>(
		array: T[],
		cancellationToken?: CancellationToken,
	): ObjectStream<T> {
		return new ObjectStream(arrayToGenerator(array), cancellationToken);
	}

	/**
	 * @brief High-level factory for streaming VS Code models line-by-line.
	 */
	public static fromTextModel(
		model: ITextModel,
		cancellationToken?: CancellationToken,
	): ObjectStream<VSBuffer> {
		return new ObjectStream(modelToGenerator(model), cancellationToken);
	}
}

/**
 * @brief Utility: Standard iterator-to-generator converter.
 */
export const arrayToGenerator = <T extends NonNullable<unknown>>(array: T[]): Generator<T, undefined> => {
	return (function* (): Generator<T, undefined> {
		for (const item of array) {
			yield item;
		}
	})();
};

/**
 * modelToGenerator - Adapts an ITextModel to a generator interface.
 * Logic: Iteratively yields document lines and their respective EOL markers, 
 * respecting the model's disposal state to prevent stale access.
 */
export const modelToGenerator = (model: ITextModel): Generator<VSBuffer, undefined> => {
	return (function* (): Generator<VSBuffer, undefined> {
		const totalLines = model.getLineCount();
		let currentLine = 1;

		while (currentLine <= totalLines) {
			if (model.isDisposed()) {
				return undefined;
			}

			yield VSBuffer.fromString(model.getLineContent(currentLine));
			
			if (currentLine !== totalLines) {
				yield VSBuffer.fromString(model.getEOL());
			}

			currentLine++;
		}
	})();
};
