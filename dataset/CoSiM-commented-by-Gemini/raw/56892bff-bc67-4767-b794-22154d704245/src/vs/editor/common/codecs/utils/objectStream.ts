/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file This file defines the `ObjectStream` class, a utility for converting a data source,
 * represented by a `Generator`, into a `ReadableStream`. Its primary purpose is to enable
 * non-blocking, chunked, and cancellable streaming of objects, which is essential for
 * handling potentially large datasets without freezing the UI or main thread.
 */

import { ITextModel } from '../../model.js';
import { VSBuffer } from '../../../../base/common/buffer.js';
import { assertNever } from '../../../../base/common/assert.js';
import { CancellationToken } from '../../../../base/common/cancellation.js';
import { ObservableDisposable } from '../../../../base/common/observableDisposable.js';
import { newWriteableStream, WriteableStream, ReadableStream } from '../../../../base/common/stream.js';

/**
 * A readable stream of objects from a `Generator`. This class acts as an adapter,
 * taking a synchronous generator and exposing it as an asynchronous, chunked stream
 * that respects cancellation and can be paused or resumed.
 */
export class ObjectStream<T extends object> extends ObservableDisposable implements ReadableStream<T> {
	/**
	 * Flag that indicates whether the stream has been ended and should not produce more data.
	 */
	private ended: boolean = false;

	/**
	 * The underlying writable stream that this class pushes data into. Consumers will
	 * listen on the readable end of this stream.
	 */
	private readonly stream: WriteableStream<T>;

	/**
	 * A handle for the `setTimeout` call used for scheduling the next chunk of data.
	 * This is the mechanism that makes the stream asynchronous and non-blocking.
	 */
	private timeoutHandle: ReturnType<typeof setTimeout> | undefined;

	constructor(
		private readonly data: Generator<T, undefined>,
		private readonly cancellationToken?: CancellationToken,
	) {
		super();

		this.stream = newWriteableStream<T>(data => data);

		// Immediately end if cancellation has already been requested.
		if (cancellationToken?.isCancellationRequested) {
			this.end();
			return;
		}

		// Proactively send the first batch of data to make the stream responsive.
		this.send(true);
	}

	/**
	 * Starts the process of sending data to the stream. This method schedules itself
	 * to run asynchronously, creating a non-blocking data pump.
	 *
	 * @param stopAfterFirstSend If true, the stream will send one batch and then stop,
	 * waiting to be manually resumed. If false, it will continuously send data.
	 */
	public send(
		stopAfterFirstSend: boolean = false,
	): void {
		// Before proceeding, check if the stream has been cancelled or ended.
		if (this.cancellationToken?.isCancellationRequested || this.ended) {
			this.end();
			return;
		}

		this.sendData()
			.then(() => {
				// Re-check state after the async sendData operation.
				if (this.cancellationToken?.isCancellationRequested || this.ended) {
					this.end();
					return;
				}

				if (stopAfterFirstSend === true) {
					this.stopStream();
					return;
				}

				// Schedule the next call to `send`, yielding to the event loop.
				this.timeoutHandle = setTimeout(() => this.send());
			})
			.catch((error) => {
				this.stream.error(error);
				this.dispose();
			});
	}

	/**
	 * Stops the automatic data sending loop by clearing the scheduled timeout.
	 */
	public stopStream(): this {
		if (this.timeoutHandle === undefined) {
			return this;
		}

		clearTimeout(this.timeoutHandle);
		delete this.timeoutHandle;

		return this;
	}

	/**
	 * Sends a batch of objects from the generator to the underlying stream.
	 * @param objectsCount The maximum number of objects to send in this batch.
	 */
	private async sendData(
		objectsCount: number = 25,
	): Promise<void> {
		// Send up to 'objectsCount' objects at a time.
		while (objectsCount > 0) {
			try {
				const next = this.data.next();
				// If the generator is done or cancellation is requested, end the stream.
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
	 * Ends the stream, preventing any more data from being sent.
	 * This is an idempotent operation.
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
	 * Attaches an event listener. Following standard stream conventions, attaching a 'data'
	 * listener will automatically start the flow of data.
	 */
	public on(event: 'data', callback: (data: T) => void): void;
	public on(event: 'error', callback: (err: Error) => void): void;
	public on(event: 'end', callback: () => void): void;
	public on(event: 'data' | 'error' | 'end', callback: (...args: any[]) => void): void {
		switch (event) {
			case 'data':
				this.stream.on(event, callback);
				// Standard stream behavior: start flowing data when a listener is attached.
				this.send();
				break;
			case 'error':
			case 'end':
				this.stream.on(event, callback);
				break;
			default:
				assertNever(
					event,
					`Unexpected event name '${event}'.`,
				);
		}
	}

	/**
	 * Cleans up resources by stopping the send interval and destroying the stream.
	 */
	public override dispose(): void {
		this.stopStream();
		this.stream.destroy();

		super.dispose();
	}

	/**
	 * A factory method to create a new `ObjectStream` from an array.
	 */
	public static fromArray<T extends object>(
		array: T[],
		cancellationToken?: CancellationToken,
	): ObjectStream<T> {
		return new ObjectStream(arrayToGenerator(array), cancellationToken);
	}

	/**
	* A factory method to create a new `ObjectStream` from a VS Code `ITextModel`,
	* streaming its content line by line as buffers.
	 */
	public static fromTextModel(
		model: ITextModel,
		cancellationToken?: CancellationToken,
	): ObjectStream<VSBuffer> {
		return new ObjectStream(modelToGenerator(model), cancellationToken);
	}
}

/**
 * A helper function to create a generator from an array.
 */
export const arrayToGenerator = <T extends NonNullable<unknown>>(array: T[]): Generator<T, undefined> => {
	return (function* (): Generator<T, undefined> {
		for (const item of array) {
			yield item;
		}
	})();
};

/**
 * A helper function to create a generator from an `ITextModel`. It yields each
 * line and its corresponding EOL sequence as separate `VSBuffer` objects.
 */
export const modelToGenerator = (model: ITextModel): Generator<VSBuffer, undefined> => {
	return (function* (): Generator<VSBuffer, undefined> {
		const totalLines = model.getLineCount();
		let currentLine = 1;

		while (currentLine <= totalLines) {
			if (model.isDisposed()) {
				return undefined;
			}

			// Yield the content of the current line.
			yield VSBuffer.fromString(
				model.getLineContent(currentLine),
			);
			// Yield the end-of-line sequence, if it's not the last line.
			if (currentLine !== totalLines) {
				yield VSBuffer.fromString(
					model.getEOL(),
				);
			}

			currentLine++;
		}
	})();
};
