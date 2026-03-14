/**
 * @file This file defines a generic `ObjectStream` class that can create a readable
 * stream from a generator of objects. It is designed to work within an environment
 * that uses streams and cancellation tokens, likely for handling data flows in an
 * editor or similar application.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { ITextModel } from '../../model.js';
import { VSBuffer } from '../../../../base/common/buffer.js';
import { assert, assertNever } from '../../../../base/common/assert.js';
import { CancellationToken } from '../../../../base/common/cancellation.js';
import { ObservableDisposable } from '../../../../base/common/observableDisposable.js';
import { newWriteableStream, WriteableStream, ReadableStream } from '../../../../base/common/stream.js';

/**
 * A readable stream of provided objects. This class takes a generator of objects
 * and pushes them into a stream in chunks, respecting backpressure and cancellation.
 * @template T The type of object in the stream.
 */
export class ObjectStream<T extends object> extends ObservableDisposable implements ReadableStream<T> {
	/**
	 * Flag that indicates whether the stream has ended.
	 */
	private ended: boolean = false;

	/**
	 * Underlying writable stream instance that we push data to.
	 */
	private readonly stream: WriteableStream<T>;

	/**
	 * Interval reference that is used to periodically send
	 * objects to the stream in the background.
	 */
	private timeoutHandle: ReturnType<typeof setTimeout> | undefined;

	/**
	 * Creates an instance of ObjectStream.
	 * @param data A generator that yields the objects to be streamed.
	 * @param cancellationToken An optional token to signal cancellation of the stream.
	 */
	constructor(
		private readonly data: Generator<T, undefined>,
		private readonly cancellationToken?: CancellationToken,
	) {
		super();

		this.stream = newWriteableStream<T>(null);

		if (cancellationToken?.isCancellationRequested) {
			this.end();
			return;
		}

		// send a first batch of data immediately
		this.send(true);
	}

	/**
	 * Starts the process of sending data to the stream. This method is called
	 * recursively using `setTimeout` to send data in chunks.
	 *
	 * @param stopAfterFirstSend - Whether to continue sending data to the stream
	 *             or stop sending after the first batch of data is sent instead.
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
				if (this.cancellationToken?.isCancellationRequested) {
					this.end();

					return;
				}

				if (this.ended) {
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
	 * Stops the data sending loop by clearing the timeout.
	 * @returns The current `ObjectStream` instance.
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
	 * Sends a specified number of objects from the generator to the underlying stream.
	 * @param objectsCount The number of objects to send in this batch. Defaults to 25.
	 * @private
	 */
	private async sendData(
		objectsCount: number = 25,
	): Promise<void> {
		// send up to 'objectsCount' objects at a time
		while (objectsCount > 0) {
			try {
				const next = this.data.next();
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
	 * Ends the stream and stops sending data objects. It's safe to call this multiple times.
	 * @private
	 * @returns The current `ObjectStream` instance.
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
	 * Pauses the stream by stopping the data sending loop and pausing the underlying stream.
	 */
	public pause(): void {
		this.stopStream();
		this.stream.pause();

		return;
	}

	/**
	 * Resumes the stream by restarting the data sending loop and resuming the underlying stream.
	 */
	public resume(): void {
		this.send();
		this.stream.resume();

		return;
	}

	/**
	 * Destroys the stream, ensuring all resources are cleaned up.
	 */
	public destroy(): void {
		this.dispose();
	}

	/**
	 * Removes a listener for a given event from the underlying stream.
	 */
	public removeListener(event: string, callback: (...args: any[]) => void): void {
		this.stream.removeListener(event, callback);

		return;
	}

	/**
	 * Listens for events on the stream.
	 * @param event The event to listen for ('data', 'error', or 'end').
	 * @param callback The function to execute when the event is emitted.
	 */
	public on(event: 'data', callback: (data: T) => void): void;
	public on(event: 'error', callback: (err: Error) => void): void;
	public on(event: 'end', callback: () => void): void;
	public on(event: 'data' | 'error' | 'end', callback: (...args: any[]) => void): void {
		if (event === 'data') {
			this.stream.on(event, callback);
			// this is the convention of the readable stream, - when
			// the `data` event is registered, the stream is started
			this.send();

			return;
		}

		if (event === 'error') {
			this.stream.on(event, callback);
			return;
		}

		if (event === 'end') {
			this.stream.on(event, callback);
			return;
		}

		assertNever(
			event,
			`Unexpected event name '${event}'.`,
		);
	}

	/**
	 * Cleans up the send interval and destroys the stream.
	 */
	public override dispose(): void {
		this.stopStream();
		this.stream.destroy();

		super.dispose();
	}

	/**
	 * Creates a new instance of the stream from a provided array.
	 * @param array The array of objects to stream.
	 * @param cancellationToken An optional token to signal cancellation.
	 * @returns A new `ObjectStream` instance.
	 */
	public static fromArray<T extends object>(
		array: T[],
		cancellationToken?: CancellationToken,
	): ObjectStream<T> {
		return new ObjectStream(arrayToGenerator(array), cancellationToken);
	}

	/**
	 * Create new instance of the stream from a provided text model.
	 * Each line and EOL is emitted as a separate `VSBuffer`.
	 * @param model The text model to stream.
	 * @param cancellationToken An optional token to signal cancellation.
	 * @returns A new `ObjectStream` instance for `VSBuffer` objects.
	 */
	public static fromTextModel(
		model: ITextModel,
		cancellationToken?: CancellationToken,
	): ObjectStream<VSBuffer> {
		return new ObjectStream(modelToGenerator(model), cancellationToken);
	}
}

/**
 * Creates a generator from a provided array.
 * @param array The array to convert into a generator.
 * @returns A generator that yields each item from the array.
 */
export const arrayToGenerator = <T extends NonNullable<unknown>>(array: T[]): Generator<T, undefined> => {
	return (function* (): Generator<T, undefined> {
		for (const item of array) {
			yield item;
		}
	})();
};

/**
 * Creates a generator from a provided text model. This generator yields each line
 * and each End-Of-Line sequence as a separate `VSBuffer`.
 * @param model The `ITextModel` to read from.
 * @returns A generator that yields buffers for each line and EOL.
 */
export const modelToGenerator = (model: ITextModel): Generator<VSBuffer, undefined> => {
	return (function* (): Generator<VSBuffer, undefined> {
		const totalLines = model.getLineCount();
		let currentLine = 1;

		while (currentLine <= totalLines) {
			if (model.isDisposed()) {
				return undefined;
			}

			yield VSBuffer.fromString(
				model.getLineContent(currentLine),
			);
			if (currentLine !== totalLines) {
				yield VSBuffer.fromString(
					model.getEOL(),
				);
			}

			currentLine++;
		}
	})();
};
