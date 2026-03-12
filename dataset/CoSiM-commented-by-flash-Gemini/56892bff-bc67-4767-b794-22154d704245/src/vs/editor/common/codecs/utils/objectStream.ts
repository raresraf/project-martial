/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @module objectStream
 * @description
 * This module defines the `ObjectStream` class, a generic readable stream designed
 * for efficiently delivering objects in chunks. It's built upon the `WriteableStream`
 * abstraction and supports various data sources like generators, arrays, and text models.
 *
 * The `ObjectStream` is particularly useful in scenarios requiring asynchronous data
 * processing and cancellation, such as loading large datasets, processing file contents,
 * or managing data flow in UI-intensive applications where responsiveness is critical.
 * It provides mechanisms for pausing, resuming, and destroying the stream, as well as
 * event-based communication for data, errors, and stream completion.
 */

import { ITextModel } from '../../model.js';
import { VSBuffer } from '../../../../base/common/buffer.js';
import { assertNever } from '../../../../base/common/assert.js';
import { CancellationToken } from '../../../../base/common/cancellation.js';
import { ObservableDisposable } from '../../../../base/common/observableDisposable.js';
import { newWriteableStream, WriteableStream, ReadableStream } from '../../../../base/common/stream.js';

/**
 * A readable stream of provided objects.
 * This class implements a readable stream capable of delivering objects
 * in a chunked, asynchronous, and cancelable manner. It leverages an
 * underlying `WriteableStream` for actual data transmission and a generator
 * for fetching data from various sources.
 * @template T The type of objects that will be streamed.
 */
export class ObjectStream<T extends object> extends ObservableDisposable implements ReadableStream<T> {
	/**
	 * Flag that indicates whether the stream has ended.
	 * Once `true`, no more data will be sent.
	 */
	private ended: boolean = false;

	/**
	 * Underlying writable stream instance. All objects are written to this
	 * stream, and `ReadableStream` events (`data`, `error`, `end`) are proxied
	 * from it.
	 */
	private readonly stream: WriteableStream<T>;

	/**
	 * Interval reference that is used to periodically send
	 * objects to the stream in the background. Used with `setTimeout`
	 * to prevent blocking the event loop.
	 */
	private timeoutHandle: ReturnType<typeof setTimeout> | undefined;

	/**
	 * The source of the objects, provided as a generator.
	 * Objects are pulled from this generator and pushed into the stream.
	 */
	private readonly data: Generator<T, undefined>;
	
	/**
	 * An optional cancellation token to abort the stream operation.
	 * If the token is already canceled upon construction, the stream will immediately end.
	 */
	private readonly cancellationToken?: CancellationToken;

	constructor(
		data: Generator<T, undefined>,
		cancellationToken?: CancellationToken,
	) {
		super();

		this.data = data;
		this.cancellationToken = cancellationToken;

		this.stream = newWriteableStream<T>(null);

		// If cancellation is requested early, end the stream immediately.
		if (cancellationToken?.isCancellationRequested) {
			this.end();
			return;
		}

		// Send a first batch of data immediately to start populating the stream.
		this.send(true);
	}

	/**
	 * Starts or continues the process of sending data objects to the stream.
	 * This method can be called periodically by `setTimeout` to ensure non-blocking
	 * data delivery. It checks for cancellation and stream end state before proceeding.
	 *
	 * @param stopAfterFirstSend Optional. If `true`, the stream will stop sending data
	 *                           after the initial batch. Defaults to `false`, allowing
	 *                           continuous background sending.
	 */
	public send(
		stopAfterFirstSend: boolean = false,
	): void {
		// This method can be called asynchronously by the `setTimeout` utility below, hence
		// the state of the cancellation token or the stream itself might have changed by that time
		if (this.cancellationToken?.isCancellationRequested || this.ended) {
			this.end();

			return;
		}

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

				// Schedule the next batch of data to be sent after a short delay
				this.timeoutHandle = setTimeout(this.send.bind(this));
			})
			.catch((error) => {
				// If an error occurs during data sending, propagate it to the stream and dispose resources.
				this.stream.error(error);
				this.dispose();
			});
	}

	/**
	 * Stop the data sending loop.
	 * Clears any pending `setTimeout` calls that would trigger `send()`.
	 * @returns {this} The current `ObjectStream` instance for chaining.
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
	 * Sends a provided number of objects from the internal generator to the stream.
	 * This is a private helper method called by `send()`.
	 * @param objectsCount The maximum number of objects to send in this batch. Defaults to 25.
	 * @returns {Promise<void>} A promise that resolves when the batch is sent or rejects on error.
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
	 * Ends the stream and stops sending data objects.
	 * This method signals the underlying writable stream to end and cleans up
	 * any scheduled sending operations.
	 * @returns {this} The current `ObjectStream` instance for chaining.
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
		// Stop the background data sending and pause the underlying stream.
		this.stopStream();
		this.stream.pause();

		return;
	}

	public resume(): void {
		// Resume the underlying stream and restart background data sending.
		this.send();
		this.stream.resume();

		return;
	}

	public destroy(): void {
		// Delegates to the dispose method for comprehensive cleanup.
		this.dispose();
	}

	/**
	 * Removes a previously registered listener for the specified event.
	 * Delegates to the underlying `WriteableStream`'s `removeListener` method.
	 * @param event The name of the event to stop listening to ('data', 'error', 'end').
	 * @param callback The function that was registered as a listener.
	 */
	public removeListener(event: string, callback: (...args: any[]) => void): void {
		this.stream.removeListener(event, callback);

		return;
	}

	/**
	 * Registers a listener for a specific stream event.
	 * Supported events are 'data', 'error', and 'end'.
	 * Registering a 'data' event listener implicitly starts the stream if it's not already running.
	 * @param event The name of the event to listen for ('data', 'error', 'end').
	 * @param callback The function to call when the event occurs.
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

		// Inline: Handles unexpected event names, ensuring type safety and catching potential bugs.
		assertNever(
			event,
			`Unexpected event name '${event}'.`,
		);
	}

	/**
	 * Cleanup send interval and destroy the stream.
	 * This method overrides the `dispose` method from `ObservableDisposable`
	 * to ensure all resources related to the stream are properly released.
	 */
	public override dispose(): void {
		this.stopStream();
		this.stream.destroy();

		super.dispose();
	}

	/**
	 * Creates a new instance of `ObjectStream` from a provided array.
	 * This is a convenience factory method for easily streaming array elements.
	 * @template T The type of objects in the array.
	 * @param array The array of objects to stream.
	 * @param cancellationToken An optional cancellation token to abort the stream.
	 * @returns A new `ObjectStream` instance.
	 */
	public static fromArray<T extends object>(
		array: T[],
		cancellationToken?: CancellationToken,
	): ObjectStream<T> {
		return new ObjectStream(arrayToGenerator(array), cancellationToken);
	}

	/**
	 * Creates a new instance of `ObjectStream` from a provided `ITextModel`.
	 * This is useful for streaming lines of text from a text model, typically
	 * used in text editor contexts. Each line is converted into a `VSBuffer`.
	 * @param model The `ITextModel` to stream.
	 * @param cancellationToken An optional cancellation token to abort the stream.
	 * @returns A new `ObjectStream` instance that streams `VSBuffer` objects.
	 */
	public static fromTextModel(
		model: ITextModel,
		cancellationToken?: CancellationToken,
	): ObjectStream<VSBuffer> {
		return new ObjectStream(modelToGenerator(model), cancellationToken);
	}
}

/**
 * Creates a generator function from a given array.
 * This generator yields each item of the array sequentially.
 * @template T The type of items in the array.
 * @param array The input array to convert into a generator.
 * @returns A generator that yields elements of the array.
 */
export const arrayToGenerator = <T extends NonNullable<unknown>>(array: T[]): Generator<T, undefined> => {
	return (function* (): Generator<T, undefined> {
		for (const item of array) {
			yield item;
		}
	})();
};

/**
 * Creates a generator function from a provided `ITextModel`.
 * This generator yields each line of the text model as a `VSBuffer` object,
 * including the End-Of-Line (EOL) characters between lines.
 * It also handles potential disposal of the text model during generation.
 * @param model The `ITextModel` to convert into a generator.
 * @returns A generator that yields `VSBuffer` objects representing lines and EOLs.
 */
export const modelToGenerator = (model: ITextModel): Generator<VSBuffer, undefined> => {
	return (function* (): Generator<VSBuffer, undefined> {
		const totalLines = model.getLineCount();
		let currentLine = 1;

		while (currentLine <= totalLines) {
			// Inline: Check if the model has been disposed to prevent errors during iteration.
			if (model.isDisposed()) {
				return undefined;
			}

			// Inline: Yield the content of the current line as a VSBuffer.
			yield VSBuffer.fromString(
				model.getLineContent(currentLine),
			);
			// Inline: If it's not the last line, yield the End-Of-Line character(s) as a VSBuffer.
			if (currentLine !== totalLines) {
				yield VSBuffer.fromString(
					model.getEOL(),
				);
			}

			currentLine++;
		}
	})();
};
