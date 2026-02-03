/**
 * @file localPty.ts
 * @brief Represents a local pseudo-terminal (pty) process.
 * @copyright Copyright (c) Microsoft Corporation. All rights reserved.
 * @license MIT
 *
 * This file defines the `LocalPty` class, which acts as a proxy for a pty
 * process running on the local pty host. It implements the
 * `ITerminalChildProcess` interface and forwards all pty-related operations
 * to the `IPtyService`.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { IProcessCreationResult, IProcessPropertyMap, IPtyService, ITerminalChildProcess, ITerminalLaunchError, ProcessPropertyType } from '../../../../platform/terminal/common/terminal.js';
import { BasePty } from '../common/basePty.js';

/**
 * Responsible for establishing and maintaining a connection with an existing terminal process
 * created on the local pty host.
 */
export class LocalPty extends BasePty implements ITerminalChildProcess {
	constructor(
		id: number,
		shouldPersist: boolean,
		private readonly _proxy: IPtyService
	) {
		super(id, shouldPersist);
	}

	/**
	 * @brief Starts the pty process.
	 * @return A promise that resolves with the result of the process creation.
	 */
	start(): Promise<ITerminalLaunchError | IProcessCreationResult | undefined> {
		return this._proxy.start(this.id);
	}

	/**
	 * @brief Detaches the pty process from the UI.
	 * @param forcePersist Whether to force the process to persist.
	 */
	detach(forcePersist?: boolean): Promise<void> {
		return this._proxy.detachFromProcess(this.id, forcePersist);
	}

	/**
	 * @brief Shuts down the pty process.
	 * @param immediate Whether to shut down immediately.
	 */
	shutdown(immediate: boolean): void {
		this._proxy.shutdown(this.id, immediate);
	}

	/**
	 * @brief Writes binary data to the pty process.
	 * @param data The binary data to write.
	 */
	async processBinary(data: string): Promise<void> {
		// Pre-condition: Do not process binary data during replay.
		if (this._inReplay) {
			return;
		}
		return this._proxy.processBinary(this.id, data);
	}

	/**
	 * @brief Writes data to the pty process.
	 * @param data The data to write.
	 */
	input(data: string): void {
		// Pre-condition: Do not send input during replay.
		if (this._inReplay) {
			return;
		}
		this._proxy.input(this.id, data);
	}

	/**
	 * @brief Sends a signal to the pty process.
	 * @param signal The signal to send.
	 */
	sendSignal(signal: string): void {
		// Pre-condition: Do not send signals during replay.
		if (this._inReplay) {
			return;
		}
		this._proxy.sendSignal(this.id, signal);
	}

	/**
	 * @brief Resizes the pty.
	 * @param cols The number of columns.
	 * @param rows The number of rows.
	 */
	resize(cols: number, rows: number): void {
		// Pre-condition: Do not resize during replay, and only resize if the
		// dimensions have changed.
		if (this._inReplay || this._lastDimensions.cols === cols && this._lastDimensions.rows === rows) {
			return;
		}
		this._lastDimensions.cols = cols;
		this._lastDimensions.rows = rows;
		this._proxy.resize(this.id, cols, rows);
	}

	async clearBuffer(): Promise<void> {
		this._proxy.clearBuffer?.(this.id);
	}

	freePortKillProcess(port: string): Promise<{ port: string; processId: string }> {
		if (!this._proxy.freePortKillProcess) {
			throw new Error('freePortKillProcess does not exist on the local pty service');
		}
		return this._proxy.freePortKillProcess(port);
	}

	async refreshProperty<T extends ProcessPropertyType>(type: T): Promise<IProcessPropertyMap[T]> {
		return this._proxy.refreshProperty(this.id, type);
	}

	async updateProperty<T extends ProcessPropertyType>(type: T, value: IProcessPropertyMap[T]): Promise<void> {
		return this._proxy.updateProperty(this.id, type, value);
	}

	/**
	 * @brief Acknowledges that data has been processed by the frontend.
	 * @param charCount The number of characters that have been processed.
	 *
	 * This is part of the flow control mechanism to prevent the backend from
	 * sending data too quickly.
	 */
	acknowledgeDataEvent(charCount: number): void {
		// Pre-condition: Do not acknowledge data during replay.
		if (this._inReplay) {
			return;
		}
		this._proxy.acknowledgeDataEvent(this.id, charCount);
	}

	setUnicodeVersion(version: '6' | '11'): Promise<void> {
		return this._proxy.setUnicodeVersion(this.id, version);
	}

	handleOrphanQuestion() {
		this._proxy.orphanQuestionReply(this.id);
	}
}