/**
 * @d0e23524-683b-443f-afcb-16008e5d3b3d/test/automation/src/code.ts
 * @brief Automation controller for VS Code smoke tests, providing high-level orchestration of Electron and Browser instances.
 * Domain: Software Testing, Automation Frameworks, UI Interaction.
 * Architecture: Implements the 'Code' class as a facade over the PlaywrightDriver, handling process lifecycle, telemetry, and synchronized UI polling.
 * Functional Utility: Manages reliable spawning/teardown of application instances and exposes semantic primitives for editor, terminal, and workbench interactions.
 */

/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import * as cp from 'child_process';
import * as os from 'os';
import { IElement, ILocaleInfo, ILocalizedStrings, ILogFile } from './driver';
import { Logger, measureAndLog } from './logger';
import { launch as launchPlaywrightBrowser } from './playwrightBrowser';
import { PlaywrightDriver } from './playwrightDriver';
import { launch as launchPlaywrightElectron } from './playwrightElectron';
import { teardown } from './processes';
import { Quality } from './application';

/**
 * @brief Configuration schema for launching an automated VS Code instance.
 */
export interface LaunchOptions {
	codePath?: string;
	readonly workspacePath: string;
	userDataDir: string;
	readonly extensionsPath: string;
	readonly logger: Logger;
	logsPath: string;
	crashesPath: string;
	verbose?: boolean;
	readonly extraArgs?: string[];
	readonly remote?: boolean;
	readonly web?: boolean;
	readonly tracing?: boolean;
	snapshots?: boolean;
	readonly headless?: boolean;
	readonly browser?: 'chromium' | 'webkit' | 'firefox' | 'chromium-msedge' | 'chromium-chrome';
	readonly quality: Quality;
}

interface ICodeInstance {
	kill: () => Promise<void>;
}

// Global state tracking for active test instances.
const instances = new Set<ICodeInstance>();

/**
 * @brief Tracks a newly spawned process and handles its I/O and lifecycle events.
 */
function registerInstance(process: cp.ChildProcess, logger: Logger, type: string) {
	const instance = { kill: () => teardown(process, logger) };
	instances.add(instance);

	process.stdout?.on('data', data => logger.log(`[${type}] stdout: ${data}`));
	process.stderr?.on('data', error => logger.log(`[${type}] stderr: ${error}`));

	process.once('exit', (code, signal) => {
		logger.log(`[${type}] Process terminated (pid: ${process.pid}, code: ${code}, signal: ${signal})`);

		instances.delete(instance);
	});
}

/**
 * @brief Global cleanup routine triggered on test process termination.
 */
async function teardownAll(signal?: number) {
	stopped = true;

	for (const instance of instances) {
		await instance.kill();
	}

	if (typeof signal === 'number') {
		process.exit(signal);
	}
}

let stopped = false;
// Event Listeners: Ensure all spawned VS Code instances are reaped on parent process exit.
process.on('exit', () => teardownAll());
process.on('SIGINT', () => teardownAll(128 + 2)); 	 // https://nodejs.org/docs/v14.16.0/api/process.html#process_signal_events
process.on('SIGTERM', () => teardownAll(128 + 15)); // same as above

/**
 * @brief Primary entry point for spawning a VS Code instance for testing.
 * Logic: Branches between Browser-based (Web) and Electron-based launch strategies.
 */
export async function launch(options: LaunchOptions): Promise<Code> {
	if (stopped) {
		throw new Error('Smoke test process has terminated, refusing to spawn Code');
	}

	// Block Logic: Web/Browser execution branch.
	if (options.web) {
		const { serverProcess, driver } = await measureAndLog(() => launchPlaywrightBrowser(options), 'launch playwright (browser)', options.logger);
		registerInstance(serverProcess, options.logger, 'server');

		return new Code(driver, options.logger, serverProcess, undefined, options.quality);
	}

	// Block Logic: Native Electron execution branch.
	else {
		const { electronProcess, driver } = await measureAndLog(() => launchPlaywrightElectron(options), 'launch playwright (electron)', options.logger);
		registerInstance(electronProcess, options.logger, 'electron');

		const safeToKill = new Promise<void>(resolve => {
			process.stdout?.on('data', data => {
				// Logic: Heuristic check for the Electron 'app.quit' signal to ensure graceful exit.
				if (data.toString().includes('Lifecycle#app.on(will-quit) - calling app.quit()')) {
					setTimeout(() => resolve(), 500 /* give Electron some time to actually terminate fully */);
				}
			});
		});

		return new Code(driver, options.logger, electronProcess, safeToKill, options.quality);
	}
}

/**
 * @brief High-level controller for a running VS Code instance.
 * Architecture: Uses a Proxy for the underlying PlaywrightDriver to inject automatic logging for all interactions.
 */
export class Code {

	readonly driver: PlaywrightDriver;

	constructor(
		driver: PlaywrightDriver,
		readonly logger: Logger,
		private readonly mainProcess: cp.ChildProcess,
		private readonly safeToKill: Promise<void> | undefined,
		readonly quality: Quality
	) {
		this.driver = new Proxy(driver, {
			get(target, prop) {
				if (typeof prop === 'symbol') {
					throw new Error('Invalid usage');
				}

				const targetProp = (target as any)[prop];
				if (typeof targetProp !== 'function') {
					return targetProp;
				}

				// Functional Utility: Decorator pattern providing execution logs for every driver call.
				return function (this: any, ...args: any[]) {
					logger.log(`${prop}`, ...args.filter(a => typeof a === 'string'));
					return targetProp.apply(this, args);
				};
			}
		});
	}

	async startTracing(name: string): Promise<void> {
		return await this.driver.startTracing(name);
	}

	async stopTracing(name: string, persist: boolean): Promise<void> {
		return await this.driver.stopTracing(name, persist);
	}

	async sendKeybinding(keybinding: string, accept?: () => Promise<void> | void): Promise<void> {
		await this.driver.sendKeybinding(keybinding, accept);
	}

	async didFinishLoad(): Promise<void> {
		return this.driver.didFinishLoad();
	}

	/**
	 * @brief Executes a graceful shutdown of the application.
	 * Strategy: First attempts a clean close via the driver, then falls back to SIGTERM with retries.
	 */
	async exit(): Promise<void> {
		return measureAndLog(() => new Promise<void>(resolve => {
			const pid = this.mainProcess.pid!;

			let done = false;

			// Start the exit flow via driver
			this.driver.close();

			let safeToKill = false;
			this.safeToKill?.then(() => safeToKill = true);

			// Block Logic: Polling loop to verify process termination.
			(async () => {
				let retries = 0;
				while (!done) {
					retries++;

					if (safeToKill) {
						this.logger.log('Smoke test exit() call did not terminate the process yet, but safeToKill is true, so we can kill it');
						process.kill(pid);
					}

					switch (retries) {

						// After 10 seconds: Escalates to forceful kill.
						case 20: {
							this.logger.log('Smoke test exit() call did not terminate process after 10s, forcefully exiting the application...');
							process.kill(pid);
							break;
						}

						// After 20 seconds: Hard timeout; resolves the promise to prevent test hang.
						case 40: {
							this.logger.log('Smoke test exit() call did not terminate process after 20s, giving up');
							process.kill(pid);
							done = true;
							resolve();
						}
					}

					try {
						// Logic: Process.kill(pid, 0) checks for process existence without sending a signal.
						process.kill(pid, 0); 
						await this.wait(500);
					} catch (error) {
						this.logger.log('Smoke test exit() call terminated process successfully');
						done = true;

						resolve();
					}
				}
			})();
		}), 'Code#exit()', this.logger);
	}

	async getElement(selector: string): Promise<IElement | undefined> {
		return (await this.driver.getElements(selector))?.[0];
	}

	async getElements(selector: string, recursive: boolean): Promise<IElement[] | undefined> {
		return this.driver.getElements(selector, recursive);
	}

	/**
	 * @brief Synchronization primitive for verifying UI text content.
	 * Logic: Repeatedly queries the DOM until the target text matches or timeout occurs.
	 */
	async waitForTextContent(selector: string, textContent?: string, accept?: (result: string) => boolean, retryCount?: number): Promise<string> {
		accept = accept || (result => textContent !== undefined ? textContent === result : !!result);

		return await this.poll(
			() => this.driver.getElements(selector).then(els => els.length > 0 ? Promise.resolve(els[0].textContent) : Promise.reject(new Error('Element not found for textContent'))),
			s => accept!(typeof s === 'string' ? s : ''),
			`get text content '${selector}'`,
			retryCount
		);
	}

	async waitAndClick(selector: string, xoffset?: number, yoffset?: number, retryCount: number = 200): Promise<void> {
		await this.poll(() => this.driver.click(selector, xoffset, yoffset), () => true, `click '${selector}'`, retryCount);
	}

	async waitForSetValue(selector: string, value: string): Promise<void> {
		await this.poll(() => this.driver.setValue(selector, value), () => true, `set value '${selector}'`);
	}

	async waitForElements(selector: string, recursive: boolean, accept: (result: IElement[]) => boolean = result => result.length > 0): Promise<IElement[]> {
		return await this.poll(() => this.driver.getElements(selector, recursive), accept, `get elements '${selector}'`);
	}

	async waitForElement(selector: string, accept: (result: IElement | undefined) => boolean = result => !!result, retryCount: number = 200): Promise<IElement> {
		return await this.poll<IElement>(() => this.driver.getElements(selector).then(els => els[0]), accept, `get element '${selector}'`, retryCount);
	}

	async waitForActiveElement(selector: string, retryCount: number = 200): Promise<void> {
		await this.poll(() => this.driver.isActiveElement(selector), r => r, `is active element '${selector}'`, retryCount);
	}

	async waitForTitle(accept: (title: string) => boolean): Promise<void> {
		await this.poll(() => this.driver.getTitle(), accept, `get title`);
	}

	async waitForTypeInEditor(selector: string, text: string): Promise<void> {
		await this.poll(() => this.driver.typeInEditor(selector, text), () => true, `type in editor '${selector}'`);
	}

	async waitForEditorSelection(selector: string, accept: (selection: { selectionStart: number; selectionEnd: number }) => boolean): Promise<void> {
		await this.poll(() => this.driver.getEditorSelection(selector), accept, `get editor selection '${selector}'`);
	}

	async waitForTerminalBuffer(selector: string, accept: (result: string[]) => boolean): Promise<void> {
		await this.poll(() => this.driver.getTerminalBuffer(selector), accept, `get terminal buffer '${selector}'`);
	}

	async writeInTerminal(selector: string, value: string): Promise<void> {
		await this.poll(() => this.driver.writeInTerminal(selector, value), () => true, `writeInTerminal '${selector}'`);
	}

	async whenWorkbenchRestored(): Promise<void> {
		await this.poll(() => this.driver.whenWorkbenchRestored(), () => true, `when workbench restored`);
	}

	getLocaleInfo(): Promise<ILocaleInfo> {
		return this.driver.getLocaleInfo();
	}

	getLocalizedStrings(): Promise<ILocalizedStrings> {
		return this.driver.getLocalizedStrings();
	}

	getLogs(): Promise<ILogFile[]> {
		return this.driver.getLogs();
	}

	wait(millis: number): Promise<void> {
		return this.driver.wait(millis);
	}

	/**
	 * @brief Generic polling engine for asynchronous state verification.
	 * Invariant: Retries the 'fn' predicate until 'acceptFn' is true or retryCount is exhausted.
	 */
	private async poll<T>(
		fn: () => Promise<T>,
		acceptFn: (result: T) => boolean,
		timeoutMessage: string,
		retryCount = 200,
		retryInterval = 100 // millis
	): Promise<T> {
		let trial = 1;
		let lastError: string = '';

		while (true) {
			if (trial > retryCount) {
				this.logger.log('Timeout!');
				this.logger.log(lastError);
				this.logger.log(`Timeout: ${timeoutMessage} after ${(retryCount * retryInterval) / 1000} seconds.`);

				throw new Error(`Timeout: ${timeoutMessage} after ${(retryCount * retryInterval) / 1000} seconds.`);
			}

			let result;
			try {
				result = await fn();
				if (acceptFn(result)) {
					return result;
				} else {
					lastError = 'Did not pass accept function';
				}
			} catch (e: any) {
				lastError = Array.isArray(e.stack) ? e.stack.join(os.EOL) : e.stack;
			}

			await this.wait(retryInterval);
			trial++;
		}
	}
}

/**
 * @brief Performs a Breadth-First Search (BFS) to locate a specific element in the component tree.
 */
export function findElement(element: IElement, fn: (element: IElement) => boolean): IElement | null {
	const queue = [element];

	while (queue.length > 0) {
		const element = queue.shift()!;

		if (fn(element)) {
			return element;
		}

		queue.push(...element.children);
	}

	return null;
}

/**
 * @brief Traverses the component tree and returns all elements satisfying the predicate.
 */
export function findElements(element: IElement, fn: (element: IElement) => boolean): IElement[] {
	const result: IElement[] = [];
	const queue = [element];

	while (queue.length > 0) {
		const element = queue.shift()!;

		if (fn(element)) {
			result.push(element);
		}

		queue.push(...element.children);
	}

	return result;
}
