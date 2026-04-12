/**
 * @file This file is the entry point for launching and controlling instances of VS Code
 * for smoke testing. It provides a high-level `Code` class that acts as a driver for
 * interacting with the application, whether it's running as a desktop Electron app
 * or in a web browser.
 *
 * It manages the lifecycle of the VS Code processes and uses Playwright for low-level
 * browser and Electron automation.
 */

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
 * Defines the options for launching a VS Code instance for testing.
 */
export interface LaunchOptions {
	/**
	 * The path to the VS Code executable or server script.
	 */
	codePath?: string;
	/**
	 * The path to the workspace to open.
	 */
	readonly workspacePath: string;
	/**
	 * The path to the user data directory.
	 */
	userDataDir: string;
	/**
	 * The path to the extensions directory.
	 */
	readonly extensionsPath: string;
	/**
	 * A logger instance for recording test output.
	 */
	readonly logger: Logger;
	/**
	 * The path where log files should be stored.
	 */
	logsPath: string;
	/**
	 * The path where crash dumps should be stored.
	 */
	crashesPath: string;
	/**
	 * Enable verbose logging.
	 */
	verbose?: boolean;
	/**
	 * Extra command-line arguments to pass to the VS Code executable.
	 */
	readonly extraArgs?: string[];
	/**
	 * Whether to launch in remote mode.
	 */
	readonly remote?: boolean;
	/**
	 * Whether to launch the web version of VS Code.
	 */
	readonly web?: boolean;
	/**
	 * Whether to enable Playwright's tracing capabilities.
	 */
	readonly tracing?: boolean;
	/**
	 * Whether to capture snapshots on failure.
	 */
	snapshots?: boolean;
	/**
	 * Whether to run the browser in headless mode.
	 */
	readonly headless?: boolean;
	/**
	 * The specific browser to use for web tests.
	 */
	readonly browser?: 'chromium' | 'webkit' | 'firefox' | 'chromium-msedge' | 'chromium-chrome';
	/**
	 * The quality of the VS Code build (e.g., 'stable', 'insider').
	 */
	readonly quality: Quality;
}

/**
 * Interface representing a launched VS Code instance for lifecycle management.
 */
interface ICodeInstance {
	kill: () => Promise<void>;
}

// A set to keep track of all running VS Code child processes.
const instances = new Set<ICodeInstance>();

/**
 * Registers a child process to be tracked for cleanup. It logs stdout/stderr
 * and ensures the process is removed from the tracking set upon exit.
 * @param process The child process to register.
 * @param logger The logger for output.
 * @param type A string identifier for the process type (e.g., 'server', 'electron').
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
 * Tears down all registered VS Code instances. This is hooked into the process
 * exit events to ensure no orphaned processes are left.
 * @param signal The exit signal code, if any.
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

// Set up global process listeners to ensure cleanup on exit.
let stopped = false;
process.on('exit', () => teardownAll());
process.on('SIGINT', () => teardownAll(128 + 2)); 	 // Standard exit code for SIGINT
process.on('SIGTERM', () => teardownAll(128 + 15)); // Standard exit code for SIGTERM

/**
 * Launches an instance of VS Code (web or Electron) and returns a `Code` driver object.
 * @param options The launch options.
 * @returns A Promise that resolves to a `Code` instance for interacting with the application.
 */
export async function launch(options: LaunchOptions): Promise<Code> {
	if (stopped) {
		throw new Error('Smoke test process has terminated, refusing to spawn Code');
	}

	// Architectural Choice: Branch between launching a web server for browser-based
	// tests or a standalone Electron application.
	if (options.web) {
		const { serverProcess, driver } = await measureAndLog(() => launchPlaywrightBrowser(options), 'launch playwright (browser)', options.logger);
		registerInstance(serverProcess, options.logger, 'server');
		return new Code(driver, options.logger, serverProcess, undefined, options.quality);
	} else {
		const { electronProcess, driver } = await measureAndLog(() => launchPlaywrightElectron(options), 'launch playwright (electron)', options.logger);
		registerInstance(electronProcess, options.logger, 'electron');

		// A promise that resolves when it's safe to forcefully kill the Electron process.
		const safeToKill = new Promise<void>(resolve => {
			process.stdout?.on('data', data => {
				if (data.toString().includes('Lifecycle#app.on(will-quit) - calling app.quit()')) {
					setTimeout(() => resolve(), 500 /* give Electron some time to actually terminate fully */);
				}
			});
		});

		return new Code(driver, options.logger, electronProcess, safeToKill, options.quality);
	}
}

/**
 * The main class for interacting with a running instance of VS Code.
 * It provides a high-level API for actions like clicking elements, typing text,
 * and waiting for UI state changes.
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
		/**
		 * Architectural Pattern: Use a JavaScript Proxy to intercept all calls to the
		 * underlying driver. This allows for transparent logging of every action
		 * performed during the test, which is invaluable for debugging.
		 */
		this.driver = new Proxy(driver, {
			get(target, prop) {
				if (typeof prop === 'symbol') {
					throw new Error('Invalid usage');
				}

				const targetProp = (target as any)[prop];
				if (typeof targetProp !== 'function') {
					return targetProp;
				}

				// Wrap the original driver function to add logging.
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
	 * Exits the application gracefully with a robust, multi-stage shutdown sequence.
	 */
	async exit(): Promise<void> {
		return measureAndLog(() => new Promise<void>(resolve => {
			const pid = this.mainProcess.pid!;
			let done = false;

			// 1. Attempt a graceful shutdown via the driver's close command.
			this.driver.close();

			let safeToKill = false;
			this.safeToKill?.then(() => safeToKill = true);

			// 2. Poll for process termination and escalate to forceful killing if needed.
			(async () => {
				let retries = 0;
				while (!done) {
					retries++;

					if (safeToKill) {
						this.logger.log('Smoke test exit() call did not terminate the process yet, but safeToKill is true, so we can kill it');
						process.kill(pid);
					}

					switch (retries) {
						// 3. After 10 seconds, forcefully kill the process.
						case 20: {
							this.logger.log('Smoke test exit() call did not terminate process after 10s, forcefully exiting the application...');
							process.kill(pid);
							break;
						}
						// 4. After 20 seconds, give up and resolve the promise.
						case 40: {
							this.logger.log('Smoke test exit() call did not terminate process after 20s, giving up');
							process.kill(pid);
							done = true;
							resolve();
						}
					}

					try {
						process.kill(pid, 0); // Throws an exception if the process doesn't exist.
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
	 * A generic polling utility function. It repeatedly executes an async function
	 * until its result is satisfactory or a timeout is reached. This is the foundation
	 * for most `waitFor...` methods in this class.
	 *
	 * @param fn The async function to execute in each attempt.
	 * @param acceptFn A function that evaluates the result of `fn` and returns true if it's acceptable.
	 * @param timeoutMessage A message to display on timeout.
	 * @param retryCount The maximum number of retries.
	 * @param retryInterval The interval in milliseconds between retries.
	 * @returns A Promise that resolves with the first acceptable result.
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
 * Helper function to find the first element in a UI tree that satisfies a predicate.
 * Uses a Breadth-First Search (BFS) traversal.
 * @param element The root element to start the search from.
 * @param fn The predicate function to apply to each element.
 * @returns The first matching element, or null if not found.
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
 * Helper function to find all elements in a UI tree that satisfy a predicate.
 * Uses a Breadth-First Search (BFS) traversal.
 * @param element The root element to start the search from.
 * @param fn The predicate function to apply to each element.
 * @returns An array of all matching elements.
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
