/**
 * @file process.ts
 * @brief Common interface definitions for the process diagnostic service in VS Code.
 * 
 * Architectural Intent: Defines the contracts for cross-process communication regarding 
 * system resource usage, process hierarchies, and performance metrics.
 * 
 * Data Integrity: Interfaces are structured to ensure JSON-serializability across 
 * IPC boundaries (e.g., converting complex objects like Colors into string literals).
 */

import { ProcessItem } from '../../../base/common/processes.js';
import { IRemoteDiagnosticError, PerformanceInfo, SystemInfo } from '../../diagnostics/common/diagnostics.js';
import { createDecorator } from '../../instantiation/common/instantiation.js';

/**
 * @interface WindowStyles
 * @brief Visual aesthetic parameters for process-related dialogs/windows.
 */
export interface WindowStyles {
	backgroundColor?: string;
	color?: string;
}

export interface WindowData {
	styles: WindowStyles;
	zoomLevel: number;
}

/**
 * @enum IssueSource
 * @brief Categorizes the origin of a reported diagnostic issue or performance regression.
 */
export enum IssueSource {
	VSCode = 'vscode',
	Extension = 'extension',
	Marketplace = 'marketplace'
}

export interface ISettingSearchResult {
	extensionId: string;
	key: string;
	score: number;
}

export const IProcessService = createDecorator<IProcessService>('processService');

/**
 * @interface IResolvedProcessInformation
 * @brief Aggregated diagnostic data containing process maps and remote diagnostics.
 */
export interface IResolvedProcessInformation {
	readonly pidToNames: [number, string][];
	readonly processes: {
		readonly name: string;
		readonly rootProcess: ProcessItem | IRemoteDiagnosticError;
	}[];
}

/**
 * @interface IProcessService
 * @brief Service contract for retrieving system-level process information.
 */
export interface IProcessService {

	readonly _serviceBrand: undefined;

	/**
	 * @brief Collects process tree snapshots from local and remote agents.
	 * @returns A promise resolving to a map of PIDs and names.
	 */
	resolveProcesses(): Promise<IResolvedProcessInformation>;

	/**
	 * @brief Retrieves the human-readable system health status.
	 */
	getSystemStatus(): Promise<string>;

	/**
	 * @brief Obtains structured system metadata (OS, Memory, CPUs).
	 */
	getSystemInfo(): Promise<SystemInfo>;

	/**
	 * @brief Gathers real-time performance telemetry.
	 */
	getPerformanceInfo(): Promise<PerformanceInfo>;
}
