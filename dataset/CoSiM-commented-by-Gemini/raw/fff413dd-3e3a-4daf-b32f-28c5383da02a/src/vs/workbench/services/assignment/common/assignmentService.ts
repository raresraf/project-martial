/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file assignmentService.ts
 * @brief Integration layer for the Microsoft Targeted Assignment Service (TAS) within the VS Code Workbench.
 * * This module facilitates A/B testing and feature flagging by connecting the TAS client 
 * with VS Code's internal telemetry, configuration, and persistent storage systems.
 * It manages the lifecycle of experiment assignments and ensures that "treatments" (variations)
 * are applied according to user settings and environment constraints.
 */

import { localize } from '../../../../nls.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import type { IKeyValueStorage, IExperimentationTelemetry } from 'tas-client-umd';
import { MementoObject, Memento } from '../../../common/memento.js';
import { ITelemetryService } from '../../../../platform/telemetry/common/telemetry.js';
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';
import { ITelemetryData } from '../../../../base/common/actions.js';
import { InstantiationType, registerSingleton } from '../../../../platform/instantiation/common/extensions.js';
import { IConfigurationService } from '../../../../platform/configuration/common/configuration.js';
import { IProductService } from '../../../../platform/product/common/productService.js';
import { IAssignmentService } from '../../../../platform/assignment/common/assignment.js';
import { Registry } from '../../../../platform/registry/common/platform.js';
import { BaseAssignmentService } from '../../../../platform/assignment/common/assignmentService.js';
import { workbenchConfigurationNodeBase } from '../../../common/configuration.js';
import { IConfigurationRegistry, Extensions as ConfigurationExtensions, ConfigurationScope } from '../../../../platform/configuration/common/configurationRegistry.js';
import { IWorkbenchEnvironmentService } from '../../environment/common/environmentService.js';

export const IWorkbenchAssignmentService = createDecorator<IWorkbenchAssignmentService>('WorkbenchAssignmentService');

/**
 * @interface IWorkbenchAssignmentService
 * @description Extends the core assignment service to provide workbench-specific experiment querying.
 */
export interface IWorkbenchAssignmentService extends IAssignmentService {
	/**
	 * @returns A promise resolving to the list of current experiment IDs (assignment context).
	 */
	getCurrentExperiments(): Promise<string[] | undefined>;
}

/**
 * @class MementoKeyValueStorage
 * @description An adapter class that satisfies the TAS client's IKeyValueStorage interface
 * by wrapping VS Code's Memento-based persistent storage.
 */
class MementoKeyValueStorage implements IKeyValueStorage {
	private mementoObj: MementoObject;
	
	/**
	 * @constructor
	 * Scopes storage to the Application level and Machine target to ensure 
	 * experiment state persists across workspace changes but stays local to the device.
	 */
	constructor(private memento: Memento) {
		this._mementoObj = memento.getMemento(StorageScope.APPLICATION, StorageTarget.MACHINE);
	}

	async getValue<T>(key: string, defaultValue?: T | undefined): Promise<T | undefined> {
		const value = await this._mementoObj[key];
		return value || defaultValue;
	}

	setValue<T>(key: string, value: T): void {
		this._mementoObj[key] = value;
		this.memento.saveMemento();
	}
}

/**
 * @class WorkbenchAssignmentServiceTelemetry
 * @description Bridge between the TAS client's telemetry requirements and the Workbench telemetry pipeline.
 * Manages the propagation of assignment context and experimental event logging.
 */
class WorkbenchAssignmentServiceTelemetry implements IExperimentationTelemetry {
	private _lastAssignmentContext: string | undefined;
	
	constructor(
		private telemetryService: ITelemetryService,
		private productService: IProductService
	) { }

	/**
	 * @property assignmentContext
	 * Parses the semicolon-delimited string of active experiment identifiers into an array.
	 */
	get assignmentContext(): string[] | undefined {
		return this._lastAssignmentContext?.split(';');
	}

	/**
	 * Maps TAS shared properties to VS Code telemetry properties.
	 * Specifically tracks the assignment context property defined in the product configuration.
	 */
	// __GDPR__COMMON__ "abexp.assignmentcontext" : { "classification": "SystemMetaData", "purpose": "FeatureInsight" }
	setSharedProperty(name: string, value: string): void {
		if (name === this.productService.tasConfig?.assignmentContextTelemetryPropertyName) {
			this._lastAssignmentContext = value;
		}

		this.telemetryService.setExperimentProperty(name, value);
	}

	/**
	 * Transforms Map-based event data from the TAS client into ITelemetryData for standard logging.
	 */
	postEvent(eventName: string, props: Map<string, string>): void {
		const data: ITelemetryData = {};
		for (const [key, value] of props.entries()) {
			data[key] = value;
		}

		/* __GDPR__
			"query-expfeature" : {
				"owner": "sbatten",
				"comment": "Logs queries to the experiment service by feature for metric calculations",
				"ABExp.queriedFeature": { "classification": "SystemMetaData", "purpose": "FeatureInsight", "comment": "The experimental feature being queried" }
			}
		*/
		this.telemetryService.publicLog(eventName, data);
	}
}

/**
 * @class WorkbenchAssignmentService
 * @description Concrete implementation of the assignment service for the VS Code Workbench.
 * Orchestrates the initialization of the TAS client and enforces workbench-level opt-out policies.
 */
export class WorkbenchAssignmentService extends BaseAssignmentService {

	constructor(
		@ITelemetryService private telemetryService: ITelemetryService,
		@IStorageService storageService: IStorageService,
		@IConfigurationService configurationService: IConfigurationService,
		@IProductService productService: IProductService,
		@IWorkbenchEnvironmentService environmentService: IWorkbenchEnvironmentService
	) {

		super(
			telemetryService.machineId,
			configurationService,
			productService,
			environmentService,
			new WorkbenchAssignmentServiceTelemetry(telemetryService, productService),
			new MementoKeyValueStorage(new Memento('experiment.service.memento', storageService))
		);
	}

	/**
	 * @property experimentsEnabled
	 * Evaluates a set of environmental and configuration constraints to determine if 
	 * external experiment fetching should be permitted.
	 * * Pre-conditions for disabling experiments:
	 * 1. Environment-level disable flag is set.
	 * 2. Currently running extension tests.
	 * 3. Smoke test driver is active.
	 * 4. User-facing configuration 'workbench.enableExperiments' is false.
	 */
	protected override get experimentsEnabled(): boolean {
		return !this.environmentService.disableExperiments &&
			!this.environmentService.extensionTestsLocationURI &&
			!(this.environmentService as IWorkbenchEnvironmentService).enableSmokeTestDriver &&
			this.configurationService.getValue('workbench.enableExperiments') === true;
	}

	/**
	 * Retrieves a treatment value for a given experiment name and logs the access 
	 * to telemetry for observability and metric calculation.
	 */
	override async getTreatment<T extends string | number | boolean>(name: string): Promise<T | undefined> {
		const result = await super.getTreatment<T>(name);
		type TASClientReadTreatmentData = {
			treatmentName: string;
			treatmentValue: string;
		};

		type TASClientReadTreatmentClassification = {
			owner: 'sbatten';
			comment: 'Logged when a treatment value is read from the experiment service';
			treatmentValue: { classification: 'SystemMetaData'; purpose: 'PerformanceAndHealth'; comment: 'The value of the read treatment' };
			treatmentName: { classification: 'SystemMetaData'; purpose: 'PerformanceAndHealth'; comment: 'The name of the treatment that was read' };
		};

		this.telemetryService.publicLog2<TASClientReadTreatmentData, TASClientReadTreatmentClassification>('tasClientReadTreatmentComplete',
			{ treatmentName: name, treatmentValue: JSON.stringify(result) });

		return result;
	}

	/**
	 * Provides the set of active experiment identifiers for diagnostic or analytical purposes.
	 * Ensures the TAS client is fully initialized before attempting to read the context.
	 */
	async getCurrentExperiments(): Promise<string[] | undefined> {
		if (!this.tasClient) {
			return undefined;
		}

		if (!this.experimentsEnabled) {
			return undefined;
		}

		// Barrier: wait for the internal TAS client promise to resolve.
		await this.tasClient;

		return (this.telemetry as WorkbenchAssignmentServiceTelemetry)?.assignmentContext;
	}
}

/**
 * Register the service as a singleton to be instantiated lazily (Delayed) when required by the workbench.
 */
registerSingleton(IWorkbenchAssignmentService, WorkbenchAssignmentService, InstantiationType.Delayed);

/**
 * Configuration Registration
 * Defines the user-facing setting to enable/disable experiment fetching.
 * Restricted scope (APPLICATION) ensures this cannot be overridden by untrusted workspaces.
 */
const registry = Registry.as<IConfigurationRegistry>(ConfigurationExtensions.Configuration);
registry.registerConfiguration({
	...workbenchConfigurationNodeBase,
	'properties': {
		'workbench.enableExperiments': {
			'type': 'boolean',
			'description': localize('workbench.enableExperiments', "Fetches experiments to run from a Microsoft online service."),
			'default': true,
			'scope': ConfigurationScope.APPLICATION,
			'restricted': true,
			'tags': ['usesOnlineServices']
		}
	}
});