/**
 * @file assignmentService.ts
 * @brief Workbench service for A/B testing and feature experimentation management.
 * This module implements the `WorkbenchAssignmentService`, which integrates with 
 * Microsoft's Treatment Assignment Service (TAS) to manage feature experiments 
 * within VS Code. It provides a robust framework for asynchronous treatment 
 * retrieval, persistent local storage of experiment state using the `Memento` 
 * pattern, and detailed telemetry auditing for experimentation tracking.
 *
 * Domain: IDE Infrastructure, Feature Flagging, A/B Testing.
 */

/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

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
 * @brief Public API for experiment discovery in the workbench.
 */
export interface IWorkbenchAssignmentService extends IAssignmentService {
	/**
	 * @return List of active experiment identifiers for the current session.
	 */
	getCurrentExperiments(): Promise<string[] | undefined>;
}

/**
 * @class MementoKeyValueStorage
 * @brief Adapter for TAS client storage utilizing the VS Code Memento pattern.
 * Functional Utility: Bridges the external library's storage requirements with 
 * VS Code's persistent machine-scoped storage.
 */
class MementoKeyValueStorage implements IKeyValueStorage {
	private mementoObj: MementoObject;
	constructor(private memento: Memento) {
		// Invariant: Uses Application scope and Machine target for experiment persistence.
		this.mementoObj = memento.getMemento(StorageScope.APPLICATION, StorageTarget.MACHINE);
	}

	async getValue<T>(key: string, defaultValue?: T | undefined): Promise<T | undefined> {
		const value = await this.mementoObj[key];
		return value || defaultValue;
	}

	setValue<T>(key: string, value: T): void {
		this.mementoObj[key] = value;
		this.memento.saveMemento();
	}
}

/**
 * @class WorkbenchAssignmentServiceTelemetry
 * @brief Integrated telemetry dispatcher for experimentation events.
 * Functional Utility: Captures assignment context and logs treatment-read events 
 * for GDPR-compliant feature insight analysis.
 */
class WorkbenchAssignmentServiceTelemetry implements IExperimentationTelemetry {
	private _lastAssignmentContext: string | undefined;
	constructor(
		private telemetryService: ITelemetryService,
		private productService: IProductService
	) { }

	/**
	 * @return Parsed list of semicolon-delimited experiment tags.
	 */
	get assignmentContext(): string[] | undefined {
		return this._lastAssignmentContext?.split(';');
	}

	// __GDPR__COMMON__ "abexp.assignmentcontext" : { "classification": "SystemMetaData", "purpose": "FeatureInsight" }
	/**
	 * Updates shared experiment properties in the global telemetry state.
	 */
	setSharedProperty(name: string, value: string): void {
		if (name === this.productService.tasConfig?.assignmentContextTelemetryPropertyName) {
			this._lastAssignmentContext = value;
		}

		this.telemetryService.setExperimentProperty(name, value);
	}

	/**
	 * Dispatches experimentation events to the central telemetry service.
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
 * @brief Primary implementation of experimentation orchestration.
 * Logic: Coordinates the TAS client initialization using local Memento storage 
 * and custom telemetry dispatchers.
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
	 * @brief Guard logic for experimentation availability.
	 * Invariant: Experiments are disabled in tests, smoke runs, or if explicitly 
	 * opted-out via user configuration.
	 */
	protected override get experimentsEnabled(): boolean {
		return !this.environmentService.disableExperiments &&
			!this.environmentService.extensionTestsLocationURI &&
			!(this.environmentService as IWorkbenchEnvironmentService).enableSmokeTestDriver &&
			this.configurationService.getValue('workbench.enableExperiments') === true;
	}

	/**
	 * @brief Retrieves the treatment value for a specific feature.
	 * Logic: Executes an asynchronous lookup via the TAS client and logs the 
	 * completion event with the read value.
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

		// Audit Log: Records the resolution of an experimental treatment.
		this.telemetryService.publicLog2<TASClientReadTreatmentData, TASClientReadTreatmentClassification>('tasClientReadTreatmentComplete',
			{ treatmentName: name, treatmentValue: JSON.stringify(result) });

		return result;
	}

	/**
	 * @brief Returns metadata about the current set of experiments.
	 */
	async getCurrentExperiments(): Promise<string[] | undefined> {
		if (!this.tasClient) {
			return undefined;
		}

		if (!this.experimentsEnabled) {
			return undefined;
		}

		// Ensure the client is fully initialized before querying context.
		await this.tasClient;

		return (this.telemetry as WorkbenchAssignmentServiceTelemetry)?.assignmentContext;
	}
}

// Global Registration: delay initialization until required by the workbench.
registerSingleton(IWorkbenchAssignmentService, WorkbenchAssignmentService, InstantiationType.Delayed);

// System Configuration: Registers the 'workbench.enableExperiments' setting.
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
