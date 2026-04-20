/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @035e78e1-6304-45f6-8100-d96616086713/src/vs/workbench/services/assignment/common/assignmentService.ts
 * @brief Workbench-level implementation of the Experimentation (A/B Testing) Assignment Service.
 * 
 * Functional Intent: Integrates the Treatment Assignment Service (TAS) client with VS Code's 
 * workbench. It manages experiment state persistence using Mementos and handles 
 * telemetry reporting for feature treatments and assignment contexts, adhering to 
 * GDPR and privacy constraints.
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
import { IEnvironmentService } from '../../../../platform/environment/common/environment.js';

export const IWorkbenchAssignmentService = createDecorator<IWorkbenchAssignmentService>('WorkbenchAssignmentService');

export interface IWorkbenchAssignmentService extends IAssignmentService {
	getCurrentExperiments(): Promise<string[] | undefined>;
}

/**
 * @brief Adapter for TAS client to use VS Code's Memento-based persistent storage.
 * 
 * Logic: Maps high-level key-value operations to an APPLICATION-scoped, 
 * MACHINE-targeted Memento, ensuring data persists across sessions on the same device.
 */
class MementoKeyValueStorage implements IKeyValueStorage {
	private mementoObj: MementoObject;
	constructor(private memento: Memento) {
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
 * @brief Telemetry provider for the Experimentation Service.
 * 
 * Functional Intent: Forwards assignment events and shared properties to the 
 * global telemetry service. It tracks the latest assignment context for 
 * diagnostic and verification purposes.
 */
class WorkbenchAssignmentServiceTelemetry implements IExperimentationTelemetry {
	private _lastAssignmentContext: string | undefined;
	constructor(
		private telemetryService: ITelemetryService,
		private productService: IProductService
	) { }

	get assignmentContext(): string[] | undefined {
		return this._lastAssignmentContext?.split(';');
	}

	// __GDPR__COMMON__ "abexp.assignmentcontext" : { "classification": "SystemMetaData", "purpose": "FeatureInsight" }
	/**
	 * Block Logic: Updates shared telemetry state.
	 * Logic: Identifies specific assignment context keys based on product configuration 
	 * and broadcasts them to the experiment property store.
	 */
	setSharedProperty(name: string, value: string): void {
		if (name === this.productService.tasConfig?.assignmentContextTelemetryPropertyName) {
			this._lastAssignmentContext = value;
		}

		this.telemetryService.setExperimentProperty(name, value);
	}

	/**
	 * Block Logic: Event dispatch for experimentation queries.
	 * Logic: Flattens Map-based properties into a plain object for compatibility 
	 * with the ITelemetryService public logging interface.
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
 * @brief Primary implementation of IWorkbenchAssignmentService.
 * 
 * Functional Intent: Orchestrates the initialization of the TAS client and provides 
 * predicates to determine if experimentation is permitted based on environment 
 * variables (smoke tests, extension tests) and user configuration.
 */
export class WorkbenchAssignmentService extends BaseAssignmentService {
	constructor(
		@ITelemetryService private telemetryService: ITelemetryService,
		@IStorageService storageService: IStorageService,
		@IConfigurationService configurationService: IConfigurationService,
		@IProductService productService: IProductService,
		@IEnvironmentService environmentService: IEnvironmentService,
		@IWorkbenchEnvironmentService private readonly workbenchEnvironmentService: IWorkbenchEnvironmentService
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
	 * Block Logic: Validation logic for experiment activation.
	 * Pre-condition: Checks global environment flags and user-specific settings.
	 * Invariant: Returns false if any safety-critical test mode (smoke/extension) is active.
	 */
	protected override get experimentsEnabled(): boolean {
		return !this.environmentService.disableExperiments &&
			!this.environmentService.extensionTestsLocationURI &&
			!this.workbenchEnvironmentService.enableSmokeTestDriver &&
			this.configurationService.getValue('workbench.enableExperiments') === true;
	}

	/**
	 * Block Logic: Retrieves a specific experimental treatment.
	 * Logic: Fetches the value from the base service and logs a telemetry event 
	 * including the treatment name and resolved value for downstream analysis.
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
	 * Functional Utility: Returns the set of currently active experiments.
	 * Logic: Waits for the TAS client to initialize and then retrieves the 
	 * cached assignment context from the telemetry provider.
	 */
	async getCurrentExperiments(): Promise<string[] | undefined> {
		if (!this.tasClient) {
			return undefined;
		}

		if (!this.experimentsEnabled) {
			return undefined;
		}

		await this.tasClient;

		return (this.telemetry as WorkbenchAssignmentServiceTelemetry)?.assignmentContext;
	}
}

registerSingleton(IWorkbenchAssignmentService, WorkbenchAssignmentService, InstantiationType.Delayed);

// Block Logic: Contribution of experiment-related settings to the configuration registry.
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
