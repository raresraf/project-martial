/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file This file provides the implementation of the workbench-specific assignment service.
 * This service is responsible for fetching, caching, and reporting experimental assignments
 * from a Microsoft online service, which is used for A/B testing and feature flagging.
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

/**
 * @interface IWorkbenchAssignmentService
 * @extends IAssignmentService
 * @description Provides methods to interact with the assignment service, including fetching current experimental assignments.
 */
export interface IWorkbenchAssignmentService extends IAssignmentService {
	/**
	 * @returns A promise that resolves to an array of current experiment names, or undefined if not available.
	 */
	getCurrentExperiments(): Promise<string[] | undefined>;
}

/**
 * @class MementoKeyValueStorage
 * @implements IKeyValueStorage
 * @description An implementation of IKeyValueStorage that uses a Memento object for persistence.
 * This class acts as a bridge between the memento pattern and a generic key-value storage interface.
 */
class MementoKeyValueStorage implements IKeyValueStorage {
	private mementoObj: MementoObject;
	constructor(private memento: Memento) {
		this.mementoObj = memento.getMemento(StorageScope.APPLICATION, StorageTarget.MACHINE);
	}

	/**
	 * Retrieves a value from the storage.
	 * @param key The key of the value to retrieve.
	 * @param defaultValue The default value to return if the key is not found.
	 * @returns A promise that resolves to the retrieved value or the default value.
	 */
	async getValue<T>(key: string, defaultValue?: T | undefined): Promise<T | undefined> {
		const value = await this.mementoObj[key];
		return value || defaultValue;
	}

	/**
	 * Stores a value in the storage.
	 * @param key The key under which to store the value.
	 * @param value The value to store.
	 */
	setValue<T>(key: string, value: T): void {
		this.mementoObj[key] = value;
		this.memento.saveMemento();
	}
}

/**
 * @class WorkbenchAssignmentServiceTelemetry
 * @implements IExperimentationTelemetry
 * @description Implements the telemetry reporting for the assignment service.
 * This class is responsible for sending experiment-related events and properties
 * to the telemetry service.
 */
class WorkbenchAssignmentServiceTelemetry implements IExperimentationTelemetry {
	private _lastAssignmentContext: string | undefined;
	constructor(
		private telemetryService: ITelemetryService,
		private productService: IProductService
	) { }

	/**
	 * Gets the last recorded assignment context.
	 */
	get assignmentContext(): string[] | undefined {
		return this._lastAssignmentContext?.split(';');
	}

	/**
	 * Sets a shared property for telemetry. If the property is the assignment context,
	 * it is stored locally.
	 * @param name The name of the property.
	 * @param value The value of the property.
	 */
	// __GDPR__COMMON__ "abexp.assignmentcontext" : { "classification": "SystemMetaData", "purpose": "FeatureInsight" }
	setSharedProperty(name: string, value: string): void {
		if (name === this.productService.tasConfig?.assignmentContextTelemetryPropertyName) {
			this._lastAssignmentContext = value;
		}

		this.telemetryService.setExperimentProperty(name, value);
	}

	/**
	 * Posts a telemetry event.
	 * @param eventName The name of the event.
	 * @param props A map of properties to include in the event.
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
 * @extends BaseAssignmentService
 * @description The core implementation of the workbench-specific assignment service. It initializes
 * the underlying TAS client and provides methods for accessing treatment variables.
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
	 * Determines if experiments are enabled based on environment, configuration, and testing flags.
	 * @returns `true` if experiments are enabled, `false` otherwise.
	 */
	protected override get experimentsEnabled(): boolean {
		return !this.environmentService.disableExperiments &&
			!this.environmentService.extensionTestsLocationURI &&
			!this.workbenchEnvironmentService.enableSmokeTestDriver &&
			this.configurationService.getValue('workbench.enableExperiments') === true;
	}

	/**
	 * Retrieves a treatment value for a given experiment name.
	 * Also logs the treatment read event to telemetry.
	 * @param name The name of the experiment.
	 * @returns A promise that resolves to the treatment value, or undefined if not available.
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
	 * Gets the list of current experiments.
	 * @returns A promise that resolves to an array of experiment names, or undefined if the service is not available.
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

// Register the assignment service as a singleton.
registerSingleton(IWorkbenchAssignmentService, WorkbenchAssignmentService, InstantiationType.Delayed);

// Register the configuration setting for enabling/disabling experiments.
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
