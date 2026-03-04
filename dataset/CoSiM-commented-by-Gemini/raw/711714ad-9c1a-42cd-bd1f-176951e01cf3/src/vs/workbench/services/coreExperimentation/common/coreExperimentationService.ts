/**
 * @file coreExperimentationService.ts
 * @brief Manages core A/B experimentation for the application.
 *
 * This service is responsible for assigning users to experiment groups for
 * features being tested, particularly for "startup experiments" that are
 * evaluated for new users. It handles cohort assignment, experiment configuration,
 * and telemetry for tracking experiment outcomes.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Disposable } from '../../../../base/common/lifecycle.js';
import { InstantiationType, registerSingleton } from '../../../../platform/instantiation/common/extensions.js';
import { IProductService } from '../../../../platform/product/common/productService.js';
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';
import { firstSessionDateStorageKey, ITelemetryService } from '../../../../platform/telemetry/common/telemetry.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { IContextKeyService, RawContextKey } from '../../../../platform/contextkey/common/contextkey.js';

export const ICoreExperimentationService = createDecorator<ICoreExperimentationService>('coreExperimentationService');

/**
 * A context key that will be set to the experiment group the user is in.
 * This can be used to show/hide UI elements or change behavior based on the experiment group.
 */
export const startupExpContext = new RawContextKey<string>('coreExperimentation.startupExpGroup', '');

/**
 * Interface representing a single experiment's state for a user.
 */
interface IExperiment {
	cohort: number;
	subCohort: number; // Optional for future use
	experimentGroup: StartupExperimentGroup;
	iteration: number;
	isInExperiment: boolean;
}

/**
 * Service interface for core experimentation.
 */
export interface ICoreExperimentationService {
	readonly _serviceBrand: undefined;
	getExperiment(): IExperiment | undefined;
}

/**
 * Defines the structure for an experiment group, including its name and the cohort range.
 */
interface ExperimentGroupDefinition {
	name: StartupExperimentGroup;
	min: number;
	max: number;
	iteration: number;
}

/**
 * Defines the overall configuration for an experiment, including its name,
 * the percentage of users to target, and the groups within the experiment.
 */
interface ExperimentConfiguration {
	experimentName: string;
	targetPercentage: number;
	groups: ExperimentGroupDefinition[];
}

/**
 * Enum defining the possible experiment groups for the startup experiment.
 */
export enum StartupExperimentGroup {
	Control = 'control',
	MaximizedChat = 'maximizedChat',
	SplitEmptyEditorChat = 'splitEmptyEditorChat',
	SplitWelcomeChat = 'splitWelcomeChat'
}

export const STARTUP_EXPERIMENT_NAME = 'startup';

/**
 * Configuration for the startup experiments, keyed by product quality (e.g., 'stable', 'insider').
 */
const EXPERIMENT_CONFIGURATIONS: Record<string, ExperimentConfiguration> = {
	stable: {
		experimentName: STARTUP_EXPERIMENT_NAME,
		targetPercentage: 20,
		groups: [
			// Bump the iteration each time we change group allocations
			{ name: StartupExperimentGroup.Control, min: 0.0, max: 0.25, iteration: 1 },
			{ name: StartupExperimentGroup.MaximizedChat, min: 0.25, max: 0.5, iteration: 1 },
			{ name: StartupExperimentGroup.SplitEmptyEditorChat, min: 0.5, max: 0.75, iteration: 1 },
			{ name: StartupExperimentGroup.SplitWelcomeChat, min: 0.75, max: 1.0, iteration: 1 }
		]
	},
	insider: {
		experimentName: STARTUP_EXPERIMENT_NAME,
		targetPercentage: 20,
		groups: [
			// Bump the iteration each time we change group allocations
			{ name: StartupExperimentGroup.Control, min: 0.0, max: 0.25, iteration: 1 },
			{ name: StartupExperimentGroup.MaximizedChat, min: 0.25, max: 0.5, iteration: 1 },
			{ name: StartupExperimentGroup.SplitEmptyEditorChat, min: 0.5, max: 0.75, iteration: 1 },
			{ name: StartupExperimentGroup.SplitWelcomeChat, min: 0.75, max: 1.0, iteration: 1 }
		]
	}
};

/**
 * Implementation of the core experimentation service.
 */
export class CoreExperimentationService extends Disposable implements ICoreExperimentationService {
	declare readonly _serviceBrand: undefined;

	private readonly experiments = new Map<string, IExperiment>();

	constructor(
		@IStorageService private readonly storageService: IStorageService,
		@ITelemetryService private readonly telemetryService: ITelemetryService,
		@IProductService private readonly productService: IProductService,
		@IContextKeyService private readonly contextKeyService: IContextKeyService
	) {
		super();
		this.initializeExperiments();
	}

	/**
	 * Initializes the experiments on startup. This method determines if the user
	 * is a candidate for any experiments and, if so, assigns them to a group.
	 */
	private initializeExperiments(): void {

		const firstSessionDateString = this.storageService.get(firstSessionDateStorageKey, StorageScope.APPLICATION) || new Date().toUTCString();
		const daysSinceFirstSession = ((+new Date()) - (+new Date(firstSessionDateString))) / 1000 / 60 / 60 / 24;
		// The startup experiment is only for users within their first day of use.
		if (daysSinceFirstSession > 1) {
			// not a startup exp candidate.
			return;
		}

		const experimentConfig = this.getExperimentConfiguration();
		if (!experimentConfig) {
			return;
		}

		// also check storage to see if this user has already seen the startup experience
		const storageKey = `coreExperimentation.${experimentConfig.experimentName}`;
		const storedExperiment = this.storageService.get(storageKey, StorageScope.APPLICATION);
		if (storedExperiment) {
			return;
		}

		const experiment = this.createStartupExperiment(experimentConfig.experimentName, experimentConfig);
		if (experiment) {
			this.experiments.set(experimentConfig.experimentName, experiment);
			this.sendExperimentTelemetry(experimentConfig.experimentName, experiment);
			startupExpContext.bindTo(this.contextKeyService).set(experiment.experimentGroup);
			this.storageService.store(
				storageKey,
				JSON.stringify(experiment),
				StorageScope.APPLICATION,
				StorageTarget.MACHINE
			);
		}
	}

	/**
	 * Retrieves the experiment configuration based on the product quality (e.g., stable, insider).
	 * @returns The experiment configuration, or undefined if not found.
	 */
	private getExperimentConfiguration(): ExperimentConfiguration | undefined {
		const quality = this.productService.quality;
		if (!quality) {
			return undefined;
		}
		return EXPERIMENT_CONFIGURATIONS[quality];
	}

	/**
	 * Creates a startup experiment for the user.
	 * This method determines if the user should be in the experiment based on a random
	 * cohort number and the target percentage. If they are, it assigns them to a group.
	 * @param experimentName - The name of the experiment.
	 * @param experimentConfig - The configuration for the experiment.
	 * @returns An experiment object if the user is included in the experiment, otherwise undefined.
	 */
	private createStartupExperiment(experimentName: string, experimentConfig: ExperimentConfiguration): IExperiment | undefined {
		const cohort = Math.random();

		if (cohort >= experimentConfig.targetPercentage / 100) {
			return undefined;
		}

		// Normalize the cohort to the experiment range [0, targetPercentage/100]
		const normalizedCohort = cohort / (experimentConfig.targetPercentage / 100);

		// Find which group this user falls into
		for (const group of experimentConfig.groups) {
			if (normalizedCohort >= group.min && normalizedCohort < group.max) {
				return {
					cohort,
					subCohort: normalizedCohort,
					experimentGroup: group.name,
					iteration: group.iteration,
					isInExperiment: true
				};
			}
		}
		return undefined;
	}

	/**
	 * Sends telemetry data about the user's experiment cohort and group.
	 * @param experimentName - The name of the experiment.
	 * @param experiment - The experiment object containing user's cohort info.
	 */
	private sendExperimentTelemetry(experimentName: string, experiment: IExperiment): void {
		type ExperimentCohortClassification = {
			owner: 'bhavyaus';
			comment: 'Records which experiment cohort the user is in for core experiments';
			experimentName: { classification: 'SystemMetaData'; purpose: 'FeatureInsight'; comment: 'The name of the experiment' };
			cohort: { classification: 'SystemMetaData'; purpose: 'FeatureInsight'; comment: 'The exact cohort number for the user' };
			subCohort: { classification: 'SystemMetaData'; purpose: 'FeatureInsight'; comment: 'The exact sub-cohort number for the user in the experiment cohort' };
			experimentGroup: { classification: 'SystemMetaData'; purpose: 'FeatureInsight'; comment: 'The experiment group the user is in' };
			iteration: { classification: 'SystemMetaData'; purpose: 'FeatureInsight'; comment: 'The iteration number for the experiment' };
			isInExperiment: { classification: 'SystemMetaData'; purpose: 'FeatureInsight'; comment: 'Whether the user is in the experiment' };
		};

		type ExperimentCohortEvent = {
			experimentName: string;
			cohort: number;
			subCohort: number;
			experimentGroup: string;
			iteration: number;
			isInExperiment: boolean;
		};

		this.telemetryService.publicLog2<ExperimentCohortEvent, ExperimentCohortClassification>(
			`coreExperimentation.experimentCohort`,
			{
				experimentName,
				cohort: experiment.cohort,
				subCohort: experiment.subCohort,
				experimentGroup: experiment.experimentGroup,
				iteration: experiment.iteration,
				isInExperiment: experiment.isInExperiment
			}
		);
	}

	/**
	 * Retrieves the startup experiment for the current session.
	 * @returns The startup experiment object, or undefined if none exists.
	 */
	getExperiment(): IExperiment | undefined {
		return this.experiments.get(STARTUP_EXPERIMENT_NAME);
	}
}

registerSingleton(ICoreExperimentationService, CoreExperimentationService, InstantiationType.Delayed);