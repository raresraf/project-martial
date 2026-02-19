/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file This file provides the service for running core A/B experiments,
 * particularly for new user startup experiences in VS Code.
 */

import { Disposable } from '../../../../base/common/lifecycle.js';
import { InstantiationType, registerSingleton } from '../../../../platform/instantiation/common/extensions.js';
import { IProductService } from '../../../../platform/product/common/productService.js';
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';
import { firstSessionDateStorageKey, ITelemetryService } from '../../../../platform/telemetry/common/telemetry.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { IContextKeyService, RawContextKey } from '../../../../platform/contextkey/common/contextkey.js';

export const ICoreExperimentationService = createDecorator<ICoreExperimentationService>('coreExperimentationService');

/**
 * A context key that is set to the name of the startup experiment group the user is in.
 * This can be used to drive UI changes via `when` clauses.
 */
export const startupExpContext = new RawContextKey<string>('coreExperimentation.startupExpGroup', '');

/**
 * Represents the details of an experiment a user is participating in.
 */
interface IExperiment {
	/** A random number between 0 and 1 assigned to the user. */
	cohort: number;
	/** A cohort number normalized to the experiment's target population. */
	subCohort: number;
	/** The specific group within the experiment that the user belongs to. */
	experimentGroup: StartupExperimentGroup;
	/** The version number of the experiment configuration. */
	iteration: number;
	/** A flag indicating if the user is part of the experiment. */
	isInExperiment: boolean;
}

/**
 * The public interface for the CoreExperimentationService.
 */
export interface ICoreExperimentationService {
	readonly _serviceBrand: undefined;

	/**
	 * Retrieves the experiment details for the current user, if they are part of an experiment.
	 * @returns The experiment object or `undefined`.
	 */
	getExperiment(): IExperiment | undefined;
}

/**
 * Defines the allocation for a single group within an experiment.
 */
interface ExperimentGroupDefinition {
	name: StartupExperimentGroup;
	/** The lower bound of the cohort range for this group (inclusive). */
	min: number;
	/** The upper bound of the cohort range for this group (exclusive). */
	max: number;
	/** The iteration number for this group definition. */
	iteration: number;
}

/**
 * Defines the complete configuration for a single A/B experiment.
 */
interface ExperimentConfiguration {
	experimentName: string;
	/** The percentage of the total user base to include in the experiment (0-100). */
	targetPercentage: number;
	/** The different groups and their cohort allocations. */
	groups: ExperimentGroupDefinition[];
}

/**
 * Enum defining the possible experiment groups for the startup experience.
 */
export enum StartupExperimentGroup {
	Control = 'control',
	MaximizedChat = 'maximizedChat',
	SplitEmptyEditorChat = 'splitEmptyEditorChat',
	SplitWelcomeChat = 'splitWelcomeChat'
}

export const STARTUP_EXPERIMENT_NAME = 'startup';

/**
 * Central configuration for A/B tests. This object defines the experiment parameters
 * for different VS Code release qualities (e.g., 'stable' vs. 'insider').
 */
const EXPERIMENT_CONFIGURATIONS: Record<string, ExperimentConfiguration> = {
	stable: {
		experimentName: STARTUP_EXPERIMENT_NAME,
		targetPercentage: 100,
		groups: [
			// Bump the iteration each time we change group allocations
			{ name: StartupExperimentGroup.Control, min: 0.0, max: 0.0, iteration: 1 },
			{ name: StartupExperimentGroup.MaximizedChat, min: 0.0, max: 1.0, iteration: 1 },
			{ name: StartupExperimentGroup.SplitEmptyEditorChat, min: 0.0, max: 0.0, iteration: 1 },
			{ name: StartupExperimentGroup.SplitWelcomeChat, min: 0.0, max: 0.0, iteration: 1 }
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
 * Service responsible for managing and assigning users to core A/B experiments.
 * It handles user bucketing, ensures stickiness across sessions, and provides
 * experiment data to other services.
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
	 * Orchestrates the experiment assignment process on startup.
	 * Functional Goal: To check if a user is eligible for a startup experiment,
	 * assign them to a group if they are, and persist that assignment for stickiness.
	 */
	private initializeExperiments(): void {

		// Block Pre-condition: Only run startup experiments for new users (within 1 day of first session).
		const firstSessionDateString = this.storageService.get(firstSessionDateStorageKey, StorageScope.APPLICATION) || new Date().toUTCString();
		const daysSinceFirstSession = ((+new Date()) - (+new Date(firstSessionDateString))) / 1000 / 60 / 60 / 24;
		if (daysSinceFirstSession > 1) {
			// not a startup exp candidate.
			return;
		}

		const experimentConfig = this.getExperimentConfiguration();
		if (!experimentConfig) {
			return;
		}

		// Block Pre-condition: Ensure stickiness by checking if the user is already in an experiment.
		const storageKey = `coreExperimentation.${experimentConfig.experimentName}`;
		const storedExperiment = this.storageService.get(storageKey, StorageScope.APPLICATION);
		if (storedExperiment) {
			return;
		}

		// If the user is eligible and not assigned, create and store the experiment assignment.
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
	 */
	private getExperimentConfiguration(): ExperimentConfiguration | undefined {
		const quality = this.productService.quality;
		// if (!quality) {
		// 	return undefined;
		// }
		return EXPERIMENT_CONFIGURATIONS[quality || 'stable'];
	}

	/**
	 * Implements the user bucketing logic.
	 * Assigns a user to an experiment group based on a random number and the group allocations.
	 * @param experimentName The name of the experiment.
	 * @param experimentConfig The configuration object for the experiment.
	 * @returns An experiment object if the user is included, otherwise undefined.
	 */
	private createStartupExperiment(experimentName: string, experimentConfig: ExperimentConfiguration): IExperiment | undefined {
		const cohort = Math.random();

		// Check if the user falls within the overall experiment population.
		if (cohort >= experimentConfig.targetPercentage / 100) {
			return undefined;
		}

		// Normalize the cohort to the experiment range [0, 1] for group assignment.
		const normalizedCohort = cohort / (experimentConfig.targetPercentage / 100);

		// Find which specific group this user falls into based on the normalized cohort.
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
	 * Sends a telemetry event to record the user's assignment to an experiment group.
	 * This is critical for later analysis of the experiment's impact.
	 * @param experimentName The name of the experiment.
	 * @param experiment The experiment details for the user.
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
	 * Public getter to retrieve the experiment details for the current session.
	 */
	getExperiment(): IExperiment | undefined {
		return this.experiments.get(STARTUP_EXPERIMENT_NAME);
	}
}

registerSingleton(ICoreExperimentationService, CoreExperimentationService, InstantiationType.Delayed);
