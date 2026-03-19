/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @module vs/workbench/services/coreExperimentation/common/coreExperimentationService
 * @description
 * This module provides the `ICoreExperimentationService` and its implementation,
 * `CoreExperimentationService`. This service is responsible for managing client-side
 * experiments, particularly for evaluating different startup experiences within VS Code.
 * It integrates with various platform services to:
 * - Determine if a user is eligible for a startup experiment (e.g., based on session age).
 * - Assign users to experiment groups using a probabilistic cohort allocation strategy.
 * - Persist experiment assignments to ensure consistent user experience across sessions.
 * - Report experiment participation telemetry for analysis.
 * - Bind the active experiment group to a context key for UI/feature activation.
 * The service aims to enable A/B testing of core VS Code features in a controlled manner.
 */

import { Disposable } from '../../../../base/common/lifecycle.js';
import { InstantiationType, registerSingleton } from '../../../../platform/instantiation/common/extensions.js';
import { IProductService } from '../../../../platform/product/common/productService.js';
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';
import { firstSessionDateStorageKey, ITelemetryService } from '../../../../platform/telemetry/common/telemetry.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { IContextKeyService, RawContextKey } from '../../../../platform/contextkey/common/contextkey.js';

/**
 * @constant ICoreExperimentationService
 * @description
 * Decorator for the Core Experimentation Service, used for dependency injection.
 * This interface defines the contract for services that manage core experiments.
 */
export const ICoreExperimentationService = createDecorator<ICoreExperimentationService>('coreExperimentationService');

/**
 * @constant startupExpContext
 * @description
 * A context key that holds the name of the active startup experiment group a user is in.
 * This allows features to be conditionally enabled or disabled based on the experiment assignment.
 */
export const startupExpContext = new RawContextKey<string>('coreExperimentation.startupExpGroup', '');

/**
 * @interface IExperiment
 * @description
 * Defines the structure for an experiment assignment for a user.
 */
interface IExperiment {
	/** A random number [0, 1) assigned to the user for this experiment. */
	cohort: number;
	/** An optional sub-cohort number, reserved for future use. */
	subCohort: number;
	/** The specific experiment group the user has been assigned to. */
	experimentGroup: StartupExperimentGroup;
	/** The iteration number of the experiment configuration. */
	iteration: number;
	/** Indicates whether the user is currently participating in this experiment. */
	isInExperiment: boolean;
}

/**
 * @interface ICoreExperimentationService
 * @description
 * Interface for the Core Experimentation Service.
 */
export interface ICoreExperimentationService {
	readonly _serviceBrand: undefined;
	/**
	 * Retrieves the currently active startup experiment assignment for the user.
	 * @returns An `IExperiment` object if the user is part of an experiment, otherwise `undefined`.
	 */
	getExperiment(): IExperiment | undefined;
}

/**
 * @interface ExperimentGroupDefinition
 * @description
 * Defines the properties of an individual experiment group within an experiment configuration.
 */
interface ExperimentGroupDefinition {
	/** The name of the experiment group (e.g., 'control', 'maximizedChat'). */
	name: StartupExperimentGroup;
	/** The minimum normalized cohort value for this group (inclusive). */
	min: number;
	/** The maximum normalized cohort value for this group (exclusive). */
	max: number;
	/** The iteration number of this group's allocation. */
	iteration: number;
}

/**
 * @interface ExperimentConfiguration
 * @description
 * Defines the overall configuration for a specific experiment.
 */
interface ExperimentConfiguration {
	/** The unique name of the experiment. */
	experimentName: string;
	/** The percentage of users targeted for this experiment [0, 100]. */
	targetPercentage: number;
	/** An array of group definitions that make up this experiment. */
	groups: ExperimentGroupDefinition[];
}

/**
 * @enum StartupExperimentGroup
 * @description
 * Enumerates the different possible startup experiment groups.
 */
export enum StartupExperimentGroup {
	/** Users who do not receive any experimental startup experience. */
	Control = 'control',
	/** Users who see a maximized chat experience on startup. */
	MaximizedChat = 'maximizedChat',
	/** Users who see a split view with an empty editor and chat on startup. */
	SplitEmptyEditorChat = 'splitEmptyEditorChat',
	/** Users who see a split view with a welcome screen and chat on startup. */
	SplitWelcomeChat = 'splitWelcomeChat'
}

/**
 * @constant STARTUP_EXPERIMENT_NAME
 * @description
 * The universal name used to identify startup-related experiments.
 */
export const STARTUP_EXPERIMENT_NAME = 'startup';

/**
 * @constant EXPERIMENT_CONFIGURATIONS
 * @description
 * A record holding the experiment configurations for different product qualities (e.g., 'stable', 'insider').
 * Each configuration specifies the experiment name, target percentage of users, and the allocation
 * of users into different experiment groups. The `iteration` number should be bumped when group
 * allocations are changed to track configuration updates.
 */
const EXPERIMENT_CONFIGURATIONS: Record<string, ExperimentConfiguration> = {
	stable: {
		experimentName: STARTUP_EXPERIMENT_NAME,
		targetPercentage: 20, // 20% of users are targeted for this experiment.
		groups: [
			// Bump the iteration each time we change group allocations
			{ name: StartupExperimentGroup.Control, min: 0.0, max: 0.25, iteration: 1 }, // 25% of targeted users (5% of total)
			{ name: StartupExperimentGroup.MaximizedChat, min: 0.25, max: 0.5, iteration: 1 }, // 25% of targeted users (5% of total)
			{ name: StartupExperimentGroup.SplitEmptyEditorChat, min: 0.5, max: 0.75, iteration: 1 }, // 25% of targeted users (5% of total)
			{ name: StartupExperimentGroup.SplitWelcomeChat, min: 0.75, max: 1.0, iteration: 1 } // 25% of targeted users (5% of total)
		]
	},
	insider: {
		experimentName: STARTUP_EXPERIMENT_NAME,
		targetPercentage: 20, // 20% of users are targeted for this experiment.
		groups: [
			// Bump the iteration each time we change group allocations
			{ name: StartupExperimentGroup.Control, min: 0.0, max: 0.25, iteration: 1 }, // 25% of targeted users (5% of total)
			{ name: StartupExperimentGroup.MaximizedChat, min: 0.25, max: 0.5, iteration: 1 }, // 25% of targeted users (5% of total)
			{ name: StartupExperimentGroup.SplitEmptyEditorChat, min: 0.5, max: 0.75, iteration: 1 }, // 25% of targeted users (5% of total)
			{ name: StartupExperimentGroup.SplitWelcomeChat, min: 0.75, max: 1.0, iteration: 1 } // 25% of targeted users (5% of total)
		]
	}
};

/**
 * @class CoreExperimentationService
 * @extends Disposable
 * @implements ICoreExperimentationService
 * @description
 * Implementation of the `ICoreExperimentationService`. This service handles the
 * initialization, assignment, and management of core experiments, particularly
 * those related to the VS Code startup experience. It ensures that users are
 * consistently assigned to experiment groups and that participation is logged.
 */
export class CoreExperimentationService extends Disposable implements ICoreExperimentationService {
	declare readonly _serviceBrand: undefined;

	// A map to store active experiments, keyed by experiment name.
	private readonly experiments = new Map<string, IExperiment>();

	constructor(
		@IStorageService private readonly storageService: IStorageService,
		@ITelemetryService private readonly telemetryService: ITelemetryService,
		@IProductService private readonly productService: IProductService,
		@IContextKeyService private readonly contextKeyService: IContextKeyService
	) {
		super();
		// Immediately initialize experiments upon service instantiation.
		this.initializeExperiments();
	}

	/**
	 * @method initializeExperiments
	 * @description
	 * Initializes and assigns the user to experiments based on predefined configurations,
	 * session age, and existing storage. This method prevents re-assignment to an
	 * experiment if the user has already been part of it and skips if not a candidate.
	 */
	private initializeExperiments(): void {
		// Retrieve the date of the user's first session. If not found, assume current date.
		const firstSessionDateString = this.storageService.get(firstSessionDateStorageKey, StorageScope.APPLICATION) || new Date().toUTCString();
		// Calculate days since the first session.
		const daysSinceFirstSession = ((+new Date()) - (+new Date(firstSessionDateString))) / 1000 / 60 / 60 / 24;
		// If more than 1 day has passed since the first session, the user is not a candidate
		// for startup experiments, which are typically targeted at new users.
		if (daysSinceFirstSession > 1) {
			// not a startup exp candidate.
			return;
		}

		// Get the experiment configuration relevant to the current product quality (stable/insider).
		const experimentConfig = this.getExperimentConfiguration();
		if (!experimentConfig) {
			return;
		}

		// Check local storage to see if this user has already been assigned to this specific experiment.
		const storageKey = `coreExperimentation.${experimentConfig.experimentName}`;
		const storedExperiment = this.storageService.get(storageKey, StorageScope.APPLICATION);
		if (storedExperiment) {
			// If a stored experiment is found, the user is already assigned, so skip re-assignment.
			return;
		}

		// Create a new startup experiment assignment for the user.
		const experiment = this.createStartupExperiment(experimentConfig.experimentName, experimentConfig);
		if (experiment) {
			// If the user is assigned to an experiment group:
			this.experiments.set(experimentConfig.experimentName, experiment); // Store the experiment locally.
			this.sendExperimentTelemetry(experimentConfig.experimentName, experiment); // Send telemetry about the assignment.
			// Bind the experiment group name to a context key, making it accessible for conditional UI/feature activation.
			startupExpContext.bindTo(this.contextKeyService).set(experiment.experimentGroup);
			// Persist the experiment assignment to storage to ensure consistency across sessions.
			this.storageService.store(
				storageKey,
				JSON.stringify(experiment), // Store the experiment object as a JSON string.
				StorageScope.APPLICATION,
				StorageTarget.MACHINE
			);
		}
	}

	/**
	 * @method getExperimentConfiguration
	 * @description
	 * Retrieves the appropriate experiment configuration based on the current product quality
	 * (e.g., 'stable', 'insider').
	 * @returns The `ExperimentConfiguration` for the current product quality, or `undefined` if not found.
	 */
	private getExperimentConfiguration(): ExperimentConfiguration | undefined {
		const quality = this.productService.quality;
		if (!quality) {
			return undefined;
		}
		// Return the configuration matching the product quality.
		return EXPERIMENT_CONFIGURATIONS[quality];
	}

	/**
	 * @method createStartupExperiment
	 * @description
	 * Assigns a user to a startup experiment group based on a random cohort number
	 * and the experiment's target percentage and group definitions.
	 * @param experimentName The name of the experiment.
	 * @param experimentConfig The configuration for the experiment.
	 * @returns An `IExperiment` object detailing the user's assignment, or `undefined`
	 *          if the user does not fall within the experiment's target percentage.
	 */
	private createStartupExperiment(experimentName: string, experimentConfig: ExperimentConfiguration): IExperiment | undefined {
		const cohort = Math.random(); // Generate a random number between 0 (inclusive) and 1 (exclusive).

		// Check if the user's cohort falls outside the targeted percentage for the experiment.
		if (cohort >= experimentConfig.targetPercentage / 100) {
			return undefined; // User is not part of the experiment.
		}

		// Normalize the cohort value to the experiment's active range [0, targetPercentage/100]
		// to distribute users evenly across experiment groups within the targeted population.
		const normalizedCohort = cohort / (experimentConfig.targetPercentage / 100);

		// Iterate through defined experiment groups to find which group the normalized cohort falls into.
		for (const group of experimentConfig.groups) {
			if (normalizedCohort >= group.min && normalizedCohort < group.max) {
				return {
					cohort,
					subCohort: normalizedCohort, // The normalized cohort value as a sub-cohort identifier.
					experimentGroup: group.name,
					iteration: group.iteration,
					isInExperiment: true
				};
			}
		}
		// Should theoretically not be reached if group ranges cover [0, 1).
		return undefined;
	}

	/**
	 * @method sendExperimentTelemetry
	 * @description
	 * Sends telemetry data recording the user's experiment cohort assignment.
	 * This data is crucial for analyzing experiment results.
	 * @param experimentName The name of the experiment.
	 * @param experiment The `IExperiment` object containing the assignment details.
	 */
	private sendExperimentTelemetry(experimentName: string, experiment: IExperiment): void {
		// Type definitions for telemetry event and classification, providing metadata for data analysis.
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

		// Log the experiment cohort event to telemetry.
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
	 * @method getExperiment
	 * @description
	 * Retrieves the details of the active startup experiment for the current user.
	 * @returns An `IExperiment` object if the user is part of the startup experiment, otherwise `undefined`.
	 */
	getExperiment(): IExperiment | undefined {
		return this.experiments.get(STARTUP_EXPERIMENT_NAME);
	}
}

/**
 * @description
 * Registers `CoreExperimentationService` as a singleton service within the VS Code
 * dependency injection system. It is instantiated lazily (`InstantiationType.Delayed`),
 * meaning it will only be created when it is first requested.
 */
registerSingleton(ICoreExperimentationService, CoreExperimentationService, InstantiationType.Delayed);