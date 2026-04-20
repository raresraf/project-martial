/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @31030aae-d803-44a4-8e16-f0374b91e746/src/vs/workbench/services/coreExperimentation/common/coreExperimentationService.ts
 * @brief Lightweight local experimentation framework for VS Code startup optimizations.
 * 
 * Functional Intent: Manages the assignment of users to experimental cohorts during 
 * the application's initial launch phase. It uses a percentage-based rollout 
 * strategy to evaluate new startup experiences (e.g., chat layouts) without 
 * relying on external service connectivity, ensuring immediate assignment on day zero.
 * 
 * Domain: A/B Testing, Feature Rollout, Client-side Experimentation.
 */

import { Disposable } from '../../../../base/common/lifecycle.js';
import { InstantiationType, registerSingleton } from '../../../../platform/instantiation/common/extensions.js';
import { IProductService } from '../../../../platform/product/common/productService.js';
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';
import { firstSessionDateStorageKey, ITelemetryService } from '../../../../platform/telemetry/common/telemetry.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { IContextKeyService, RawContextKey } from '../../../../platform/contextkey/common/contextkey.js';

export const ICoreExperimentationService = createDecorator<ICoreExperimentationService>('coreExperimentationService');
export const startupExpContext = new RawContextKey<string>('coreExperimentation.startupExpGroup', '');

interface IExperiment {
	cohort: number;
	subCohort: number; 
	experimentGroup: StartupExperimentGroup;
	iteration: number;
	isInExperiment: boolean;
}

export interface ICoreExperimentationService {
	readonly _serviceBrand: undefined;
	getExperiment(): IExperiment | undefined;
}

interface ExperimentGroupDefinition {
	name: StartupExperimentGroup;
	min: number;
	max: number;
	iteration: number;
}

interface ExperimentConfiguration {
	experimentName: string;
	targetPercentage: number;
	groups: ExperimentGroupDefinition[];
}

export enum StartupExperimentGroup {
	Control = 'control',
	MaximizedChat = 'maximizedChat',
	SplitEmptyEditorChat = 'splitEmptyEditorChat',
	SplitWelcomeChat = 'splitWelcomeChat'
}

export const STARTUP_EXPERIMENT_NAME = 'startup';

const EXPERIMENT_CONFIGURATIONS: Record<string, ExperimentConfiguration> = {
	stable: {
		experimentName: STARTUP_EXPERIMENT_NAME,
		targetPercentage: 100,
		groups: [
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
			{ name: StartupExperimentGroup.Control, min: 0.0, max: 0.25, iteration: 1 },
			{ name: StartupExperimentGroup.MaximizedChat, min: 0.25, max: 0.5, iteration: 1 },
			{ name: StartupExperimentGroup.SplitEmptyEditorChat, min: 0.5, max: 0.75, iteration: 1 },
			{ name: StartupExperimentGroup.SplitWelcomeChat, min: 0.75, max: 1.0, iteration: 1 }
		]
	}
};

/**
 * @brief Primary implementation of the client-side experimentation logic.
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
	 * initializeExperiments - Orchestrates the assignment lifecycle on service creation.
	 * 
	 * Block Logic: Candidate eligibility and persistence checks.
	 * Logic: 
	 * 1. Restricts assignments to "fresh" installs (first session < 24h old).
	 * 2. Checks persistent storage to ensure sticky assignments across restarts.
	 * 3. On successful assignment: Records telemetry and binds group ID to 
	 *    UI context keys for conditional layout rendering.
	 */
	private initializeExperiments(): void {

		const firstSessionDateString = this.storageService.get(firstSessionDateStorageKey, StorageScope.APPLICATION) || new Date().toUTCString();
		const daysSinceFirstSession = ((+new Date()) - (+new Date(firstSessionDateString))) / 1000 / 60 / 60 / 24;

		if (daysSinceFirstSession > 1) {
			return;
		}

		const experimentConfig = this.getExperimentConfiguration();
		if (!experimentConfig) {
			return;
		}

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

	private getExperimentConfiguration(): ExperimentConfiguration | undefined {
		const quality = this.productService.quality;
		return EXPERIMENT_CONFIGURATIONS[quality || 'stable'];
	}

	/**
	 * createStartupExperiment - Probabilistic assignment engine.
	 * 
	 * Algorithm: Uniform random cohort mapping.
	 * Logic: 
	 * 1. Generates a random float [0, 1].
	 * 2. Evaluates against 'targetPercentage' to decide if the user participates.
	 * 3. Normalizes the participant's value to the [0, 1] range.
	 * 4. Maps the normalized value to a specific group's [min, max] interval.
	 * 
	 * Invariant: Exactly one group is assigned for participants.
	 */
	private createStartupExperiment(experimentName: string, experimentConfig: ExperimentConfiguration): IExperiment | undefined {
		const cohort = Math.random();

		if (cohort >= experimentConfig.targetPercentage / 100) {
			return undefined;
		}

		const normalizedCohort = cohort / (experimentConfig.targetPercentage / 100);

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
	 * @brief Dispatches assignment metadata to the global telemetry collector.
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

	getExperiment(): IExperiment | undefined {
		return this.experiments.get(STARTUP_EXPERIMENT_NAME);
	}
}

registerSingleton(ICoreExperimentationService, CoreExperimentationService, InstantiationType.Delayed);
