/**
 * @file PolicyStepsRegistryTests.java
 * @brief Unit tests for the PolicyStepsRegistry class in Elasticsearch ILM.
 * 
 * Architectural Intent: Validates the lookup and resolution of ILM steps based on index metadata 
 * and policy definitions. Ensures that step transitions and policy updates are correctly 
 * propagated to the registry's internal state and cache.
 */

package org.elasticsearch.xpack.ilm;

import org.elasticsearch.client.internal.Client;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.cluster.metadata.LifecycleExecutionState;
import org.elasticsearch.cluster.metadata.ProjectMetadata;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.index.Index;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xcontent.NamedXContentRegistry;
import org.elasticsearch.xpack.core.ilm.ErrorStep;
import org.elasticsearch.xpack.core.ilm.IndexLifecycleMetadata;
import org.elasticsearch.xpack.core.ilm.InitializePolicyContextStep;
import org.elasticsearch.xpack.core.ilm.LifecycleAction;
import org.elasticsearch.xpack.core.ilm.LifecyclePolicy;
import org.elasticsearch.xpack.core.ilm.LifecyclePolicyMetadata;
import org.elasticsearch.xpack.core.ilm.LifecyclePolicyTests;
import org.elasticsearch.xpack.core.ilm.LifecycleSettings;
import org.elasticsearch.xpack.core.ilm.MigrateAction;
import org.elasticsearch.xpack.core.ilm.MockStep;
import org.elasticsearch.xpack.core.ilm.OperationMode;
import org.elasticsearch.xpack.core.ilm.Phase;
import org.elasticsearch.xpack.core.ilm.PhaseExecutionInfo;
import org.elasticsearch.xpack.core.ilm.ShrinkAction;
import org.elasticsearch.xpack.core.ilm.ShrinkStep;
import org.elasticsearch.xpack.core.ilm.Step;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.elasticsearch.cluster.metadata.LifecycleExecutionState.ILM_CUSTOM_METADATA_KEY;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.sameInstance;
import static org.mockito.Mockito.mock;

/**
 * @brief Test suite for PolicyStepsRegistry logic and synchronization.
 */
public class PolicyStepsRegistryTests extends ESTestCase {
    private static final Step.StepKey MOCK_STEP_KEY = new Step.StepKey("mock", "mock", "mock");
    private static final NamedXContentRegistry REGISTRY = new NamedXContentRegistry(new IndexLifecycle(Settings.EMPTY).getNamedXContent());

    private IndexMetadata emptyMetadata(Index index) {
        return IndexMetadata.builder(index.getName())
            .settings(settings(IndexVersion.current()))
            .numberOfShards(randomIntBetween(1, 5))
            .numberOfReplicas(randomIntBetween(0, 5))
            .build();
    }

    /**
     * Functional Utility: Verifies retrieval of the entry-point step for a known policy.
     */
    public void testGetFirstStep() {
        String policyName = randomAlphaOfLengthBetween(2, 10);
        Step expectedFirstStep = new MockStep(MOCK_STEP_KEY, null);
        Map<String, Step> firstStepMap = Map.of(policyName, expectedFirstStep);
        PolicyStepsRegistry registry = new PolicyStepsRegistry(null, firstStepMap, null, NamedXContentRegistry.EMPTY, null, null);
        Step actualFirstStep = registry.getFirstStep(policyName);
        assertThat(actualFirstStep, sameInstance(expectedFirstStep));
    }

    public void testGetFirstStepUnknownPolicy() {
        String policyName = randomAlphaOfLengthBetween(2, 10);
        Step expectedFirstStep = new MockStep(MOCK_STEP_KEY, null);
        Map<String, Step> firstStepMap = Map.of(policyName, expectedFirstStep);
        PolicyStepsRegistry registry = new PolicyStepsRegistry(null, firstStepMap, null, NamedXContentRegistry.EMPTY, null, null);
        Step actualFirstStep = registry.getFirstStep(policyName + "unknown");
        assertNull(actualFirstStep);
    }

    /**
     * Block Logic: Step resolution from index state.
     * Invariant: Retrieves the specific step instance matching the phase/action/name defined in the execution context.
     */
    public void testGetStep() {
        Client client = mock(Client.class);
        Mockito.when(client.settings()).thenReturn(Settings.EMPTY);
        LifecyclePolicy policy = LifecyclePolicyTests.randomTimeseriesLifecyclePolicyWithAllPhases("policy");
        LifecyclePolicyMetadata policyMetadata = new LifecyclePolicyMetadata(policy, Map.of(), 1, randomNonNegativeLong());
        String phaseName = randomFrom(policy.getPhases().keySet());
        Phase phase = policy.getPhases().get(phaseName);
        PhaseExecutionInfo pei = new PhaseExecutionInfo(policy.getName(), phase, 1, randomNonNegativeLong());
        String phaseJson = Strings.toString(pei);
        LifecycleAction action = randomValueOtherThan(MigrateAction.DISABLED, () -> randomFrom(phase.getActions().values()));
        Step step = randomFrom(action.toSteps(client, phaseName, MOCK_STEP_KEY, null));
        LifecycleExecutionState.Builder lifecycleState = LifecycleExecutionState.builder();
        lifecycleState.setPhaseDefinition(phaseJson);
        IndexMetadata indexMetadata = IndexMetadata.builder("test")
            .settings(indexSettings(IndexVersion.current(), 1, 0).put(LifecycleSettings.LIFECYCLE_NAME, "policy"))
            .putCustom(ILM_CUSTOM_METADATA_KEY, lifecycleState.build().asMap())
            .build();
        SortedMap<String, LifecyclePolicyMetadata> metas = new TreeMap<>();
        metas.put("policy", policyMetadata);
        PolicyStepsRegistry registry = new PolicyStepsRegistry(metas, null, null, REGISTRY, client, null);
        Step actualStep = registry.getStep(indexMetadata, step.getKey());
        assertThat(actualStep.getKey(), equalTo(step.getKey()));
    }

    public void testGetStepErrorStep() {
        Step.StepKey errorStepKey = new Step.StepKey(randomAlphaOfLengthBetween(1, 10), randomAlphaOfLengthBetween(1, 10), ErrorStep.NAME);
        Step expectedStep = new ErrorStep(errorStepKey);
        Index index = new Index("test", "uuid");
        Map<Index, List<Step>> indexSteps = Map.of(index, List.of(expectedStep));
        PolicyStepsRegistry registry = new PolicyStepsRegistry(null, null, null, NamedXContentRegistry.EMPTY, null, null);
        Step actualStep = registry.getStep(emptyMetadata(index), errorStepKey);
        assertThat(actualStep, equalTo(expectedStep));
    }

    public void testGetStepUnknownPolicy() {
        PolicyStepsRegistry registry = new PolicyStepsRegistry(null, null, null, NamedXContentRegistry.EMPTY, null, null);
        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> registry.getStep(emptyMetadata(new Index("test", "uuid")), MOCK_STEP_KEY)
        );
        assertThat(
            e.getMessage(),
            containsString(
                "failed to retrieve step {\"phase\":\"mock\",\"action\":\"mock\",\"name\":\"mock\"}" + " as index [test] has no policy"
            )
        );
    }

    public void testGetStepForIndexWithNoPhaseGetsInitializationStep() {
        Client client = mock(Client.class);
        Mockito.when(client.settings()).thenReturn(Settings.EMPTY);
        LifecyclePolicy policy = LifecyclePolicyTests.randomTimeseriesLifecyclePolicy("policy");
        LifecyclePolicyMetadata policyMetadata = new LifecyclePolicyMetadata(policy, Map.of(), 1, randomNonNegativeLong());
        IndexMetadata indexMetadata = IndexMetadata.builder("test")
            .settings(indexSettings(IndexVersion.current(), 1, 0).put(LifecycleSettings.LIFECYCLE_NAME, "policy").build())
            .build();
        SortedMap<String, LifecyclePolicyMetadata> metas = new TreeMap<>();
        metas.put("policy", policyMetadata);
        PolicyStepsRegistry registry = new PolicyStepsRegistry(metas, null, null, REGISTRY, client, null);
        Step step = registry.getStep(indexMetadata, InitializePolicyContextStep.KEY);
        assertNotNull(step);
    }

    public void testGetStepUnknownStepKey() {
        Client client = mock(Client.class);
        Mockito.when(client.settings()).thenReturn(Settings.EMPTY);
        LifecyclePolicy policy = LifecyclePolicyTests.randomTimeseriesLifecyclePolicyWithAllPhases("policy");
        LifecyclePolicyMetadata policyMetadata = new LifecyclePolicyMetadata(policy, Map.of(), 1, randomNonNegativeLong());
        String phaseName = randomFrom(policy.getPhases().keySet());
        Phase phase = policy.getPhases().get(phaseName);
        PhaseExecutionInfo pei = new PhaseExecutionInfo(policy.getName(), phase, 1, randomNonNegativeLong());
        String phaseJson = Strings.toString(pei);
        LifecycleAction action = randomValueOtherThan(MigrateAction.DISABLED, () -> randomFrom(phase.getActions().values()));
        Step step = randomFrom(action.toSteps(client, phaseName, MOCK_STEP_KEY, null));
        LifecycleExecutionState.Builder lifecycleState = LifecycleExecutionState.builder();
        lifecycleState.setPhaseDefinition(phaseJson);
        IndexMetadata indexMetadata = IndexMetadata.builder("test")
            .settings(indexSettings(IndexVersion.current(), 1, 0).put(LifecycleSettings.LIFECYCLE_NAME, "policy").build())
            .putCustom(ILM_CUSTOM_METADATA_KEY, lifecycleState.build().asMap())
            .build();
        SortedMap<String, LifecyclePolicyMetadata> metas = new TreeMap<>();
        metas.put("policy", policyMetadata);
        PolicyStepsRegistry registry = new PolicyStepsRegistry(metas, null, null, REGISTRY, client, null);
        Step.StepKey badStepKey = new Step.StepKey(step.getKey().phase(), step.getKey().action(), step.getKey().name() + "-bad");
        assertNull(registry.getStep(indexMetadata, badStepKey));
        // repeat the test to make sure that nulls don't poison the registry's cache
        assertNull(registry.getStep(indexMetadata, badStepKey));
    }

    /**
     * Block Logic: Policy lifecycle progression.
     * Invariant: Updates to metadata correctly trigger step re-resolution across all defined phases.
     */
    public void testUpdateFromNothingToSomethingToNothing() throws Exception {
        Index index = new Index("test", "uuid");
        Client client = mock(Client.class);
        Mockito.when(client.settings()).thenReturn(Settings.EMPTY);
        String policyName = randomAlphaOfLength(5);
        LifecyclePolicy newPolicy = LifecyclePolicyTests.randomTestLifecyclePolicy(policyName);
        logger.info("--> policy: {}", newPolicy);
        List<Step> policySteps = newPolicy.toSteps(client, null);
        Map<String, String> headers = new HashMap<>();
        if (randomBoolean()) {
            headers.put(randomAlphaOfLength(10), randomAlphaOfLength(10));
            headers.put(randomAlphaOfLength(10), randomAlphaOfLength(10));
        }
        Map<String, LifecyclePolicyMetadata> policyMap = Map.of(
            newPolicy.getName(),
            new LifecyclePolicyMetadata(newPolicy, headers, randomNonNegativeLong(), randomNonNegativeLong())
        );
        IndexLifecycleMetadata lifecycleMetadata = new IndexLifecycleMetadata(policyMap, OperationMode.RUNNING);
        LifecycleExecutionState.Builder lifecycleState = LifecycleExecutionState.builder();
        lifecycleState.setPhase("new");
        ProjectMetadata currentProject = ProjectMetadata.builder(randomProjectIdOrDefault())
            .putCustom(IndexLifecycleMetadata.TYPE, lifecycleMetadata)
            .put(
                IndexMetadata.builder("test")
                    .settings(
                        indexSettings(1, 0).put("index.uuid", "uuid")
                            .put(IndexMetadata.SETTING_VERSION_CREATED, IndexVersion.current())
                            .put(LifecycleSettings.LIFECYCLE_NAME, policyName)
                    )
                    .putCustom(ILM_CUSTOM_METADATA_KEY, lifecycleState.build().asMap())
            )
            .build();

        PolicyStepsRegistry registry = new PolicyStepsRegistry(NamedXContentRegistry.EMPTY, client, null);
        registry.update(currentProject.custom(IndexLifecycleMetadata.TYPE));

        assertThat(registry.getFirstStep(newPolicy.getName()), equalTo(policySteps.get(0)));
        assertThat(registry.getLifecyclePolicyMap().size(), equalTo(1));
        assertNotNull(registry.getLifecyclePolicyMap().get(newPolicy.getName()));
        assertThat(registry.getLifecyclePolicyMap().get(newPolicy.getName()).getHeaders(), equalTo(headers));
        assertThat(registry.getFirstStepMap().size(), equalTo(1));
        assertThat(registry.getStepMap().size(), equalTo(1));
        Map<Step.StepKey, Step> registeredStepsForPolicy = registry.getStepMap().get(newPolicy.getName());
        assertThat(registeredStepsForPolicy.size(), equalTo(policySteps.size()));
        
        /**
         * Block Logic: Sequential step validation.
         */
        for (Step step : policySteps) {
            LifecycleExecutionState.Builder newIndexState = LifecycleExecutionState.builder();
            newIndexState.setPhase(step.getKey().phase());
            currentProject = ProjectMetadata.builder(currentProject)
                .put(
                    IndexMetadata.builder(currentProject.index("test"))
                        .settings(Settings.builder().put(currentProject.index("test").getSettings()))
                        .putCustom(ILM_CUSTOM_METADATA_KEY, newIndexState.build().asMap())
                )
                .build();
            registry.update(currentProject.custom(IndexLifecycleMetadata.TYPE));
            assertThat(registeredStepsForPolicy.get(step.getKey()), equalTo(step));
            assertThat(registry.getStep(currentProject.index(index), step.getKey()), equalTo(step));
        }

        Map<String, LifecyclePolicyMetadata> registryPolicyMap = registry.getLifecyclePolicyMap();
        Map<String, Step> registryFirstStepMap = registry.getFirstStepMap();
        Map<String, Map<Step.StepKey, Step>> registryStepMap = registry.getStepMap();
        registry.update(currentProject.custom(IndexLifecycleMetadata.TYPE));
        assertThat(registry.getLifecyclePolicyMap(), equalTo(registryPolicyMap));
        assertThat(registry.getFirstStepMap(), equalTo(registryFirstStepMap));
        assertThat(registry.getStepMap(), equalTo(registryStepMap));

        lifecycleMetadata = new IndexLifecycleMetadata(Map.of(), OperationMode.RUNNING);
        currentProject = ProjectMetadata.builder(currentProject).putCustom(IndexLifecycleMetadata.TYPE, lifecycleMetadata).build();
        registry.update(currentProject.custom(IndexLifecycleMetadata.TYPE));
        assertTrue(registry.getLifecyclePolicyMap().isEmpty());
        assertTrue(registry.getFirstStepMap().isEmpty());
        assertTrue(registry.getStepMap().isEmpty());
    }

    public void testUpdateChangedPolicy() {
        Client client = mock(Client.class);
        Mockito.when(client.settings()).thenReturn(Settings.EMPTY);
        String policyName = randomAlphaOfLengthBetween(5, 10);
        LifecyclePolicy newPolicy = LifecyclePolicyTests.randomTestLifecyclePolicy(policyName);
        Map<String, String> headers = new HashMap<>();
        if (randomBoolean()) {
            headers.put(randomAlphaOfLength(10), randomAlphaOfLength(10));
            headers.put(randomAlphaOfLength(10), randomAlphaOfLength(10));
        }
        Map<String, LifecyclePolicyMetadata> policyMap = Map.of(
            newPolicy.getName(),
            new LifecyclePolicyMetadata(newPolicy, headers, randomNonNegativeLong(), randomNonNegativeLong())
        );
        IndexLifecycleMetadata lifecycleMetadata = new IndexLifecycleMetadata(policyMap, OperationMode.RUNNING);
        ProjectMetadata currentProject = ProjectMetadata.builder(randomProjectIdOrDefault())
            .putCustom(IndexLifecycleMetadata.TYPE, lifecycleMetadata)
            .build();
        PolicyStepsRegistry registry = new PolicyStepsRegistry(NamedXContentRegistry.EMPTY, client, null);
        registry.update(currentProject.custom(IndexLifecycleMetadata.TYPE));

        newPolicy = LifecyclePolicyTests.randomTestLifecyclePolicy(policyName);
        lifecycleMetadata = new IndexLifecycleMetadata(
            Map.of(policyName, new LifecyclePolicyMetadata(newPolicy, Map.of(), randomNonNegativeLong(), randomNonNegativeLong())),
            OperationMode.RUNNING
        );
        currentProject = ProjectMetadata.builder(currentProject).putCustom(IndexLifecycleMetadata.TYPE, lifecycleMetadata).build();
        registry.update(currentProject.custom(IndexLifecycleMetadata.TYPE));
    }

    /**
     * Block Logic: Immutable phase snapshot validation.
     * Invariant: Ensures that indices stick to their phase definition even if the global policy is updated, 
     * unless the index itself transitions to a new phase.
     */
    public void testUpdatePolicyButNoPhaseChangeIndexStepsDontChange() throws Exception {
        Index index = new Index("test", "uuid");
        Client client = mock(Client.class);
        Mockito.when(client.settings()).thenReturn(Settings.EMPTY);
        String policyName = randomAlphaOfLength(5);
        Map<String, LifecycleAction> actions = new HashMap<>();
        actions.put("shrink", new ShrinkAction(1, null, false));
        Map<String, Phase> phases = new HashMap<>();
        Phase warmPhase = new Phase("warm", TimeValue.ZERO, actions);
        PhaseExecutionInfo pei = new PhaseExecutionInfo(policyName, warmPhase, 1, randomNonNegativeLong());
        String phaseJson = Strings.toString(pei);
        phases.put("warm", new Phase("warm", TimeValue.ZERO, actions));
        LifecyclePolicy newPolicy = new LifecyclePolicy(policyName, phases);
        
        actions = new HashMap<>();
        actions.put("shrink", new ShrinkAction(2, null, false));
        phases = new HashMap<>();
        phases.put("warm", new Phase("warm", TimeValue.ZERO, actions));
        LifecyclePolicy updatedPolicy = new LifecyclePolicy(policyName, phases);
        
        Map<String, String> headers = new HashMap<>();
        Map<String, LifecyclePolicyMetadata> policyMap = Map.of(
            newPolicy.getName(),
            new LifecyclePolicyMetadata(newPolicy, headers, randomNonNegativeLong(), randomNonNegativeLong())
        );
        IndexLifecycleMetadata lifecycleMetadata = new IndexLifecycleMetadata(policyMap, OperationMode.RUNNING);
        LifecycleExecutionState.Builder lifecycleState = LifecycleExecutionState.builder();
        lifecycleState.setPhase("warm");
        lifecycleState.setPhaseDefinition(phaseJson);
        ProjectMetadata currentProject = ProjectMetadata.builder(randomProjectIdOrDefault())
            .putCustom(IndexLifecycleMetadata.TYPE, lifecycleMetadata)
            .put(
                IndexMetadata.builder("test")
                    .settings(
                        indexSettings(1, 0).put("index.uuid", "uuid")
                            .put(IndexMetadata.SETTING_VERSION_CREATED, IndexVersion.current())
                            .put(LifecycleSettings.LIFECYCLE_NAME, policyName)
                    )
                    .putCustom(ILM_CUSTOM_METADATA_KEY, lifecycleState.build().asMap())
            )
            .build();

        PolicyStepsRegistry registry = new PolicyStepsRegistry(REGISTRY, client, null);
        registry.update(currentProject.custom(IndexLifecycleMetadata.TYPE));

        Map<Step.StepKey, Step> registeredStepsForPolicy = registry.getStepMap().get(newPolicy.getName());
        Step shrinkStep = registeredStepsForPolicy.entrySet()
            .stream()
            .filter(e -> e.getKey().phase().equals("warm") && e.getKey().name().equals("shrink"))
            .findFirst()
            .get()
            .getValue();
        Step gotStep = registry.getStep(currentProject.index(index), shrinkStep.getKey());
        assertThat(((ShrinkStep) shrinkStep).getNumberOfShards(), equalTo(1));
        assertThat(((ShrinkStep) gotStep).getNumberOfShards(), equalTo(1));

        policyMap = Map.of(
            updatedPolicy.getName(),
            new LifecyclePolicyMetadata(updatedPolicy, headers, randomNonNegativeLong(), randomNonNegativeLong())
        );
        lifecycleMetadata = new IndexLifecycleMetadata(policyMap, OperationMode.RUNNING);
        currentProject = ProjectMetadata.builder(currentProject).putCustom(IndexLifecycleMetadata.TYPE, lifecycleMetadata).build();

        registry.update(currentProject.custom(IndexLifecycleMetadata.TYPE));

        registeredStepsForPolicy = registry.getStepMap().get(newPolicy.getName());
        shrinkStep = registeredStepsForPolicy.entrySet()
            .stream()
            .filter(e -> e.getKey().phase().equals("warm") && e.getKey().name().equals("shrink"))
            .findFirst()
            .get()
            .getValue();
        gotStep = registry.getStep(currentProject.index(index), shrinkStep.getKey());
        assertThat(((ShrinkStep) shrinkStep).getNumberOfShards(), equalTo(2));
        assertThat(((ShrinkStep) gotStep).getNumberOfShards(), equalTo(1));
    }

    /**
     * Block Logic: Concurrency stress test.
     * Invariant: Registry lookups remain consistent even when the internal cache is being frequently invalidated 
     * by background metadata updates.
     */
    public void testGetStepMultithreaded() throws Exception {
        Client client = mock(Client.class);
        Mockito.when(client.settings()).thenReturn(Settings.EMPTY);

        LifecyclePolicy policy = LifecyclePolicyTests.randomTimeseriesLifecyclePolicyWithAllPhases("policy");
        String phaseName = randomFrom(policy.getPhases().keySet());
        Phase phase = policy.getPhases().get(phaseName);

        LifecycleExecutionState lifecycleState = LifecycleExecutionState.builder()
            .setPhaseDefinition(Strings.toString(new PhaseExecutionInfo(policy.getName(), phase, 1, randomNonNegativeLong())))
            .build();
        IndexMetadata indexMetadata = IndexMetadata.builder("test")
            .settings(indexSettings(IndexVersion.current(), 1, 0).put(LifecycleSettings.LIFECYCLE_NAME, "policy").build())
            .putCustom(ILM_CUSTOM_METADATA_KEY, lifecycleState.asMap())
            .build();

        SortedMap<String, LifecyclePolicyMetadata> metas = new TreeMap<>();
        metas.put("policy", new LifecyclePolicyMetadata(policy, Map.of(), 1, randomNonNegativeLong()));
        IndexLifecycleMetadata meta = new IndexLifecycleMetadata(metas, OperationMode.RUNNING);

        PolicyStepsRegistry registry = new PolicyStepsRegistry(REGISTRY, client, null);
        registry.update(meta);

        for (int i = 0; i < scaledRandomIntBetween(100, 1000); i++) {
            LifecycleAction action = randomValueOtherThan(MigrateAction.DISABLED, () -> randomFrom(phase.getActions().values()));
            Step step = randomFrom(action.toSteps(client, phaseName, MOCK_STEP_KEY, null));
            Step actualStep = registry.getStep(indexMetadata, step.getKey());
            assertThat(actualStep.getKey(), equalTo(step.getKey()));
        }

        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicBoolean done = new AtomicBoolean(false);

        Thread t = new Thread(() -> {
            latch.countDown(); 
            while (done.get() == false) {
                registry.update(meta);
            }
        });
        t.start();

        try {
            latch.await(); 

            for (int i = 0; i < scaledRandomIntBetween(100, 1000); i++) {
                LifecycleAction action = randomValueOtherThan(MigrateAction.DISABLED, () -> randomFrom(phase.getActions().values()));
                Step step = randomFrom(action.toSteps(client, phaseName, MOCK_STEP_KEY, null));
                Step actualStep = registry.getStep(indexMetadata, step.getKey());
                assertThat(actualStep.getKey(), equalTo(step.getKey()));
            }
        } finally {
            done.set(true);
            t.join(1000);
        }
    }
}
