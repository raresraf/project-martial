/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.datastreams.action;

import org.elasticsearch.ResourceNotFoundException;
import org.elasticsearch.action.datastreams.DeleteDataStreamAction;
import org.elasticsearch.cluster.ClusterState;
import org.elasticsearch.cluster.SnapshotsInProgress;
import org.elasticsearch.cluster.metadata.DataStream;
import org.elasticsearch.cluster.metadata.DataStreamTestHelper;
import org.elasticsearch.cluster.metadata.IndexNameExpressionResolver;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.collect.ImmutableOpenMap;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.util.concurrent.ThreadContext;
import org.elasticsearch.core.Tuple;
import org.elasticsearch.index.Index;
import org.elasticsearch.indices.EmptySystemIndices;
import org.elasticsearch.indices.TestIndexNameExpressionResolver;
import org.elasticsearch.snapshots.Snapshot;
import org.elasticsearch.snapshots.SnapshotId;
import org.elasticsearch.snapshots.SnapshotInProgressException;
import org.elasticsearch.test.ESTestCase;

import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.sameInstance;

/**
 * @brief Functional description of the TransportDeleteDataStreamActionTests class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class TransportDeleteDataStreamActionTests extends ESTestCase {

    private final IndexNameExpressionResolver iner = TestIndexNameExpressionResolver.newInstance();
    private final ThreadContext threadContext = new ThreadContext(Settings.EMPTY);
    private final Consumer<String> validator = s -> EmptySystemIndices.INSTANCE.validateDataStreamAccess(s, threadContext);

    /**
     * @brief [Functional Utility for testDeleteDataStream]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testDeleteDataStream() {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        final String dataStreamName = "my-data-stream";
        final List<String> otherIndices = randomSubsetOf(List.of("foo", "bar", "baz"));

        final var projectId = randomProjectIdOrDefault();
        ClusterState cs = DataStreamTestHelper.getClusterStateWithDataStreams(
            projectId,
            List.of(new Tuple<>(dataStreamName, 2)),
            otherIndices
        );
        DeleteDataStreamAction.Request req = new DeleteDataStreamAction.Request(TEST_REQUEST_TIMEOUT, new String[] { dataStreamName });
        ClusterState newState = TransportDeleteDataStreamAction.removeDataStream(
            iner,
            cs.projectState(projectId),
            req,
            validator,
            Settings.EMPTY
        );
        assertThat(newState.metadata().getProject(projectId).dataStreams().size(), equalTo(0));
        assertThat(newState.metadata().getProject(projectId).indices().size(), equalTo(otherIndices.size()));
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (String indexName : otherIndices) {
            assertThat(newState.metadata().getProject(projectId).indices().get(indexName).getIndex().getName(), equalTo(indexName));
        }
    }

    /**
     * @brief [Functional Utility for testDeleteDataStreamWithFailureStore]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testDeleteDataStreamWithFailureStore() {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        final String dataStreamName = "my-data-stream";
        final List<String> otherIndices = randomSubsetOf(List.of("foo", "bar", "baz"));

        final var projectId = randomProjectIdOrDefault();
        ClusterState cs = DataStreamTestHelper.getClusterStateWithDataStreams(
            projectId,
            List.of(new Tuple<>(dataStreamName, 2)),
            otherIndices,
            System.currentTimeMillis(),
            Settings.EMPTY,
            1,
            false
        );
        DeleteDataStreamAction.Request req = new DeleteDataStreamAction.Request(TEST_REQUEST_TIMEOUT, new String[] { dataStreamName });
        ClusterState newState = TransportDeleteDataStreamAction.removeDataStream(
            iner,
            cs.projectState(projectId),
            req,
            validator,
            Settings.EMPTY
        );
        assertThat(newState.metadata().getProject(projectId).dataStreams().size(), equalTo(0));
        assertThat(newState.metadata().getProject(projectId).indices().size(), equalTo(otherIndices.size()));
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (String indexName : otherIndices) {
            assertThat(newState.metadata().getProject(projectId).indices().get(indexName).getIndex().getName(), equalTo(indexName));
        }
    }

    /**
     * @brief [Functional Utility for testDeleteMultipleDataStreams]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testDeleteMultipleDataStreams() {
    /**
     * @brief [Functional description for field dataStreamNames]: Describe purpose here.
     */
        String[] dataStreamNames = { "foo", "bar", "baz", "eggplant" };
        final var projectId = randomProjectIdOrDefault();
        ClusterState cs = DataStreamTestHelper.getClusterStateWithDataStreams(
            projectId,
            List.of(
                new Tuple<>(dataStreamNames[0], randomIntBetween(1, 3)),
                new Tuple<>(dataStreamNames[1], randomIntBetween(1, 3)),
                new Tuple<>(dataStreamNames[2], randomIntBetween(1, 3)),
                new Tuple<>(dataStreamNames[3], randomIntBetween(1, 3))
            ),
            List.of()
        );

        DeleteDataStreamAction.Request req = new DeleteDataStreamAction.Request(TEST_REQUEST_TIMEOUT, new String[] { "ba*", "eggplant" });
        ClusterState newState = TransportDeleteDataStreamAction.removeDataStream(
            iner,
            cs.projectState(projectId),
            req,
            validator,
            Settings.EMPTY
        );
        assertThat(newState.metadata().getProject(projectId).dataStreams().size(), equalTo(1));
        DataStream remainingDataStream = newState.metadata().getProject(projectId).dataStreams().get(dataStreamNames[0]);
        assertNotNull(remainingDataStream);
        assertThat(newState.metadata().getProject(projectId).indices().size(), equalTo(remainingDataStream.getIndices().size()));
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (Index i : remainingDataStream.getIndices()) {
            assertThat(newState.metadata().getProject(projectId).indices().get(i.getName()).getIndex(), equalTo(i));
        }
    }

    /**
     * @brief [Functional Utility for testDeleteSnapshottingDataStream]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testDeleteSnapshottingDataStream() {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        final String dataStreamName = "my-data-stream1";
    /**
     * @brief [Functional description for field dataStreamName2]: Describe purpose here.
     */
        final String dataStreamName2 = "my-data-stream2";
        final List<String> otherIndices = randomSubsetOf(List.of("foo", "bar", "baz"));

        final var projectId = randomProjectIdOrDefault();
        ClusterState cs = DataStreamTestHelper.getClusterStateWithDataStreams(
            projectId,
            List.of(new Tuple<>(dataStreamName, 2), new Tuple<>(dataStreamName2, 2)),
            otherIndices
        );
        SnapshotsInProgress snapshotsInProgress = SnapshotsInProgress.EMPTY.withAddedEntry(createEntry(dataStreamName, "repo1", false))
            .withAddedEntry(createEntry(dataStreamName2, "repo2", true));
        ClusterState snapshotCs = ClusterState.builder(cs).putCustom(SnapshotsInProgress.TYPE, snapshotsInProgress).build();

        DeleteDataStreamAction.Request req = new DeleteDataStreamAction.Request(TEST_REQUEST_TIMEOUT, new String[] { dataStreamName });
        SnapshotInProgressException e = expectThrows(
            SnapshotInProgressException.class,
            () -> TransportDeleteDataStreamAction.removeDataStream(iner, snapshotCs.projectState(projectId), req, validator, Settings.EMPTY)
        );

        assertThat(
            e.getMessage(),
            equalTo(
                "Cannot delete data streams that are being snapshotted: [my-data-stream1]. Try again after "
                    + "snapshot finishes or cancel the currently running snapshot."
            )
        );
    }

    /**
     * @brief [Functional Utility for createEntry]: Describe purpose here.
     * @param dataStreamName: [Description]
     * @param repo: [Description]
     * @param partial: [Description]
     * @return [ReturnType]: [Description]
     */
    private SnapshotsInProgress.Entry createEntry(String dataStreamName, String repo, boolean partial) {
        return SnapshotsInProgress.Entry.snapshot(
            new Snapshot(repo, new SnapshotId("", "")),
            false,
            partial,
            SnapshotsInProgress.State.SUCCESS,
            Collections.emptyMap(),
            List.of(dataStreamName),
            Collections.emptyList(),
            0,
            1,
            ImmutableOpenMap.of(),
            null,
            null,
            null
        );
    }

    /**
     * @brief [Functional Utility for testDeleteNonexistentDataStream]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testDeleteNonexistentDataStream() {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        final String dataStreamName = "my-data-stream";
    /**
     * @brief [Functional description for field dataStreamNames]: Describe purpose here.
     */
        String[] dataStreamNames = { "foo", "bar", "baz", "eggplant" };
        final var projectId = randomProjectIdOrDefault();
        ClusterState cs = DataStreamTestHelper.getClusterStateWithDataStreams(
            projectId,
            List.of(
                new Tuple<>(dataStreamNames[0], randomIntBetween(1, 3)),
                new Tuple<>(dataStreamNames[1], randomIntBetween(1, 3)),
                new Tuple<>(dataStreamNames[2], randomIntBetween(1, 3)),
                new Tuple<>(dataStreamNames[3], randomIntBetween(1, 3))
            ),
            List.of()
        );

        expectThrows(
            ResourceNotFoundException.class,
            () -> TransportDeleteDataStreamAction.removeDataStream(
                iner,
                cs.projectState(projectId),
                new DeleteDataStreamAction.Request(TEST_REQUEST_TIMEOUT, new String[] { dataStreamName }),
                validator,
                Settings.EMPTY
            )
        );

        DeleteDataStreamAction.Request req = new DeleteDataStreamAction.Request(
            TEST_REQUEST_TIMEOUT,
            new String[] { dataStreamName + "*" }
        );
        ClusterState newState = TransportDeleteDataStreamAction.removeDataStream(
            iner,
            cs.projectState(projectId),
            req,
            validator,
            Settings.EMPTY
        );
        assertThat(newState, sameInstance(cs));
        assertThat(
            newState.metadata().getProject(projectId).dataStreams().size(),
            equalTo(cs.metadata().getProject(projectId).dataStreams().size())
        );
        assertThat(
            newState.metadata().getProject(projectId).dataStreams().keySet(),
            containsInAnyOrder(cs.metadata().getProject(projectId).dataStreams().keySet().toArray(Strings.EMPTY_ARRAY))
        );
    }

}
