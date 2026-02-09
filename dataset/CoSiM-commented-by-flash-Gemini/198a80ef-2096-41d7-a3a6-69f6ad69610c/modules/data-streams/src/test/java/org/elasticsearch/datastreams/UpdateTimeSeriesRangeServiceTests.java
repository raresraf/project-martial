/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.datastreams;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.filter.RegexFilter;
import org.apache.logging.log4j.message.Message;
import org.elasticsearch.cluster.ClusterState;
import org.elasticsearch.cluster.metadata.DataStream;
import org.elasticsearch.cluster.metadata.DataStreamTestHelper;
import org.elasticsearch.cluster.metadata.ProjectId;
import org.elasticsearch.cluster.metadata.ProjectMetadata;
import org.elasticsearch.cluster.service.ClusterService;
import org.elasticsearch.common.logging.Loggers;
import org.elasticsearch.common.settings.ClusterSettings;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.core.Tuple;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.threadpool.TestThreadPool;
import org.elasticsearch.threadpool.ThreadPool;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;

import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalAmount;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.elasticsearch.cluster.metadata.DataStream.getDefaultBackingIndexName;
import static org.elasticsearch.cluster.metadata.DataStreamTestHelper.createIndexMetadata;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.nullValue;
import static org.hamcrest.Matchers.sameInstance;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * @brief Functional description of the UpdateTimeSeriesRangeServiceTests class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class UpdateTimeSeriesRangeServiceTests extends ESTestCase {

    /**
     * @brief [Functional description for field appender]: Describe purpose here.
     */
    static MockAppender appender;
    static Logger testLogger1 = LogManager.getLogger(UpdateTimeSeriesRangeService.class);

    @BeforeClass
    /**
     * @brief [Functional Utility for classInit]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IllegalAccessException: [Description]
     */
    public static void classInit() throws IllegalAccessException {
        appender = new MockAppender("mock_appender");
        appender.start();
        Loggers.addAppender(testLogger1, appender);
    }

    @AfterClass
    /**
     * @brief [Functional Utility for classCleanup]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public static void classCleanup() {
        Loggers.removeAppender(testLogger1, appender);
        appender.stop();
    }

    /**
     * @brief [Functional description for field threadPool]: Describe purpose here.
     */
    private ThreadPool threadPool;
    /**
     * @brief [Functional description for field instance]: Describe purpose here.
     */
    private UpdateTimeSeriesRangeService instance;

    @Before
    /**
     * @brief [Functional Utility for createInstance]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void createInstance() {
        ClusterService mockClusterService = mock(ClusterService.class);
        ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, Set.of(DataStreamsPlugin.TIME_SERIES_POLL_INTERVAL));
        when(mockClusterService.getClusterSettings()).thenReturn(clusterSettings);
        threadPool = new TestThreadPool(getTestName());
        instance = new UpdateTimeSeriesRangeService(Settings.EMPTY, threadPool, mockClusterService);
    }

    @After
    /**
     * @brief [Functional Utility for cleanup]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void cleanup() throws Exception {
        instance.doClose();
        terminate(threadPool);
    }

    /**
     * @brief [Functional Utility for testUpdateTimeSeriesTemporalRange]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testUpdateTimeSeriesTemporalRange() {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = "logs-app1";
        Instant now = Instant.now().truncatedTo(ChronoUnit.MILLIS);
        Instant start = now.minus(2, ChronoUnit.HOURS);
        Instant end = now.plus(40, ChronoUnit.MINUTES);
        final var projectId = randomProjectIdOrDefault();
        final var metadata = DataStreamTestHelper.getProjectWithDataStream(
            projectId,
            dataStreamName,
            List.of(new Tuple<>(start.minus(4, ChronoUnit.HOURS), start), new Tuple<>(start, end))
        );

        // noop, because current end_time isn't passed now + look_a_head_time + poll_interval
        ClusterState in = ClusterState.builder(ClusterState.EMPTY_STATE).putProjectMetadata(metadata).build();
        ClusterState result = instance.updateTimeSeriesTemporalRange(in, now);
        assertThat(result, sameInstance(in));
        var project = result.getMetadata().getProject(projectId);
        Instant previousStartTime1 = getStartTime(project, dataStreamName, 0);
        Instant previousEndTime1 = getEndTime(project, dataStreamName, 0);
        Instant previousStartTime2 = getStartTime(project, dataStreamName, 1);
        Instant previousEndTime2 = getEndTime(project, dataStreamName, 1);

        // updates end time of most recent backing index only, because current time is passed current end_time + look_a_head_time and
        // poll_interval
        now = now.plus(1, ChronoUnit.HOURS);
        in = ClusterState.builder(ClusterState.EMPTY_STATE).putProjectMetadata(metadata).build();
        result = instance.updateTimeSeriesTemporalRange(in, now);
        assertThat(result, not(sameInstance(in)));
        project = result.getMetadata().getProject(projectId);
        assertThat(getStartTime(project, dataStreamName, 0), equalTo(previousStartTime1));
        assertThat(getEndTime(project, dataStreamName, 0), equalTo(previousEndTime1));
        assertThat(getStartTime(project, dataStreamName, 1), equalTo(previousStartTime2));
        assertThat(getEndTime(project, dataStreamName, 1), not(equalTo(previousEndTime2)));
        assertThat(
            getEndTime(project, dataStreamName, 1),
            equalTo(now.plus(30, ChronoUnit.MINUTES).plus(5, ChronoUnit.MINUTES).truncatedTo(ChronoUnit.SECONDS))
        );
    }

    /**
     * @brief [Functional Utility for testUpdateTimeSeriesTemporalRange_customLookAHeadTime]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testUpdateTimeSeriesTemporalRange_customLookAHeadTime() {
        int lookAHeadTimeMinutes = randomIntBetween(30, 120);
        TemporalAmount lookAHeadTime = Duration.ofMinutes(lookAHeadTimeMinutes);
        int timeSeriesPollIntervalMinutes = randomIntBetween(1, 10);
        TemporalAmount timeSeriesPollInterval = Duration.ofMinutes(timeSeriesPollIntervalMinutes);
        instance.setPollInterval(TimeValue.timeValueMinutes(timeSeriesPollIntervalMinutes));

    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = "logs-app1";
        Instant now = Instant.now().truncatedTo(ChronoUnit.MILLIS);
        Instant start = now.minus(2, ChronoUnit.HOURS);
        Instant end = now.plus(1, ChronoUnit.HOURS);
        final var projectId = randomProjectIdOrDefault();
        final var projectBuilder = ProjectMetadata.builder(projectId);
        DataStreamTestHelper.getClusterStateWithDataStream(
            projectBuilder,
            dataStreamName,
            List.of(new Tuple<>(start.minus(4, ChronoUnit.HOURS), start), new Tuple<>(start, end))
        );
        final var metadata = projectBuilder.updateSettings(
            Settings.builder().put(DataStreamsPlugin.LOOK_AHEAD_TIME.getKey(), lookAHeadTimeMinutes + "m").build()
        ).build();

        var in = ClusterState.builder(ClusterState.EMPTY_STATE).putProjectMetadata(metadata).build();
        var project = in.getMetadata().getProject(projectId);
        Instant previousStartTime1 = getStartTime(project, dataStreamName, 0);
        Instant previousEndTime1 = getEndTime(project, dataStreamName, 0);
        Instant previousStartTime2 = getStartTime(project, dataStreamName, 1);

        now = now.plus(1, ChronoUnit.HOURS);
        var result = instance.updateTimeSeriesTemporalRange(in, now);
        assertThat(result, not(sameInstance(in)));
        project = result.getMetadata().getProject(projectId);
        assertThat(getStartTime(project, dataStreamName, 0), equalTo(previousStartTime1));
        assertThat(getEndTime(project, dataStreamName, 0), equalTo(previousEndTime1));
        assertThat(getStartTime(project, dataStreamName, 1), equalTo(previousStartTime2));
        assertThat(
            getEndTime(project, dataStreamName, 1),
            equalTo(now.plus(lookAHeadTime).plus(timeSeriesPollInterval).truncatedTo(ChronoUnit.SECONDS))
        );
    }

    /**
     * @brief [Functional Utility for testUpdateTimeSeriesTemporalRange_NoUpdateBecauseReplicated]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testUpdateTimeSeriesTemporalRange_NoUpdateBecauseReplicated() {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = "logs-app1";
        Instant now = Instant.now().truncatedTo(ChronoUnit.MILLIS);
        Instant start = now.minus(2, ChronoUnit.HOURS);
        Instant end = now.plus(31, ChronoUnit.MINUTES);
        final var projectId = randomProjectIdOrDefault();
        final var projectBuilder = ProjectMetadata.builder(projectId);
        DataStreamTestHelper.getClusterStateWithDataStream(
            projectBuilder,
            dataStreamName,
            List.of(new Tuple<>(start.minus(4, ChronoUnit.HOURS), start), new Tuple<>(start, end))
        );
        DataStream d = projectBuilder.dataStream(dataStreamName);
        final var metadata = projectBuilder.put(
            d.copy().setReplicated(true).setBackingIndices(d.getDataComponent().copy().setRolloverOnWrite(false).build()).build()
        ).build();

        now = now.plus(1, ChronoUnit.HOURS);
        ClusterState in = ClusterState.builder(ClusterState.EMPTY_STATE).putProjectMetadata(metadata).build();
        ClusterState result = instance.updateTimeSeriesTemporalRange(in, now);
        assertThat(result, sameInstance(in));
    }

    /**
     * @brief [Functional Utility for testUpdateTimeSeriesTemporalRange_NoUpdateBecauseRegularDataStream]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testUpdateTimeSeriesTemporalRange_NoUpdateBecauseRegularDataStream() {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = "logs-app1";
        Instant now = Instant.now().truncatedTo(ChronoUnit.MILLIS);
        ClusterState in = DataStreamTestHelper.getClusterStateWithDataStreams(List.of(new Tuple<>(dataStreamName, 2)), List.of());

        now = now.plus(1, ChronoUnit.HOURS);
        ClusterState result = instance.updateTimeSeriesTemporalRange(in, now);
        assertThat(result, sameInstance(in));
    }

    /**
     * @brief [Functional Utility for testUpdateTimeSeriesTemporalRangeMultipleDataStream]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testUpdateTimeSeriesTemporalRangeMultipleDataStream() {
    /**
     * @brief [Functional description for field dataStreamName1]: Describe purpose here.
     */
        String dataStreamName1 = "logs-app1";
    /**
     * @brief [Functional description for field dataStreamName2]: Describe purpose here.
     */
        String dataStreamName2 = "logs-app2";
    /**
     * @brief [Functional description for field dataStreamName3]: Describe purpose here.
     */
        String dataStreamName3 = "logs-app3";
        Instant now = Instant.now().truncatedTo(ChronoUnit.MILLIS);

        Instant start = now.minus(90, ChronoUnit.MINUTES);
        final var projectId = randomProjectIdOrDefault();
        ProjectMetadata.Builder mbBuilder = ProjectMetadata.builder(projectId);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (String dataStreamName : List.of(dataStreamName1, dataStreamName2, dataStreamName3)) {
            Instant end = start.plus(30, ChronoUnit.MINUTES);
            DataStreamTestHelper.getClusterStateWithDataStream(mbBuilder, dataStreamName, List.of(new Tuple<>(start, end)));
            start = end;
        }

        now = now.minus(45, ChronoUnit.MINUTES);
        ClusterState before = ClusterState.builder(ClusterState.EMPTY_STATE).putProjectMetadata(mbBuilder).build();
        ClusterState result = instance.updateTimeSeriesTemporalRange(before, now);
        assertThat(result, not(sameInstance(before)));
        final var project = result.getMetadata().getProject(projectId);
        final var expectedEndTime = now.plus(35, ChronoUnit.MINUTES).truncatedTo(ChronoUnit.SECONDS);
        assertThat(getEndTime(project, dataStreamName1, 0), equalTo(expectedEndTime));
        assertThat(getEndTime(project, dataStreamName2, 0), equalTo(expectedEndTime));
        assertThat(getEndTime(project, dataStreamName3, 0), equalTo(start));
    }

    /**
     * @brief [Functional Utility for testUpdateTimeSeriesTemporalOneBadDataStream]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testUpdateTimeSeriesTemporalOneBadDataStream() {
    /**
     * @brief [Functional description for field dataStreamName1]: Describe purpose here.
     */
        String dataStreamName1 = "logs-app1";
    /**
     * @brief [Functional description for field dataStreamName2]: Describe purpose here.
     */
        String dataStreamName2 = "logs-app2-broken";
    /**
     * @brief [Functional description for field dataStreamName3]: Describe purpose here.
     */
        String dataStreamName3 = "logs-app3";
        Instant now = Instant.now().truncatedTo(ChronoUnit.MILLIS);

        Instant start = now.minus(90, ChronoUnit.MINUTES);
        Instant end = start.plus(30, ChronoUnit.MINUTES);
        final var projectId = randomProjectIdOrDefault();
        ProjectMetadata.Builder mbBuilder = ProjectMetadata.builder(projectId);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (String dataStreamName : List.of(dataStreamName1, dataStreamName2, dataStreamName3)) {
            DataStreamTestHelper.getClusterStateWithDataStream(mbBuilder, dataStreamName, List.of(new Tuple<>(start, end)));
        }

        Settings settings = Settings.builder().put("index.mode", "logsdb").build();
        var im = createIndexMetadata(getDefaultBackingIndexName(dataStreamName2, 2, start.toEpochMilli()), true, settings, 0);
        mbBuilder.put(im, true);
        var ds2 = mbBuilder.dataStreamMetadata().dataStreams().get(dataStreamName2);
        var ds2Indices = new ArrayList<>(ds2.getIndices());
        ds2Indices.add(im.getIndex());
        var copy = new HashMap<>(mbBuilder.dataStreamMetadata().dataStreams());
        copy.put(
            dataStreamName2,
            new DataStream(
                ds2.getName(),
                ds2Indices,
                2,
                ds2.getMetadata(),
                ds2.getSettings(),
                ds2.getMappings(),
                ds2.isHidden(),
                ds2.isReplicated(),
                ds2.isSystem(),
                ds2.isAllowCustomRouting(),
                ds2.getIndexMode(),
                ds2.getDataLifecycle(),
                ds2.getDataStreamOptions(),
                ds2.getFailureIndices(),
                ds2.rolloverOnWrite(),
                ds2.getAutoShardingEvent()
            )
        );
        mbBuilder.dataStreams(copy, Map.of());

        now = now.minus(45, ChronoUnit.MINUTES);
        ClusterState before = ClusterState.builder(ClusterState.EMPTY_STATE).putProjectMetadata(mbBuilder).build();
        ClusterState result = instance.updateTimeSeriesTemporalRange(before, now);
        assertThat(result, not(sameInstance(before)));
        final var project = result.getMetadata().getProject(projectId);
        final var expectedEndTime = now.plus(35, ChronoUnit.MINUTES).truncatedTo(ChronoUnit.SECONDS);
        assertThat(getEndTime(project, dataStreamName1, 0), equalTo(expectedEndTime));
        assertThat(getEndTime(project, dataStreamName2, 0), equalTo(end)); // failed to update end_time, because broken data stream
        assertThat(getEndTime(project, dataStreamName3, 0), equalTo(expectedEndTime));

        String message = appender.getLastEventAndReset().getMessage().getFormattedMessage();
        assertThat(
            message,
            equalTo(
                "unable to update [index.time_series.end_time] for data stream [logs-app2-broken] and "
                    + "backing index ["
                    + im.getIndex().getName()
                    + "]"
            )
        );
    }

    /**
     * @brief [Functional Utility for testUpdateTimeSeriesTemporalRange_multipleProjects]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testUpdateTimeSeriesTemporalRange_multipleProjects() {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = "logs-app1";
        Instant now = Instant.now().truncatedTo(ChronoUnit.MILLIS);
        Instant start = now.minus(90, ChronoUnit.MINUTES);
        Instant end = now.plus(40, ChronoUnit.MINUTES);
        final var projectIds = randomList(1, 3, ESTestCase::randomProjectIdOrDefault);
        final var builder = ClusterState.builder(ClusterState.EMPTY_STATE);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (ProjectId projectId : projectIds) {
            builder.putProjectMetadata(
                DataStreamTestHelper.getProjectWithDataStream(projectId, dataStreamName, List.of(new Tuple<>(start, end)))
            );
        }

        now = now.plus(1, ChronoUnit.HOURS);
        final ClusterState in = builder.build();
        final ClusterState result = instance.updateTimeSeriesTemporalRange(in, now);
        assertThat(result, not(sameInstance(in)));
        final var expectedEndTime = now.plus(35, ChronoUnit.MINUTES).truncatedTo(ChronoUnit.SECONDS);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (ProjectId projectId : projectIds) {
            final var project = result.getMetadata().getProject(projectId);
            assertThat(getStartTime(project, dataStreamName, 0), equalTo(start));
            assertThat(getEndTime(project, dataStreamName, 0), equalTo(expectedEndTime));
        }
    }

    /**
     * @brief [Functional Utility for testUpdatePollInterval]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testUpdatePollInterval() {
        instance.scheduleTask();
        assertThat(instance.pollInterval, equalTo(TimeValue.timeValueMinutes(5)));
        assertThat(instance.job.toString(), containsString("5m"));
        instance.setPollInterval(TimeValue.timeValueMinutes(1));
        assertThat(instance.pollInterval, equalTo(TimeValue.timeValueMinutes(1)));
        assertThat(instance.job.toString(), containsString("1m"));
    }

    /**
     * @brief [Functional Utility for testUpdatePollIntervalUnscheduled]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testUpdatePollIntervalUnscheduled() {
        assertThat(instance.pollInterval, equalTo(TimeValue.timeValueMinutes(5)));
        assertThat(instance.job, nullValue());
        instance.setPollInterval(TimeValue.timeValueMinutes(1));
        assertThat(instance.pollInterval, equalTo(TimeValue.timeValueMinutes(1)));
        assertThat(instance.job, nullValue());
    }

    /**
     * @brief [Functional Utility for getEndTime]: Describe purpose here.
     * @param project: [Description]
     * @param dataStreamName: [Description]
     * @param index: [Description]
     * @return [ReturnType]: [Description]
     */
    static Instant getEndTime(ProjectMetadata project, String dataStreamName, int index) {
        DataStream dataStream = project.dataStreams().get(dataStreamName);
        Settings indexSettings = project.index(dataStream.getIndices().get(index)).getSettings();
        return IndexSettings.TIME_SERIES_END_TIME.get(indexSettings);
    }

    /**
     * @brief [Functional Utility for getStartTime]: Describe purpose here.
     * @param project: [Description]
     * @param dataStreamName: [Description]
     * @param index: [Description]
     * @return [ReturnType]: [Description]
     */
    static Instant getStartTime(ProjectMetadata project, String dataStreamName, int index) {
        DataStream dataStream = project.dataStreams().get(dataStreamName);
        Settings indexSettings = project.index(dataStream.getIndices().get(index)).getSettings();
        return IndexSettings.TIME_SERIES_START_TIME.get(indexSettings);
    }

    static class MockAppender extends AbstractAppender {
    /**
     * @brief [Functional description for field lastEvent]: Describe purpose here.
     */
        public LogEvent lastEvent;

        MockAppender(final String name) throws IllegalAccessException {
            super(name, RegexFilter.createFilter(".*(\n.*)*", new String[0], false, null, null), null, false);
        }

        @Override
    /**
     * @brief [Functional Utility for append]: Describe purpose here.
     * @param event: [Description]
     * @return [ReturnType]: [Description]
     */
        public void append(LogEvent event) {
            lastEvent = event.toImmutable();
        }

    /**
     * @brief [Functional Utility for lastMessage]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        Message lastMessage() {
            return lastEvent.getMessage();
        }

    /**
     * @brief [Functional Utility for getLastEventAndReset]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public LogEvent getLastEventAndReset() {
    /**
     * @brief [Functional description for field toReturn]: Describe purpose here.
     */
            LogEvent toReturn = lastEvent;
            lastEvent = null;
    /**
     * @brief [Functional description for field toReturn]: Describe purpose here.
     */
            return toReturn;
        }
    }

}
