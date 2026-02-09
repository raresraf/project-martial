/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.logsdb;

import org.elasticsearch.Version;
import org.elasticsearch.cluster.metadata.ComposableIndexTemplate;
import org.elasticsearch.cluster.metadata.ComposableIndexTemplateMetadata;
import org.elasticsearch.cluster.metadata.DataStream;
import org.elasticsearch.cluster.metadata.DataStreamTestHelper;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.cluster.metadata.Metadata;
import org.elasticsearch.cluster.metadata.MetadataIndexTemplateService;
import org.elasticsearch.cluster.metadata.ProjectMetadata;
import org.elasticsearch.cluster.metadata.Template;
import org.elasticsearch.common.compress.CompressedXContent;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.core.Tuple;
import org.elasticsearch.index.IndexMode;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.IndexSortConfig;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.MapperTestUtils;
import org.elasticsearch.index.mapper.SourceFieldMapper;
import org.elasticsearch.license.License;
import org.elasticsearch.license.LicenseService;
import org.elasticsearch.license.MockLicenseState;
import org.elasticsearch.test.ESTestCase;
import org.junit.Before;

import java.io.IOException;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.elasticsearch.common.settings.Settings.builder;
import static org.elasticsearch.xpack.logsdb.LogsdbLicenseServiceTests.createEnterpriseLicense;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * @brief Functional description of the LogsdbIndexModeSettingsProviderTests class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class LogsdbIndexModeSettingsProviderTests extends ESTestCase {

    private static final String DATA_STREAM_NAME = "logs-app1";
    public static final String DEFAULT_MAPPING = """
        {
            "_doc": {
                "properties": {
                    "@timestamp": {
                        "type": "date"
                    },
                    "message": {
                        "type": "keyword"
                    },
                    "host.name": {
                        "type": "keyword"
                    }
                }
            }
        }
        """;

    /**
     * @brief [Functional description for field logsdbLicenseService]: Describe purpose here.
     */
    private LogsdbLicenseService logsdbLicenseService;
    private final AtomicInteger newMapperServiceCounter = new AtomicInteger();

    @Before
    /**
     * @brief [Functional Utility for setup]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void setup() throws Exception {
        MockLicenseState licenseState = MockLicenseState.createMock();
        when(licenseState.isAllowed(any())).thenReturn(true);
        var mockLicenseService = mock(LicenseService.class);
        License license = createEnterpriseLicense();
        when(mockLicenseService.getLicense()).thenReturn(license);
        logsdbLicenseService = new LogsdbLicenseService(Settings.EMPTY);
        logsdbLicenseService.setLicenseState(licenseState);
        logsdbLicenseService.setLicenseService(mockLicenseService);
    }

    /**
     * @brief [Functional Utility for withSyntheticSourceDemotionSupport]: Describe purpose here.
     * @param enabled: [Description]
     * @return [ReturnType]: [Description]
     */
    private LogsdbIndexModeSettingsProvider withSyntheticSourceDemotionSupport(boolean enabled) {
        return withSyntheticSourceDemotionSupport(enabled, Version.CURRENT);
    }

    /**
     * @brief [Functional Utility for withSyntheticSourceDemotionSupport]: Describe purpose here.
     * @param enabled: [Description]
     * @param version: [Description]
     * @return [ReturnType]: [Description]
     */
    private LogsdbIndexModeSettingsProvider withSyntheticSourceDemotionSupport(boolean enabled, Version version) {
        newMapperServiceCounter.set(0);
        var provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", enabled).build()
        );
        provider.init(im -> {
            newMapperServiceCounter.incrementAndGet();
            return MapperTestUtils.newMapperService(xContentRegistry(), createTempDir(), im.getSettings(), im.getIndex().getName());
        }, IndexVersion::current, () -> version, true, true);
    /**
     * @brief [Functional description for field provider]: Describe purpose here.
     */
        return provider;
    }

    /**
     * @brief [Functional Utility for withoutMapperService]: Describe purpose here.
     * @param enabled: [Description]
     * @return [ReturnType]: [Description]
     */
    private LogsdbIndexModeSettingsProvider withoutMapperService(boolean enabled) {
        var provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", enabled).build()
        );
        provider.init(im -> null, IndexVersion::current, () -> Version.CURRENT, true, true);
    /**
     * @brief [Functional description for field provider]: Describe purpose here.
     */
        return provider;
    }

    /**
     * @brief [Functional Utility for generateLogsdbSettings]: Describe purpose here.
     * @param settings: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    private Settings generateLogsdbSettings(Settings settings) throws IOException {
        return generateLogsdbSettings(settings, null, Version.CURRENT);
    }

    /**
     * @brief [Functional Utility for generateLogsdbSettings]: Describe purpose here.
     * @param settings: [Description]
     * @param mapping: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    private Settings generateLogsdbSettings(Settings settings, String mapping) throws IOException {
        return generateLogsdbSettings(settings, mapping, Version.CURRENT);
    }

    /**
     * @brief [Functional Utility for generateLogsdbSettings]: Describe purpose here.
     * @param settings: [Description]
     * @param mapping: [Description]
     * @param version: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    private Settings generateLogsdbSettings(Settings settings, String mapping, Version version) throws IOException {
        var provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", true).build()
        );
        provider.init(im -> {
            newMapperServiceCounter.incrementAndGet();
            return MapperTestUtils.newMapperService(xContentRegistry(), createTempDir(), im.getSettings(), im.getIndex().getName());
        }, IndexVersion::current, () -> version, true, true);
        var result = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(DATA_STREAM_NAME, 0),
            DATA_STREAM_NAME,
            IndexMode.LOGSDB,
            emptyProject(),
            Instant.now(),
            settings,
            mapping == null ? List.of() : List.of(new CompressedXContent(mapping))
        );
        return builder().put(result).build();
    }

    /**
     * @brief [Functional Utility for testDisabled]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testDisabled() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", false).build()
        );

        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            emptyProject(),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertTrue(additionalIndexSettings.isEmpty());
    }

    /**
     * @brief [Functional Utility for testOnIndexCreation]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testOnIndexCreation() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", true).build()
        );

        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            "logs-apache-production",
            null,
            null,
            emptyProject(),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertTrue(additionalIndexSettings.isEmpty());
    }

    /**
     * @brief [Functional Utility for testOnExplicitStandardIndex]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testOnExplicitStandardIndex() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", true).build()
        );

        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            emptyProject(),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.STANDARD.getName()).build(),
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertTrue(additionalIndexSettings.isEmpty());
    }

    /**
     * @brief [Functional Utility for testOnExplicitTimeSeriesIndex]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testOnExplicitTimeSeriesIndex() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", true).build()
        );

        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            emptyProject(),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.TIME_SERIES.getName()).build(),
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertTrue(additionalIndexSettings.isEmpty());
    }

    /**
     * @brief [Functional Utility for testNonLogsDataStream]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testNonLogsDataStream() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", true).build()
        );

        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs",
            null,
            emptyProject(),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertTrue(additionalIndexSettings.isEmpty());
    }

    /**
     * @brief [Functional Utility for testWithoutLogsComponentTemplate]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testWithoutLogsComponentTemplate() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = withoutMapperService(true);
        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            buildMetadata(List.of("*"), List.of()),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertIndexMode(additionalIndexSettings, IndexMode.LOGSDB.getName());
    }

    /**
     * @brief [Functional Utility for testWithLogsComponentTemplate]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testWithLogsComponentTemplate() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = withoutMapperService(true);
        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            buildMetadata(List.of("*"), List.of("logs@settings")),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertIndexMode(additionalIndexSettings, IndexMode.LOGSDB.getName());
    }

    /**
     * @brief [Functional Utility for testWithMultipleComponentTemplates]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testWithMultipleComponentTemplates() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = withoutMapperService(true);
        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            buildMetadata(List.of("*"), List.of("logs@settings", "logs@custom")),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertIndexMode(additionalIndexSettings, IndexMode.LOGSDB.getName());
    }

    /**
     * @brief [Functional Utility for testWithCustomComponentTemplatesOnly]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testWithCustomComponentTemplatesOnly() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = withoutMapperService(true);
        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            buildMetadata(List.of("*"), List.of("logs@custom", "custom-component-template")),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertIndexMode(additionalIndexSettings, IndexMode.LOGSDB.getName());
    }

    /**
     * @brief [Functional Utility for testNonMatchingTemplateIndexPattern]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testNonMatchingTemplateIndexPattern() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = withoutMapperService(true);
        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            buildMetadata(List.of("standard-apache-production"), List.of("logs@settings")),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertIndexMode(additionalIndexSettings, IndexMode.LOGSDB.getName());
    }

    /**
     * @brief [Functional Utility for testCaseSensitivity]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testCaseSensitivity() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", true).build()
        );

        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "LOGS-apache-production",
            null,
            emptyProject(),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertTrue(additionalIndexSettings.isEmpty());
    }

    /**
     * @brief [Functional Utility for testMultipleHyphensInDataStreamName]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testMultipleHyphensInDataStreamName() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = withoutMapperService(true);

        final Settings additionalIndexSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production-eu",
            null,
            emptyProject(),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertIndexMode(additionalIndexSettings, IndexMode.LOGSDB.getName());
    }

    /**
     * @brief [Functional Utility for testBeforeAndAfterSettingUpdate]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testBeforeAndAfterSettingUpdate() throws IOException {
        final LogsdbIndexModeSettingsProvider provider = withoutMapperService(false);
        final Settings beforeSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            buildMetadata(List.of("*"), List.of("logs@settings")),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertTrue(beforeSettings.isEmpty());

        provider.updateClusterIndexModeLogsdbEnabled(true);

        final Settings afterSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            buildMetadata(List.of("*"), List.of("logs@settings")),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertIndexMode(afterSettings, IndexMode.LOGSDB.getName());

        provider.updateClusterIndexModeLogsdbEnabled(false);

        final Settings laterSettings = provider.getAdditionalIndexSettings(
            null,
            "logs-apache-production",
            null,
            buildMetadata(List.of("*"), List.of("logs@settings")),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Settings.EMPTY,
            List.of(new CompressedXContent(DEFAULT_MAPPING))
        );

        assertTrue(laterSettings.isEmpty());
    }

    /**
     * @brief [Functional Utility for buildMetadata]: Describe purpose here.
     * @param indexPatterns: [Description]
     * @param componentTemplates: [Description]
     * @return [ReturnType]: [Description]
     */
    private static ProjectMetadata buildMetadata(final List<String> indexPatterns, final List<String> componentTemplates)
        throws IOException {
        final Template template = new Template(Settings.EMPTY, new CompressedXContent(DEFAULT_MAPPING), null);
        final ComposableIndexTemplate composableTemplate = ComposableIndexTemplate.builder()
            .indexPatterns(indexPatterns)
            .template(template)
            .componentTemplates(componentTemplates)
            .priority(1_000L)
            .version(1L)
            .build();
        return ProjectMetadata.builder(Metadata.DEFAULT_PROJECT_ID)
            .putCustom(ComposableIndexTemplateMetadata.TYPE, new ComposableIndexTemplateMetadata(Map.of("composable", composableTemplate)))
            .build();
    }

    /**
     * @brief [Functional Utility for assertIndexMode]: Describe purpose here.
     * @param settings: [Description]
     * @param expectedIndexMode: [Description]
     * @return [ReturnType]: [Description]
     */
    private void assertIndexMode(final Settings settings, final String expectedIndexMode) {
        assertEquals(expectedIndexMode, settings.get(IndexSettings.MODE.getKey()));
    }

    /**
     * @brief [Functional Utility for testNewIndexHasSyntheticSourceUsage]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testNewIndexHasSyntheticSourceUsage() throws IOException {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = DATA_STREAM_NAME;
        String indexName = DataStream.getDefaultBackingIndexName(dataStreamName, 0);
    /**
     * @brief [Functional description for field settings]: Describe purpose here.
     */
        Settings settings = Settings.EMPTY;
        LogsdbIndexModeSettingsProvider provider = withSyntheticSourceDemotionSupport(false);
        {
            String mapping = """
                {
                    "_doc": {
                        "_source": {
                            "mode": "synthetic"
                        },
                        "properties": {
                            "my_field": {
                                "type": "keyword"
                            }
                        }
                    }
                }
                """;
            boolean result = provider.getMappingHints(indexName, null, settings, List.of(new CompressedXContent(mapping)))
                .hasSyntheticSourceUsage();
            assertFalse("_source.mode is a noop", result);
            assertThat(newMapperServiceCounter.get(), equalTo(1));
            assertWarnings(SourceFieldMapper.DEPRECATION_WARNING);
        }
        {
    /**
     * @brief [Functional description for field mapping]: Describe purpose here.
     */
            String mapping;
            boolean withSourceMode = randomBoolean();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (withSourceMode) {
                mapping = """
                    {
                        "_doc": {
                            "_source": {
                                "mode": "stored"
                            },
                            "properties": {
                                "my_field": {
                                    "type": "keyword"
                                }
                            }
                        }
                    }
                    """;
        // Block Logic: [Describe purpose of this else/else if block]
            } else {
                mapping = """
                    {
                        "_doc": {
                            "properties": {
                                "my_field": {
                                    "type": "keyword"
                                }
                            }
                        }
                    }
                    """;
            }
            boolean result = provider.getMappingHints(indexName, null, settings, List.of(new CompressedXContent(mapping)))
                .hasSyntheticSourceUsage();
            assertFalse(result);
            assertThat(newMapperServiceCounter.get(), equalTo(2));
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (withSourceMode) {
                assertWarnings(SourceFieldMapper.DEPRECATION_WARNING);
            }
        }
    }

    /**
     * @brief [Functional Utility for testValidateIndexName]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testValidateIndexName() throws IOException {
    /**
     * @brief [Functional description for field indexName]: Describe purpose here.
     */
        String indexName = MetadataIndexTemplateService.VALIDATE_INDEX_NAME;
        String mapping = """
            {
                "_doc": {
                    "_source": {
                        "mode": "synthetic"
                    },
                    "properties": {
                        "my_field": {
                            "type": "keyword"
                        }
                    }
                }
            }
            """;
    /**
     * @brief [Functional description for field settings]: Describe purpose here.
     */
        Settings settings = Settings.EMPTY;
        LogsdbIndexModeSettingsProvider provider = withSyntheticSourceDemotionSupport(false);
        boolean result = provider.getMappingHints(indexName, null, settings, List.of(new CompressedXContent(mapping)))
            .hasSyntheticSourceUsage();
        assertFalse(result);
    }

    /**
     * @brief [Functional Utility for testNewIndexHasSyntheticSourceUsageLogsdbIndex]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testNewIndexHasSyntheticSourceUsageLogsdbIndex() throws IOException {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = DATA_STREAM_NAME;
        String indexName = DataStream.getDefaultBackingIndexName(dataStreamName, 0);
        String mapping = """
            {
                "_doc": {
                    "properties": {
                        "my_field": {
                            "type": "keyword"
                        }
                    }
                }
            }
            """;
        LogsdbIndexModeSettingsProvider provider = withSyntheticSourceDemotionSupport(false);
        {
            Settings settings = Settings.builder().put("index.mode", "logsdb").build();
            boolean result = provider.getMappingHints(indexName, null, settings, List.of(new CompressedXContent(mapping)))
                .hasSyntheticSourceUsage();
            assertTrue(result);
            assertThat(newMapperServiceCounter.get(), equalTo(1));
        }
        {
            Settings settings = Settings.builder().put("index.mode", "logsdb").build();
            boolean result = provider.getMappingHints(indexName, null, settings, List.of()).hasSyntheticSourceUsage();
            assertTrue(result);
            assertThat(newMapperServiceCounter.get(), equalTo(2));
        }
        {
            boolean result = provider.getMappingHints(indexName, null, Settings.EMPTY, List.of()).hasSyntheticSourceUsage();
            assertFalse(result);
            assertThat(newMapperServiceCounter.get(), equalTo(3));
        }
        {
            boolean result = provider.getMappingHints(indexName, null, Settings.EMPTY, List.of(new CompressedXContent(mapping)))
                .hasSyntheticSourceUsage();
            assertFalse(result);
            assertThat(newMapperServiceCounter.get(), equalTo(4));
        }
    }

    /**
     * @brief [Functional Utility for testNewIndexHasSyntheticSourceUsageTimeSeries]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testNewIndexHasSyntheticSourceUsageTimeSeries() throws IOException {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = DATA_STREAM_NAME;
        String indexName = DataStream.getDefaultBackingIndexName(dataStreamName, 0);
        String mapping = """
            {
                "_doc": {
                    "properties": {
                        "my_field": {
                            "type": "keyword",
                            "time_series_dimension": true
                        }
                    }
                }
            }
            """;
        LogsdbIndexModeSettingsProvider provider = withSyntheticSourceDemotionSupport(false);
        {
            Settings settings = Settings.builder().put("index.mode", "time_series").put("index.routing_path", "my_field").build();
            boolean result = provider.getMappingHints(indexName, null, settings, List.of(new CompressedXContent(mapping)))
                .hasSyntheticSourceUsage();
            assertTrue(result);
        }
        {
            Settings settings = Settings.builder().put("index.mode", "time_series").put("index.routing_path", "my_field").build();
            boolean result = provider.getMappingHints(indexName, null, settings, List.of()).hasSyntheticSourceUsage();
            assertTrue(result);
        }
        {
            boolean result = provider.getMappingHints(indexName, null, Settings.EMPTY, List.of()).hasSyntheticSourceUsage();
            assertFalse(result);
        }
        {
            boolean result = provider.getMappingHints(indexName, null, Settings.EMPTY, List.of(new CompressedXContent(mapping)))
                .hasSyntheticSourceUsage();
            assertFalse(result);
        }
    }

    /**
     * @brief [Functional Utility for testNewIndexHasSyntheticSourceUsageInvalidSettings]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testNewIndexHasSyntheticSourceUsageInvalidSettings() throws IOException {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = DATA_STREAM_NAME;
        String indexName = DataStream.getDefaultBackingIndexName(dataStreamName, 0);
        Settings settings = Settings.builder().put("index.soft_deletes.enabled", false).build();
        LogsdbIndexModeSettingsProvider provider = withSyntheticSourceDemotionSupport(false);
        {
            String mapping = """
                {
                    "_doc": {
                        "_source": {
                            "mode": "synthetic"
                        },
                        "properties": {
                            "my_field": {
                                "type": "keyword"
                            }
                        }
                    }
                }
                """;
            boolean result = provider.getMappingHints(indexName, null, settings, List.of(new CompressedXContent(mapping)))
                .hasSyntheticSourceUsage();
            assertFalse(result);
            assertThat(newMapperServiceCounter.get(), equalTo(1));
        }
        {
            String mapping = """
                {
                    "_doc": {
                        "properties": {
                            "my_field": {
                                "type": "keyword"
                            }
                        }
                    }
                }
                """;
            boolean result = provider.getMappingHints(indexName, null, settings, List.of(new CompressedXContent(mapping)))
                .hasSyntheticSourceUsage();
            assertFalse(result);
            assertThat(newMapperServiceCounter.get(), equalTo(2));
        }
    }

    /**
     * @brief [Functional Utility for testGetAdditionalIndexSettingsDowngradeFromSyntheticSource]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testGetAdditionalIndexSettingsDowngradeFromSyntheticSource() {
    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = DATA_STREAM_NAME;
        final var projectId = randomProjectIdOrDefault();
        ProjectMetadata project = DataStreamTestHelper.getClusterStateWithDataStreams(
            projectId,
            List.of(Tuple.tuple(dataStreamName, 1)),
            List.of(),
            Instant.now().toEpochMilli(),
            builder().build(),
            1
        ).metadata().getProject(projectId);
        LogsdbIndexModeSettingsProvider provider = withSyntheticSourceDemotionSupport(false);
        Settings settings = builder().put(IndexSettings.INDEX_MAPPER_SOURCE_MODE_SETTING.getKey(), SourceFieldMapper.Mode.SYNTHETIC)
            .build();

        Settings result = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(dataStreamName, 2),
            dataStreamName,
            null,
            project,
            Instant.ofEpochMilli(1L),
            settings,
            List.of()
        );
        assertThat(result.size(), equalTo(0));
        assertThat(newMapperServiceCounter.get(), equalTo(1));

        logsdbLicenseService.setSyntheticSourceFallback(true);
        result = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(dataStreamName, 2),
            dataStreamName,
            null,
            project,
            Instant.ofEpochMilli(1L),
            settings,
            List.of()
        );
        assertThat(result.size(), equalTo(1));
        assertEquals(SourceFieldMapper.Mode.STORED, IndexSettings.INDEX_MAPPER_SOURCE_MODE_SETTING.get(result));
        assertThat(newMapperServiceCounter.get(), equalTo(2));

        result = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(dataStreamName, 2),
            dataStreamName,
            IndexMode.TIME_SERIES,
            project,
            Instant.ofEpochMilli(1L),
            settings,
            List.of()
        );
        assertThat(result.size(), equalTo(1));
        assertEquals(SourceFieldMapper.Mode.STORED, IndexSettings.INDEX_MAPPER_SOURCE_MODE_SETTING.get(result));
        assertThat(newMapperServiceCounter.get(), equalTo(3));

        result = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(dataStreamName, 2),
            dataStreamName,
            IndexMode.LOGSDB,
            project,
            Instant.ofEpochMilli(1L),
            settings,
            List.of()
        );
        assertThat(result.size(), equalTo(3));
        assertEquals(SourceFieldMapper.Mode.STORED, IndexSettings.INDEX_MAPPER_SOURCE_MODE_SETTING.get(result));
        assertTrue(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertTrue(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertThat(newMapperServiceCounter.get(), equalTo(4));
    }

    /**
     * @brief [Functional Utility for testGetAdditionalIndexSettingsDowngradeFromSyntheticSourceOldNode]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testGetAdditionalIndexSettingsDowngradeFromSyntheticSourceOldNode() {
        logsdbLicenseService.setSyntheticSourceFallback(true);
        LogsdbIndexModeSettingsProvider provider = withSyntheticSourceDemotionSupport(true, Version.V_8_16_0);
        final var projectId = randomProjectIdOrDefault();
        ProjectMetadata project = DataStreamTestHelper.getClusterStateWithDataStreams(
            projectId,
            List.of(Tuple.tuple(DATA_STREAM_NAME, 1)),
            List.of(),
            Instant.now().toEpochMilli(),
            builder().build(),
            1
        ).metadata().getProject(projectId);
        Settings settings = builder().put(IndexSettings.INDEX_MAPPER_SOURCE_MODE_SETTING.getKey(), SourceFieldMapper.Mode.SYNTHETIC)
            .build();
        var result = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(DATA_STREAM_NAME, 2),
            DATA_STREAM_NAME,
            null,
            project,
            Instant.ofEpochMilli(1L),
            settings,
            List.of()
        );
        assertTrue(result.isEmpty());
    }

    /**
     * @brief [Functional Utility for testGetAdditionalIndexSettingsDowngradeFromSyntheticSourceFileMatch]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testGetAdditionalIndexSettingsDowngradeFromSyntheticSourceFileMatch() throws IOException {
        logsdbLicenseService.setSyntheticSourceFallback(true);
        LogsdbIndexModeSettingsProvider provider = withSyntheticSourceDemotionSupport(true);
    /**
     * @brief [Functional description for field settings]: Describe purpose here.
     */
        final Settings settings = Settings.EMPTY;

    /**
     * @brief [Functional description for field dataStreamName]: Describe purpose here.
     */
        String dataStreamName = DATA_STREAM_NAME;
        final var projectId = randomProjectIdOrDefault();
        ProjectMetadata project = DataStreamTestHelper.getClusterStateWithDataStreams(
            projectId,
            List.of(Tuple.tuple(dataStreamName, 1)),
            List.of(),
            Instant.now().toEpochMilli(),
            builder().build(),
            1
        ).metadata().getProject(projectId);
        Settings result = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(dataStreamName, 2),
            dataStreamName,
            null,
            project,
            Instant.ofEpochMilli(1L),
            settings,
            List.of()
        );
        assertThat(result.size(), equalTo(0));

        dataStreamName = "logs-app1-0";
        project = DataStreamTestHelper.getClusterStateWithDataStreams(
            projectId,
            List.of(Tuple.tuple(dataStreamName, 1)),
            List.of(),
            Instant.now().toEpochMilli(),
            builder().build(),
            1
        ).metadata().getProject(projectId);

        result = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(dataStreamName, 2),
            dataStreamName,
            null,
            project,
            Instant.ofEpochMilli(1L),
            settings,
            List.of()
        );
        assertThat(result.size(), equalTo(4));
        assertEquals(SourceFieldMapper.Mode.STORED, IndexSettings.INDEX_MAPPER_SOURCE_MODE_SETTING.get(result));
        assertEquals(IndexMode.LOGSDB, IndexSettings.MODE.get(result));
        assertTrue(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertTrue(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));

        result = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(dataStreamName, 2),
            dataStreamName,
            null,
            project,
            Instant.ofEpochMilli(1L),
            builder().put(IndexSettings.MODE.getKey(), IndexMode.STANDARD.toString()).build(),
            List.of()
        );
        assertThat(result.size(), equalTo(0));
    }

    /**
     * @brief [Functional Utility for testRoutingPathOnSortFields]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testRoutingPathOnSortFields() throws Exception {
        var settings = Settings.builder()
            .put(IndexSortConfig.INDEX_SORT_FIELD_SETTING.getKey(), "host,message")
            .put(IndexSettings.LOGSDB_ROUTE_ON_SORT_FIELDS.getKey(), true)
            .build();
        Settings result = generateLogsdbSettings(settings);
        assertThat(IndexMetadata.INDEX_ROUTING_PATH.get(result), contains("host", "message"));
    }

    /**
     * @brief [Functional Utility for testRoutingPathOnSortFieldsDisabledInOldNode]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testRoutingPathOnSortFieldsDisabledInOldNode() throws Exception {
        var settings = Settings.builder()
            .put(IndexSortConfig.INDEX_SORT_FIELD_SETTING.getKey(), "host,message")
            .put(IndexSettings.LOGSDB_ROUTE_ON_SORT_FIELDS.getKey(), true)
            .build();
        Settings result = generateLogsdbSettings(settings, null, Version.V_8_17_0);
        assertTrue(result.isEmpty());
    }

    /**
     * @brief [Functional Utility for testRoutingPathOnSortFieldsFilterTimestamp]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testRoutingPathOnSortFieldsFilterTimestamp() throws Exception {
        var settings = Settings.builder()
            .put(IndexSortConfig.INDEX_SORT_FIELD_SETTING.getKey(), "host,message,@timestamp")
            .put(IndexSettings.LOGSDB_ROUTE_ON_SORT_FIELDS.getKey(), true)
            .build();
        Settings result = generateLogsdbSettings(settings);
        assertThat(IndexMetadata.INDEX_ROUTING_PATH.get(result), contains("host", "message"));
    }

    /**
     * @brief [Functional Utility for testRoutingPathOnSortSingleField]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testRoutingPathOnSortSingleField() throws Exception {
        var settings = Settings.builder()
            .put(IndexSortConfig.INDEX_SORT_FIELD_SETTING.getKey(), "host")
            .put(IndexSettings.LOGSDB_ROUTE_ON_SORT_FIELDS.getKey(), true)
            .build();
        Exception e = expectThrows(IllegalStateException.class, () -> generateLogsdbSettings(settings));
        assertThat(
            e.getMessage(),
            equalTo(
                "data stream ["
                    + DATA_STREAM_NAME
                    + "] in logsdb mode and with [index.logsdb.route_on_sort_fields] index setting has only 1 sort fields "
                    + "(excluding timestamp), needs at least 2"
            )
        );
    }

    /**
     * @brief [Functional Utility for testExplicitRoutingPathMatchesSortFields]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testExplicitRoutingPathMatchesSortFields() throws Exception {
        var settings = Settings.builder()
            .put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB)
            .put(IndexSortConfig.INDEX_SORT_FIELD_SETTING.getKey(), "host,message,@timestamp")
            .put(IndexMetadata.INDEX_ROUTING_PATH.getKey(), "host,message")
            .put(IndexSettings.LOGSDB_ROUTE_ON_SORT_FIELDS.getKey(), true)
            .build();
        Settings result = generateLogsdbSettings(settings);
        assertTrue(result.isEmpty());
    }

    /**
     * @brief [Functional Utility for testExplicitRoutingPathDoesNotMatchSortFields]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testExplicitRoutingPathDoesNotMatchSortFields() {
        var settings = Settings.builder()
            .put(IndexSortConfig.INDEX_SORT_FIELD_SETTING.getKey(), "host,message,@timestamp")
            .put(IndexMetadata.INDEX_ROUTING_PATH.getKey(), "host,message,foo")
            .put(IndexSettings.LOGSDB_ROUTE_ON_SORT_FIELDS.getKey(), true)
            .build();
        Exception e = expectThrows(IllegalStateException.class, () -> generateLogsdbSettings(settings));
        assertThat(
            e.getMessage(),
            equalTo(
                "data stream ["
                    + DATA_STREAM_NAME
                    + "] in logsdb mode and with [index."
                    + "logsdb.route_on_sort_fields] index setting has mismatching sort "
                    + "and routing fields, [index.routing_path:[host, message, foo]], [index.sort.fields:[host, message]]"
            )
        );
    }

    /**
     * @brief [Functional Utility for testExplicitRoutingPathNotAllowedByLicense]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testExplicitRoutingPathNotAllowedByLicense() throws Exception {
        MockLicenseState licenseState = MockLicenseState.createMock();
        when(licenseState.copyCurrentLicenseState()).thenReturn(licenseState);
        when(licenseState.isAllowed(same(LogsdbLicenseService.LOGSDB_ROUTING_ON_SORT_FIELDS_FEATURE))).thenReturn(false);
        logsdbLicenseService = new LogsdbLicenseService(Settings.EMPTY);
        logsdbLicenseService.setLicenseState(licenseState);

        var settings = Settings.builder()
            .put(IndexSortConfig.INDEX_SORT_FIELD_SETTING.getKey(), "host,message")
            .put(IndexSettings.LOGSDB_ROUTE_ON_SORT_FIELDS.getKey(), true)
            .build();
        Settings result = generateLogsdbSettings(settings);
        assertFalse(IndexSettings.LOGSDB_ROUTE_ON_SORT_FIELDS.get(result));
        assertThat(IndexMetadata.INDEX_ROUTING_PATH.get(result), empty());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNamePropagateValue]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNamePropagateValue() throws Exception {
        var settings = Settings.builder()
            .put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB)
            .put(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.getKey(), true)
            .put(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.getKey(), true)
            .build();
        Settings result = generateLogsdbSettings(settings);
        assertTrue(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertTrue(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(0, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNameWithCustomSortConfig]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNameWithCustomSortConfig() throws Exception {
        var settings = Settings.builder()
            .put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB)
            .put(IndexSortConfig.INDEX_SORT_FIELD_SETTING.getKey(), "foo,bar")
            .build();
        Settings result = generateLogsdbSettings(settings);
        assertFalse(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertFalse(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(0, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNoHost]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNoHost() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertTrue(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertTrue(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNoHostOldNode]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNoHostOldNode() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings, Version.V_8_17_0);
        assertTrue(result.isEmpty());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNameKeyword]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNameKeyword() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        },
                        "host.name": {
                            "type": "keyword"
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertTrue(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertFalse(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNameKeywordNoDocvalues]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNameKeywordNoDocvalues() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        },
                        "host.name": {
                            "type": "keyword",
                            "doc_values": false
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertFalse(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertFalse(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNameInteger]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNameInteger() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        },
                        "host.name": {
                            "type": "integer"
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertTrue(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertFalse(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNameIntegerNoDocvalues]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNameIntegerNoDocvalues() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        },
                        "host.name": {
                            "type": "integer",
                            "doc_values": false
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertFalse(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertFalse(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNameBoolean]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNameBoolean() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        },
                        "host.name": {
                            "type": "boolean"
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertFalse(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertFalse(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostObject]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostObject() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        },
                        "host": {
                            "type": "object"
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertTrue(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertTrue(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostField]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostField() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        },
                        "host": {
                            "type": "keyword"
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertFalse(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertFalse(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostFieldSubobjectsFalse]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostFieldSubobjectsFalse() throws Exception {
        var settings = Settings.builder().put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB).build();
        var mappings = """
            {
                "_doc": {
                    "subobjects": false,
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        },
                        "host": {
                            "type": "keyword"
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertTrue(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertTrue(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortAndHostNameObject]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortAndHostNameObject() throws Exception {
        var settings = Settings.builder()
            .put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB)
            .put(IndexSettings.INDEX_FAST_REFRESH_SETTING.getKey(), true)
            .build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        },
                        "host.name.sub": {
                            "type": "keyword"
                        }
                    }
                }
            }
            """;
        Settings result = generateLogsdbSettings(settings, mappings);
        assertFalse(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
        assertFalse(IndexSettings.LOGSDB_ADD_HOST_NAME_FIELD.get(result));
        assertEquals(1, newMapperServiceCounter.get());
    }

    /**
     * @brief [Functional Utility for testSortFastRefresh]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws Exception: [Description]
     */
    public void testSortFastRefresh() throws Exception {
        var settings = Settings.builder()
            .put(IndexSettings.MODE.getKey(), IndexMode.LOGSDB)
            .put(IndexSettings.INDEX_FAST_REFRESH_SETTING.getKey(), true)
            .build();
        var mappings = """
            {
                "_doc": {
                    "properties": {
                        "@timestamp": {
                            "type": "date"
                        }
                    }
                }
            }
            """;

    /**
     * @brief [Functional description for field systemIndex]: Describe purpose here.
     */
        String systemIndex = ".security-profile";
        var provider = new LogsdbIndexModeSettingsProvider(
            logsdbLicenseService,
            Settings.builder().put("cluster.logsdb.enabled", true).build()
        );
        provider.init(
            im -> MapperTestUtils.newMapperService(xContentRegistry(), createTempDir(), im.getSettings(), im.getIndex().getName()),
            IndexVersion::current,
            () -> Version.CURRENT,
            true,
            true
        );
        var additionalIndexSettings = provider.getAdditionalIndexSettings(
            DataStream.getDefaultBackingIndexName(systemIndex, 0),
            systemIndex,
            IndexMode.LOGSDB,
            emptyProject(),
            Instant.now(),
            settings,
            List.of(new CompressedXContent(mappings))
        );

        Settings result = builder().put(additionalIndexSettings).build();
        assertTrue(IndexSettings.LOGSDB_SORT_ON_HOST_NAME.get(result));
    }
}
