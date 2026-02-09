/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.action;

import org.elasticsearch.action.admin.indices.create.CreateIndexRequestBuilder;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.mapper.extras.MapperExtrasPlugin;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.test.ESIntegTestCase;
import org.elasticsearch.test.ESIntegTestCase.ClusterScope;
import org.elasticsearch.xpack.core.esql.action.ColumnInfo;
import org.elasticsearch.xpack.esql.VerificationException;
import org.elasticsearch.xpack.esql.core.type.DataType;
import org.elasticsearch.xpack.esql.plan.logical.join.Join;
import org.elasticsearch.xpack.esql.plugin.EsqlPlugin;
import org.elasticsearch.xpack.spatial.SpatialPlugin;
import org.elasticsearch.xpack.unsignedlong.UnsignedLongMapperPlugin;
import org.elasticsearch.xpack.versionfield.VersionFieldPlugin;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import static org.elasticsearch.test.ESIntegTestCase.Scope.SUITE;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertAcked;
import static org.elasticsearch.xpack.esql.core.type.DataType.AGGREGATE_METRIC_DOUBLE;
import static org.elasticsearch.xpack.esql.core.type.DataType.BOOLEAN;
import static org.elasticsearch.xpack.esql.core.type.DataType.BYTE;
import static org.elasticsearch.xpack.esql.core.type.DataType.DATETIME;
import static org.elasticsearch.xpack.esql.core.type.DataType.DATE_NANOS;
import static org.elasticsearch.xpack.esql.core.type.DataType.DOC_DATA_TYPE;
import static org.elasticsearch.xpack.esql.core.type.DataType.DOUBLE;
import static org.elasticsearch.xpack.esql.core.type.DataType.FLOAT;
import static org.elasticsearch.xpack.esql.core.type.DataType.HALF_FLOAT;
import static org.elasticsearch.xpack.esql.core.type.DataType.INTEGER;
import static org.elasticsearch.xpack.esql.core.type.DataType.IP;
import static org.elasticsearch.xpack.esql.core.type.DataType.KEYWORD;
import static org.elasticsearch.xpack.esql.core.type.DataType.LONG;
import static org.elasticsearch.xpack.esql.core.type.DataType.NULL;
import static org.elasticsearch.xpack.esql.core.type.DataType.SCALED_FLOAT;
import static org.elasticsearch.xpack.esql.core.type.DataType.SHORT;
import static org.elasticsearch.xpack.esql.core.type.DataType.TEXT;
import static org.elasticsearch.xpack.esql.core.type.DataType.TSID_DATA_TYPE;
import static org.elasticsearch.xpack.esql.core.type.DataType.UNDER_CONSTRUCTION;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

/**
 * This test suite tests the lookup join functionality in ESQL with various data types.
 * For each pair of types being tested, it builds a main index called "index" containing a single document with as many fields as
 * types being tested on the left of the pair, and then creates that many other lookup indexes, each with a single document containing
 * exactly two fields: the field to join on, and a field to return.
 * The assertion is that for valid combinations, the return result should exist, and for invalid combinations an exception should be thrown.
 * If no exception is thrown, and no result is returned, our validation rules are not aligned with the internal behaviour (i.e. a bug).
 * Let's assume we want to test a lookup using a byte field in the main index and integer in the lookup index, then we'll create 2 indices,
 * named {@code main_index} and {@code lookup_byte_integer} resp.
 * The main index contains a field called {@code main_byte} and the lookup index has {@code lookup_integer}. To test the pair, we run
 * {@code FROM main_index | RENAME main_byte AS lookup_integer | LOOKUP JOIN lookup_index ON lookup_integer | KEEP other}
 * and assert that the result exists and is equal to "value".
 */
@ClusterScope(scope = SUITE, numClientNodes = 1, numDataNodes = 1)
/**
 * @brief Functional description of the LookupJoinTypesIT class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class LookupJoinTypesIT extends ESIntegTestCase {
    private static final String MAIN_INDEX_PREFIX = "main_";
    private static final String MAIN_INDEX = MAIN_INDEX_PREFIX + "index";
    private static final String LOOKUP_INDEX_PREFIX = "lookup_";

    protected Collection<Class<? extends Plugin>> nodePlugins() {
        return List.of(
            EsqlPlugin.class,
            MapperExtrasPlugin.class,
            VersionFieldPlugin.class,
            UnsignedLongMapperPlugin.class,
            SpatialPlugin.class
        );
    }

    private static final Map<String, TestConfigs> testConfigurations = new HashMap<>();
    static {
        // Initialize the test configurations for string tests
        {
            TestConfigs configs = testConfigurations.computeIfAbsent("strings", TestConfigs::new);
            configs.addPasses(KEYWORD, KEYWORD);
            configs.addPasses(TEXT, KEYWORD);
            configs.addFailsUnsupported(KEYWORD, TEXT);
        }

        // Test integer types
        var integerTypes = List.of(BYTE, SHORT, INTEGER, LONG);
        {
            TestConfigs configs = testConfigurations.computeIfAbsent("integers", TestConfigs::new);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            for (DataType mainType : integerTypes) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                for (DataType lookupType : integerTypes) {
                    configs.addPasses(mainType, lookupType);
                }
            }
        }

        // Test float and double
        var floatTypes = List.of(HALF_FLOAT, FLOAT, DOUBLE, SCALED_FLOAT);
        {
            TestConfigs configs = testConfigurations.computeIfAbsent("floats", TestConfigs::new);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            for (DataType mainType : floatTypes) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                for (DataType lookupType : floatTypes) {
                    configs.addPasses(mainType, lookupType);
                }
            }
        }

        // Tests for mixed-numerical types
        {
            TestConfigs configs = testConfigurations.computeIfAbsent("mixed-numerical", TestConfigs::new);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            for (DataType mainType : integerTypes) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                for (DataType lookupType : floatTypes) {
                    configs.addPasses(mainType, lookupType);
                    configs.addPasses(lookupType, mainType);
                }
            }
        }

        // Tests for mixed-date/time types
        var dateTypes = List.of(DATETIME, DATE_NANOS);
        {
            TestConfigs configs = testConfigurations.computeIfAbsent("mixed-temporal", TestConfigs::new);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            for (DataType mainType : dateTypes) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                for (DataType lookupType : dateTypes) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (mainType != lookupType) {
                        configs.addFails(mainType, lookupType);
                    }
                }
            }
        }

        // Tests for all unsupported types
    /**
     * @brief [Functional description for field unsupported]: Describe purpose here.
     */
        DataType[] unsupported = Join.UNSUPPORTED_TYPES;
        {
            Collection<TestConfigs> existing = testConfigurations.values();
            TestConfigs configs = testConfigurations.computeIfAbsent("unsupported", TestConfigs::new);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            for (DataType type : unsupported) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (type == NULL
                    || type == DOC_DATA_TYPE
                    || type == TSID_DATA_TYPE
                    || type == AGGREGATE_METRIC_DOUBLE
                    || type.esType() == null
                    || type.isCounter()
                    || DataType.isRepresentable(type) == false) {
                    // Skip unmappable types, or types not supported in ES|QL in general
                    continue;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (existingIndex(existing, type, type)) {
                    // Skip existing configurations
                    continue;
                }
                configs.addFailsUnsupported(type, type);
            }
        }

        // Tests for all types where left and right are the same type
        DataType[] supported = {
            BOOLEAN,
            LONG,
            INTEGER,
            DOUBLE,
            SHORT,
            BYTE,
            FLOAT,
            HALF_FLOAT,
            DATETIME,
            DATE_NANOS,
            IP,
            KEYWORD,
            SCALED_FLOAT };
        {
            Collection<TestConfigs> existing = testConfigurations.values();
            TestConfigs configs = testConfigurations.computeIfAbsent("same", TestConfigs::new);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            for (DataType type : supported) {
                assertThat("Claiming supported for unsupported type: " + type, List.of(unsupported).contains(type), is(false));
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (existingIndex(existing, type, type) == false) {
                    // Only add the configuration if it doesn't already exist
                    configs.addPasses(type, type);
                }
            }
        }

        // Assert that unsupported types are not in the supported list
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (DataType type : unsupported) {
            assertThat("Claiming supported for unsupported type: " + type, List.of(supported).contains(type), is(false));
        }

        // Assert that unsupported+supported covers all types:
        List<DataType> missing = new ArrayList<>();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (DataType type : DataType.values()) {
            boolean isUnsupported = List.of(unsupported).contains(type);
            boolean isSupported = List.of(supported).contains(type);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (isUnsupported == false && isSupported == false) {
                missing.add(type);
            }
        }
        assertThat(missing + " are not in the supported or unsupported list", missing.size(), is(0));

        // Tests for all other type combinations
        {
            Collection<TestConfigs> existing = testConfigurations.values();
            TestConfigs configs = testConfigurations.computeIfAbsent("others", TestConfigs::new);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            for (DataType mainType : supported) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                for (DataType lookupType : supported) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (existingIndex(existing, mainType, lookupType) == false) {
                        // Only add the configuration if it doesn't already exist
                        configs.addFails(mainType, lookupType);
                    }
                }
            }
        }

        // Make sure we have never added two configurations with the same index name
        Set<String> knownTypes = new HashSet<>();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (TestConfigs configs : testConfigurations.values()) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            for (TestConfig config : configs.configs.values()) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (knownTypes.contains(config.lookupIndexName())) {
                    throw new IllegalArgumentException("Duplicate index name: " + config.lookupIndexName());
                }
                knownTypes.add(config.lookupIndexName());
            }
        }
    }

    /**
     * @brief [Functional Utility for existingIndex]: Describe purpose here.
     * @param existing: [Description]
     * @param mainType: [Description]
     * @param lookupType: [Description]
     * @return [ReturnType]: [Description]
     */
    private static boolean existingIndex(Collection<TestConfigs> existing, DataType mainType, DataType lookupType) {
        String indexName = LOOKUP_INDEX_PREFIX + mainType.esType() + "_" + lookupType.esType();
        return existing.stream().anyMatch(c -> c.exists(indexName));
    }

    /**
     * @brief [Functional Utility for testLookupJoinStrings]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testLookupJoinStrings() {
        testLookupJoinTypes("strings");
    }

    /**
     * @brief [Functional Utility for testLookupJoinIntegers]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testLookupJoinIntegers() {
        testLookupJoinTypes("integers");
    }

    /**
     * @brief [Functional Utility for testLookupJoinFloats]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testLookupJoinFloats() {
        testLookupJoinTypes("floats");
    }

    /**
     * @brief [Functional Utility for testLookupJoinMixedNumerical]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testLookupJoinMixedNumerical() {
        testLookupJoinTypes("mixed-numerical");
    }

    /**
     * @brief [Functional Utility for testLookupJoinMixedTemporal]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testLookupJoinMixedTemporal() {
        testLookupJoinTypes("mixed-temporal");
    }

    /**
     * @brief [Functional Utility for testLookupJoinSame]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testLookupJoinSame() {
        testLookupJoinTypes("same");
    }

    /**
     * @brief [Functional Utility for testLookupJoinUnsupported]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testLookupJoinUnsupported() {
        testLookupJoinTypes("unsupported");
    }

    /**
     * @brief [Functional Utility for testLookupJoinOthers]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void testLookupJoinOthers() {
        testLookupJoinTypes("others");
    }

    /**
     * @brief [Functional Utility for testLookupJoinTypes]: Describe purpose here.
     * @param group: [Description]
     * @return [ReturnType]: [Description]
     */
    private void testLookupJoinTypes(String group) {
        TestConfigs configs = testConfigurations.get(group);
        initIndexes(configs);
        initData(configs);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (TestConfig config : configs.values()) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if ((isValidDataType(config.mainType()) && isValidDataType(config.lookupType())) == false) {
                continue;
            }
            config.validateMainIndex();
            config.validateLookupIndex();

            config.doTest();
        }
    }

    /**
     * @brief [Functional Utility for initIndexes]: Describe purpose here.
     * @param configs: [Description]
     * @return [ReturnType]: [Description]
     */
    private void initIndexes(TestConfigs configs) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (TestMapping mapping : configs.indices()) {
            CreateIndexRequestBuilder builder = prepareCreate(mapping.indexName).setMapping(mapping.properties);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (mapping.settings != null) {
                builder = builder.setSettings(mapping.settings);
            }
            assertAcked(builder);
        }
    }

    /**
     * @brief [Functional Utility for initData]: Describe purpose here.
     * @param configs: [Description]
     * @return [ReturnType]: [Description]
     */
    private void initData(TestConfigs configs) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (TestDocument doc : configs.docs()) {
            index(doc.indexName, doc.id, doc.source);
            refresh(doc.indexName);
        }
    }

    /**
     * @brief [Functional Utility for lookupPropertyFor]: Describe purpose here.
     * @param config: [Description]
     * @return [ReturnType]: [Description]
     */
    private static String lookupPropertyFor(TestConfig config) {
        return String.format(Locale.ROOT, "\"%s\": %s", config.lookupFieldName(), sampleDataTextFor(config.lookupType()));
    }

    /**
     * @brief [Functional Utility for mainPropertyFor]: Describe purpose here.
     * @param config: [Description]
     * @return [ReturnType]: [Description]
     */
    private static String mainPropertyFor(TestConfig config) {
        return String.format(Locale.ROOT, "\"%s\": %s", config.mainFieldName(), sampleDataTextFor(config.mainType()));
    }

    /**
     * @brief [Functional Utility for sampleDataTextFor]: Describe purpose here.
     * @param type: [Description]
     * @return [ReturnType]: [Description]
     */
    private static String sampleDataTextFor(DataType type) {
        return sampleDataForValue(sampleDataFor(type));
    }

    /**
     * @brief [Functional Utility for sampleDataForValue]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
    private static String sampleDataForValue(Object value) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (value instanceof String) {
            return "\"" + value + "\"";
        // Block Logic: [Describe purpose of this else/else if block]
        } else if (value instanceof List<?> list) {
            return "[" + list.stream().map(LookupJoinTypesIT::sampleDataForValue).collect(Collectors.joining(", ")) + "]";
        }
        return String.valueOf(value);
    }

    private static final double SCALING_FACTOR = 10.0;

    /**
     * @brief [Functional Utility for sampleDataFor]: Describe purpose here.
     * @param type: [Description]
     * @return [ReturnType]: [Description]
     */
    private static Object sampleDataFor(DataType type) {
    /**
     * @brief [Functional Utility for switch]: Describe purpose here.
     * @param type: [Description]
     * @return [ReturnType]: [Description]
     */
        return switch (type) {
            case BOOLEAN -> true;
            case DATETIME, DATE_NANOS -> "2025-04-02T12:00:00.000Z";
            case IP -> "127.0.0.1";
            case KEYWORD, TEXT -> "key";
            case BYTE, SHORT, INTEGER -> 1;
            case LONG, UNSIGNED_LONG -> 1L;
            case HALF_FLOAT, FLOAT, DOUBLE, SCALED_FLOAT -> 1.0;
            case VERSION -> "1.2.19";
            case GEO_POINT, CARTESIAN_POINT -> "POINT (1.0 2.0)";
            case GEO_SHAPE, CARTESIAN_SHAPE -> "POLYGON ((0.0 0.0, 1.0 0.0, 1.0 1.0, 0.0 1.0, 0.0 0.0))";
            case DENSE_VECTOR -> List.of(0.2672612f, 0.5345224f, 0.8017837f);
            default -> throw new IllegalArgumentException("Unsupported type: " + type);
        };
    }

    private record TestMapping(String indexName, String properties, Settings settings) {};

    private record TestDocument(String indexName, String id, String source) {};

    private static class TestConfigs {
    /**
     * @brief [Functional description for field group]: Describe purpose here.
     */
        final String group;
        final Map<String, TestConfig> configs;

        TestConfigs(String group) {
            this.group = group;
            this.configs = new LinkedHashMap<>();
        }

    /**
     * @brief [Functional Utility for indices]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        protected List<TestMapping> indices() {
            List<TestMapping> results = new ArrayList<>();

    /**
     * @brief [Functional description for field propertyPrefix]: Describe purpose here.
     */
            String propertyPrefix = "{\n  \"properties\" : {\n";
    /**
     * @brief [Functional description for field propertySuffix]: Describe purpose here.
     */
            String propertySuffix = "  }\n}\n";
            // The main index will have many fields, one of each type to use in later type specific joins
            String mainFields = propertyPrefix + configs.values()
                .stream()
                .map(TestConfig::mainPropertySpec)
                .distinct()
                .collect(Collectors.joining(",\n    ")) + propertySuffix;

            results.add(new TestMapping(MAIN_INDEX, mainFields, null));

            Settings.Builder settings = Settings.builder()
                .put("index.number_of_shards", 1)
                .put("index.number_of_replicas", 0)
                .put("index.mode", "lookup");
            configs.values()
                .forEach(
                    // Each lookup index will get a document with a field to join on, and a results field to get back
                    (c) -> results.add(
                        new TestMapping(c.lookupIndexName(), propertyPrefix + c.lookupPropertySpec() + propertySuffix, settings.build())
                    )
                );

    /**
     * @brief [Functional description for field results]: Describe purpose here.
     */
            return results;
        }

    /**
     * @brief [Functional Utility for docs]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        protected List<TestDocument> docs() {
            List<TestDocument> results = new ArrayList<>();

    /**
     * @brief [Functional description for field docId]: Describe purpose here.
     */
            int docId = 0;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            for (TestConfig config : configs.values()) {
                String doc = String.format(Locale.ROOT, """
                    {
                      %s,
                      "other": "value"
                    }
                    """, lookupPropertyFor(config));
                results.add(new TestDocument(config.lookupIndexName(), "" + (++docId), doc));
            }
            List<String> mainProperties = configs.values()
                .stream()
                .map(LookupJoinTypesIT::mainPropertyFor)
                .distinct()
                .collect(Collectors.toList());
            results.add(new TestDocument(MAIN_INDEX, "1", String.format(Locale.ROOT, """
                {
                  %s
                }
                """, String.join(",\n  ", mainProperties))));

    /**
     * @brief [Functional description for field results]: Describe purpose here.
     */
            return results;
        }

    /**
     * @brief [Functional Utility for values]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        private Collection<TestConfig> values() {
            return configs.values();
        }

    /**
     * @brief [Functional Utility for exists]: Describe purpose here.
     * @param indexName: [Description]
     * @return [ReturnType]: [Description]
     */
        private boolean exists(String indexName) {
            return configs.containsKey(indexName);
        }

    /**
     * @brief [Functional Utility for add]: Describe purpose here.
     * @param config: [Description]
     * @return [ReturnType]: [Description]
     */
        private void add(TestConfig config) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (configs.containsKey(config.lookupIndexName())) {
                throw new IllegalArgumentException("Duplicate index name: " + config.lookupIndexName());
            }
            configs.put(config.lookupIndexName(), config);
        }

    /**
     * @brief [Functional Utility for addPasses]: Describe purpose here.
     * @param mainType: [Description]
     * @param lookupType: [Description]
     * @return [ReturnType]: [Description]
     */
        private void addPasses(DataType mainType, DataType lookupType) {
            add(new TestConfigPasses(mainType, lookupType, true));
        }

    /**
     * @brief [Functional Utility for addFails]: Describe purpose here.
     * @param mainType: [Description]
     * @param lookupType: [Description]
     * @return [ReturnType]: [Description]
     */
        private void addFails(DataType mainType, DataType lookupType) {
            String fieldName = LOOKUP_INDEX_PREFIX + lookupType.esType();
            String errorMessage = String.format(
                Locale.ROOT,
                "JOIN left field [%s] of type [%s] is incompatible with right field [%s] of type [%s]",
                fieldName,
                mainType.widenSmallNumeric(),
                fieldName,
                lookupType.widenSmallNumeric()
            );
            add(
                new TestConfigFails<>(
                    mainType,
                    lookupType,
                    VerificationException.class,
                    e -> assertThat(e.getMessage(), containsString(errorMessage))
                )
            );
        }

    /**
     * @brief [Functional Utility for addFailsUnsupported]: Describe purpose here.
     * @param mainType: [Description]
     * @param lookupType: [Description]
     * @return [ReturnType]: [Description]
     */
        private void addFailsUnsupported(DataType mainType, DataType lookupType) {
            String fieldName = "lookup_" + lookupType.esType();
            String errorMessage = String.format(
                Locale.ROOT,
                "JOIN with right field [%s] of type [%s] is not supported",
                fieldName,
                lookupType
            );
            add(
                new TestConfigFails<>(
                    mainType,
                    lookupType,
                    VerificationException.class,
                    e -> assertThat(e.getMessage(), containsString(errorMessage))
                )
            );
        }
    }

    interface TestConfig {
        DataType mainType();

        DataType lookupType();

        default String lookupIndexName() {
            return LOOKUP_INDEX_PREFIX + mainType().esType() + "_" + lookupType().esType();
        }

        default String mainFieldName() {
            return MAIN_INDEX_PREFIX + mainType().esType();
        }

        default String lookupFieldName() {
            return LOOKUP_INDEX_PREFIX + lookupType().esType();
        }

        default String mainPropertySpec() {
            return propertySpecFor(mainFieldName(), mainType(), "");
        }

        /**
         * If the main field is supposed to be a union type, this will be the property spec for the second index.
         */
        default String secondMainPropertySpecForUnionTypes() {
            return propertySpecFor(mainFieldName(), mainType(), "");
        }

        default String lookupPropertySpec() {
            return propertySpecFor(lookupFieldName(), lookupType(), ", \"other\": { \"type\" : \"keyword\" }");
        }

        /** Make sure the left index has the expected fields and types */
        default void validateMainIndex() {
            validateIndex(MAIN_INDEX, mainFieldName(), sampleDataFor(mainType()));
        }

        /** Make sure the lookup index has the expected fields and types */
        default void validateLookupIndex() {
            validateIndex(lookupIndexName(), lookupFieldName(), sampleDataFor(lookupType()));
        }

        default String testQuery() {
            String mainField = mainFieldName();
            String lookupField = lookupFieldName();
            String lookupIndex = lookupIndexName();

            return String.format(
                Locale.ROOT,
                "FROM %s | RENAME %s AS %s | LOOKUP JOIN %s ON %s | KEEP other",
                MAIN_INDEX,
                mainField,
                lookupField,
                lookupIndex,
                lookupField
            );
        }

        void doTest();
    }

    /**
     * @brief [Functional Utility for propertySpecFor]: Describe purpose here.
     * @param fieldName: [Description]
     * @param type: [Description]
     * @param extra: [Description]
     * @return [ReturnType]: [Description]
     */
    private static String propertySpecFor(String fieldName, DataType type, String extra) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (type == SCALED_FLOAT) {
            return String.format(
                Locale.ROOT,
                "\"%s\": { \"type\" : \"%s\", \"scaling_factor\": %f }",
                fieldName,
                type.esType(),
                SCALING_FACTOR
            ) + extra;
        }
        return String.format(Locale.ROOT, "\"%s\": { \"type\" : \"%s\" }", fieldName, type.esType().replaceAll("cartesian_", "")) + extra;
    }

    /**
     * @brief [Functional Utility for validateIndex]: Describe purpose here.
     * @param indexName: [Description]
     * @param fieldName: [Description]
     * @param expectedValue: [Description]
     * @return [ReturnType]: [Description]
     */
    private static void validateIndex(String indexName, String fieldName, Object expectedValue) {
        String query = String.format(Locale.ROOT, "FROM %s | KEEP %s", indexName, fieldName);
        try (var response = EsqlQueryRequestBuilder.newRequestBuilder(client()).query(query).get()) {
            ColumnInfo info = response.response().columns().getFirst();
            assertThat("Expected index '" + indexName + "' to have column '" + fieldName + ": " + query, info.name(), is(fieldName));
            Iterator<Object> results = response.response().column(0).iterator();
            assertTrue("Expected at least one result for query: " + query, results.hasNext());
            Object indexedResult = response.response().column(0).iterator().next();
            assertThat("Expected valid result: " + query, indexedResult, is(expectedValue));
        }
    }

    private record TestConfigPasses(DataType mainType, DataType lookupType, boolean hasResults) implements TestConfig {
        @Override
    /**
     * @brief [Functional Utility for doTest]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public void doTest() {
            String query = testQuery();
            try (var response = EsqlQueryRequestBuilder.newRequestBuilder(client()).query(query).get()) {
                Iterator<Object> results = response.response().column(0).iterator();
                assertTrue("Expected at least one result for query: " + query, results.hasNext());
                Object indexedResult = response.response().column(0).iterator().next();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (hasResults) {
                    assertThat("Expected valid result: " + query, indexedResult, equalTo("value"));
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    assertThat("Expected empty results for query: " + query, indexedResult, is(nullValue()));
                }
            }
        }
    }

    private record TestConfigFails<E extends Exception>(DataType mainType, DataType lookupType, Class<E> exception, Consumer<E> assertion)
        implements
            TestConfig {
        @Override
    /**
     * @brief [Functional Utility for doTest]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public void doTest() {
            String query = testQuery();
            E e = expectThrows(
                exception(),
                "Expected exception " + exception().getSimpleName() + " but no exception was thrown: " + query,
                () -> {
                    // noinspection EmptyTryBlock
                    try (var ignored = EsqlQueryRequestBuilder.newRequestBuilder(client()).query(query).get()) {
                        // We use try-with-resources to ensure the request is closed if the exception is not thrown (less cluttered errors)
                    }
                }
            );
            assertion().accept(e);
        }
    }

    /**
     * @brief [Functional Utility for isValidDataType]: Describe purpose here.
     * @param dataType: [Description]
     * @return [ReturnType]: [Description]
     */
    private boolean isValidDataType(DataType dataType) {
        return UNDER_CONSTRUCTION.get(dataType) == null || UNDER_CONSTRUCTION.get(dataType).isEnabled();
    }
}
