/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.mapper;

import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.ElasticsearchException;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.geo.ShapeRelation;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.fielddata.FieldDataContext;
import org.elasticsearch.index.fielddata.IndexFieldDataCache;
import org.elasticsearch.index.query.ExistsQueryBuilder;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.indices.breaker.NoneCircuitBreakerService;
import org.elasticsearch.script.Script;
import org.elasticsearch.script.ScriptContext;
import org.elasticsearch.script.ScriptFactory;
import org.elasticsearch.search.lookup.SearchLookup;
import org.elasticsearch.search.lookup.SourceProvider;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.json.JsonXContent;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * @brief Functional description of the AbstractScriptFieldTypeTestCase class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public abstract class AbstractScriptFieldTypeTestCase extends MapperServiceTestCase {

    private static final ToXContent.Params INCLUDE_DEFAULTS = new ToXContent.MapParams(Map.of("include_defaults", "true"));

    protected abstract MappedFieldType simpleMappedFieldType();

    protected abstract MappedFieldType loopFieldType();

    protected abstract String typeName();

    /**
     * Add the provided document to the provided writer, and randomly flush.
     * This is useful for situations where there are not enough documents indexed to trigger random flush and commit performed
     * by {@link RandomIndexWriter}. Flushing is important to obtain multiple slices and inter-segment concurrency.
     */
    protected static <T extends IndexableField> void addDocument(RandomIndexWriter iw, Iterable<T> indexableFields) throws IOException {
        iw.addDocument(indexableFields);
        if (randomBoolean()) {
            iw.flush();
        }
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public final void testMinimalSerializesToItself() throws IOException {
        XContentBuilder orig = JsonXContent.contentBuilder().startObject();
        createMapperService(runtimeFieldMapping(this::minimalMapping)).documentMapper().mapping().toXContent(orig, ToXContent.EMPTY_PARAMS);
        orig.endObject();
        XContentBuilder parsedFromOrig = JsonXContent.contentBuilder().startObject();
        createMapperService(orig).documentMapper().mapping().toXContent(parsedFromOrig, ToXContent.EMPTY_PARAMS);
        parsedFromOrig.endObject();
        assertEquals(Strings.toString(orig), Strings.toString(parsedFromOrig));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public final void testMeta() throws IOException {
        XContentBuilder mapping = runtimeFieldMapping(b -> {
            minimalMapping(b);
            b.field("meta", Collections.singletonMap("foo", "bar"));
        });
        MapperService mapperService = createMapperService(mapping);
        assertEquals(
            XContentHelper.convertToMap(BytesReference.bytes(mapping), false, mapping.contentType()).v2(),
            XContentHelper.convertToMap(mapperService.documentMapper().mappingSource().uncompressed(), false, mapping.contentType()).v2()
        );

        mapping = runtimeFieldMapping(this::minimalMapping);
        merge(mapperService, mapping);
        assertEquals(
            XContentHelper.convertToMap(BytesReference.bytes(mapping), false, mapping.contentType()).v2(),
            XContentHelper.convertToMap(mapperService.documentMapper().mappingSource().uncompressed(), false, mapping.contentType()).v2()
        );

        mapping = runtimeFieldMapping(b -> {
            minimalMapping(b);
            b.field("meta", Collections.singletonMap("baz", "quux"));
        });
        merge(mapperService, mapping);
        assertEquals(
            XContentHelper.convertToMap(BytesReference.bytes(mapping), false, mapping.contentType()).v2(),
            XContentHelper.convertToMap(mapperService.documentMapper().mappingSource().uncompressed(), false, mapping.contentType()).v2()
        );
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public final void testMinimalMappingToMaximal() throws IOException {
        XContentBuilder orig = JsonXContent.contentBuilder().startObject();
        createMapperService(runtimeFieldMapping(this::minimalMapping)).documentMapper().mapping().toXContent(orig, INCLUDE_DEFAULTS);
        orig.endObject();
        XContentBuilder parsedFromOrig = JsonXContent.contentBuilder().startObject();
        createMapperService(orig).documentMapper().mapping().toXContent(parsedFromOrig, INCLUDE_DEFAULTS);
        parsedFromOrig.endObject();
        assertEquals(Strings.toString(orig), Strings.toString(parsedFromOrig));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testCopyToIsNotSupported() throws IOException {
        XContentBuilder mapping = runtimeFieldMapping(b -> {
            minimalMapping(b);
            b.field("copy_to", "target");
        });
        MapperParsingException exception = expectThrows(MapperParsingException.class, () -> createMapperService(mapping));
        assertThat(exception.getMessage(), containsString("unknown parameter [copy_to] on runtime field"));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testMultiFieldsIsNotSupported() throws IOException {
        XContentBuilder mapping = runtimeFieldMapping(b -> {
            minimalMapping(b);
            b.startObject("fields").startObject("test").field("type", "keyword").endObject().endObject();
        });
        MapperParsingException exception = expectThrows(MapperParsingException.class, () -> createMapperService(mapping));
        assertThat(exception.getMessage(), containsString("unknown parameter [fields] on runtime field"));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testStoredScriptsAreNotSupported() throws Exception {
        XContentBuilder mapping = runtimeFieldMapping(b -> {
            b.field("type", typeName());
            b.startObject("script").field("id", "test").endObject();
        });
        MapperParsingException exception = expectThrows(MapperParsingException.class, () -> createMapperService(mapping));
        assertEquals("Failed to parse mapping: stored scripts are not supported for runtime field [field]", exception.getMessage());
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testFieldCaps() throws Exception {
        MapperService scriptIndexMapping = createMapperService(runtimeFieldMapping(this::minimalMapping));
        MapperService concreteIndexMapping;
        {
            XContentBuilder mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("_doc")
                .startObject("properties")
                .startObject("field")
                .field("type", typeName())
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            concreteIndexMapping = createMapperService(mapping);
        }
        MappedFieldType scriptFieldType = scriptIndexMapping.fieldType("field");
        MappedFieldType concreteIndexType = concreteIndexMapping.fieldType("field");
        assertEquals(concreteIndexType.familyTypeName(), scriptFieldType.familyTypeName());
        assertEquals(concreteIndexType.isSearchable(), scriptFieldType.isSearchable());
        assertEquals(concreteIndexType.isAggregatable(), scriptFieldType.isAggregatable());
    }

    /**
     * Check that running query on a runtime field script that fails has the expected behaviour according to its configuration
     */
    public final void testOnScriptError() throws IOException {
        try (Directory directory = newDirectory(); RandomIndexWriter iw = new RandomIndexWriter(random(), directory)) {
            iw.addDocument(List.of(new StoredField("_source", new BytesRef("{\"foo\": [1]}"))));
            try (DirectoryReader reader = iw.getReader()) {
                IndexSearcher searcher = newSearcher(reader);
                {
                    AbstractScriptFieldType<?> fieldType = build("error", Collections.emptyMap(), OnScriptError.CONTINUE);
                    SearchExecutionContext searchExecutionContext = mockContext(true, fieldType);
                    Query query = new ExistsQueryBuilder("test").rewrite(searchExecutionContext).toQuery(searchExecutionContext);
                    assertEquals(0, searcher.count(query));
                }
                {
                    AbstractScriptFieldType<?> fieldType = build("error", Collections.emptyMap(), OnScriptError.FAIL);
                    SearchExecutionContext searchExecutionContext = mockContext(true, fieldType);
                    Query query = new ExistsQueryBuilder("test").rewrite(searchExecutionContext).toQuery(searchExecutionContext);
                    expectThrows(RuntimeException.class, () -> searcher.count(query));
                }
            }
        }
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public final void testOnScriptErrorFail() throws IOException {
        try (Directory directory = newDirectory(); RandomIndexWriter iw = new RandomIndexWriter(random(), directory)) {
            iw.addDocument(List.of(new StoredField("_source", new BytesRef("{\"foo\": [1]}"))));
            try (DirectoryReader reader = iw.getReader()) {
                IndexSearcher searcher = newSearcher(reader);
                AbstractScriptFieldType<?> fieldType = build("error", Collections.emptyMap(), OnScriptError.FAIL);
                SearchExecutionContext searchExecutionContext = mockContext(true, fieldType);
                Query query = new ExistsQueryBuilder("test").rewrite(searchExecutionContext).toQuery(searchExecutionContext);
                expectThrows(RuntimeException.class, () -> searcher.count(query));
            }
        }
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testFieldHasValue() {
        assertTrue(getMappedFieldType().fieldHasValue(new FieldInfos(new FieldInfo[] { getFieldInfoWithName(randomAlphaOfLength(5)) })));
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testFieldHasValueWithEmptyFieldInfos() {
        assertTrue(getMappedFieldType().fieldHasValue(FieldInfos.EMPTY));
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public MappedFieldType getMappedFieldType() {
        return simpleMappedFieldType();
    }

    protected abstract AbstractScriptFieldType<?> build(String error, Map<String, Object> emptyMap, OnScriptError onScriptError);

    @SuppressWarnings("unused")
    public abstract void testDocValues() throws IOException;

    @SuppressWarnings("unused")
    public abstract void testSort() throws IOException;

    @SuppressWarnings("unused")
    public abstract void testUsedInScript() throws IOException;

    @SuppressWarnings("unused")
    public abstract void testExistsQuery() throws IOException;

    @SuppressWarnings("unused")
    public abstract void testRangeQuery() throws IOException;

    protected abstract Query randomRangeQuery(MappedFieldType ft, SearchExecutionContext ctx);

    @SuppressWarnings("unused")
    public abstract void testTermQuery() throws IOException;

    protected abstract Query randomTermQuery(MappedFieldType ft, SearchExecutionContext ctx);

    @SuppressWarnings("unused")
    public abstract void testTermsQuery() throws IOException;

    protected abstract Query randomTermsQuery(MappedFieldType ft, SearchExecutionContext ctx);

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected static SearchExecutionContext mockContext() {
        return mockContext(true);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected static FieldDataContext mockFielddataContext() {
        SearchExecutionContext searchExecutionContext = mockContext();
        return new FieldDataContext(
            "test",
            null,
            searchExecutionContext::lookup,
            mockContext()::sourcePath,
            MappedFieldType.FielddataOperation.SCRIPT
        );
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param allowExpensiveQueries: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected static SearchExecutionContext mockContext(boolean allowExpensiveQueries) {
        return mockContext(allowExpensiveQueries, null);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected boolean supportsTermQueries() {
        return true;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected boolean supportsRangeQueries() {
        return true;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param allowExpensiveQueries: [Description]
     * @param mappedFieldType: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected static SearchExecutionContext mockContext(boolean allowExpensiveQueries, MappedFieldType mappedFieldType) {
        return mockContext(allowExpensiveQueries, mappedFieldType, SourceProvider.fromStoredFields());
    }

    protected static SearchExecutionContext mockContext(
        boolean allowExpensiveQueries,
        MappedFieldType mappedFieldType,
        SourceProvider sourceProvider
    ) {
        SearchExecutionContext context = mock(SearchExecutionContext.class);
        if (mappedFieldType != null) {
            when(context.getFieldType(anyString())).thenReturn(mappedFieldType);
        }
        when(context.allowExpensiveQueries()).thenReturn(allowExpensiveQueries);
        SearchLookup lookup = new SearchLookup(
            context::getFieldType,
            (mft, lookupSupplier, fdo) -> mft.fielddataBuilder(new FieldDataContext("test", null, lookupSupplier, context::sourcePath, fdo))
                .build(null, null),
            sourceProvider
        );
        when(context.lookup()).thenReturn(lookup);
        when(context.getForField(any(), any())).then(args -> {
            MappedFieldType ft = args.getArgument(0);
            MappedFieldType.FielddataOperation fdo = args.getArgument(1);
            return ft.fielddataBuilder(new FieldDataContext("test", null, context::lookup, context::sourcePath, fdo))
                .build(new IndexFieldDataCache.None(), new NoneCircuitBreakerService());
        });
        when(context.getMatchingFieldNames(any())).thenReturn(Set.of("dummy_field"));
        return context;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testExistsQueryIsExpensive() {
        checkExpensiveQuery(MappedFieldType::existsQuery);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testExistsQueryInLoop() {
        checkLoop(MappedFieldType::existsQuery);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testRangeQueryWithShapeRelationIsError() {
        Exception e = expectThrows(
            IllegalArgumentException.class,
            () -> simpleMappedFieldType().rangeQuery(1, 2, true, true, ShapeRelation.DISJOINT, null, null, null)
        );
        assertThat(e.getMessage(), equalTo("Runtime field [test] of type [" + typeName() + "] does not support DISJOINT ranges"));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testRangeQueryIsExpensive() {
        assumeTrue("Impl does not support range queries", supportsRangeQueries());
        checkExpensiveQuery(this::randomRangeQuery);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testRangeQueryInLoop() {
        assumeTrue("Impl does not support range queries", supportsRangeQueries());
        checkLoop(this::randomRangeQuery);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testTermQueryIsExpensive() {
        assumeTrue("Impl does not support term queries", supportsTermQueries());
        checkExpensiveQuery(this::randomTermQuery);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testTermQueryInLoop() {
        assumeTrue("Impl does not support term queries", supportsTermQueries());
        checkLoop(this::randomTermQuery);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testTermsQueryIsExpensive() {
        assumeTrue("Impl does not support term queries", supportsTermQueries());
        checkExpensiveQuery(this::randomTermsQuery);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testTermsQueryInLoop() {
        assumeTrue("Impl does not support term queries", supportsTermQueries());
        checkLoop(this::randomTermsQuery);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testPhraseQueryIsError() {
        assumeTrue("Impl does not support term queries", supportsTermQueries());
        assertQueryOnlyOnText("phrase", () -> simpleMappedFieldType().phraseQuery(null, 1, false, null));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testPhrasePrefixQueryIsError() {
        assumeTrue("Impl does not support term queries", supportsTermQueries());
        assertQueryOnlyOnText("phrase prefix", () -> simpleMappedFieldType().phrasePrefixQuery(null, 1, 1, null));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testMultiPhraseQueryIsError() {
        assumeTrue("Impl does not support term queries", supportsTermQueries());
        assertQueryOnlyOnText("phrase", () -> simpleMappedFieldType().multiPhraseQuery(null, 1, false, null));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testSpanPrefixQueryIsError() {
        assumeTrue("Impl does not support term queries", supportsTermQueries());
        assertQueryOnlyOnText("span prefix", () -> simpleMappedFieldType().spanPrefixQuery(null, null, null));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public final void testCacheable() throws IOException {
        XContentBuilder mapping = runtimeMapping(b -> {
            b.startObject("field")
                .field("type", typeName())
                .startObject("script")
                .field("source", "dummy_source")
                .field("lang", "test")
                .endObject()
                .endObject()
                .startObject("field_source")
                .field("type", typeName())
                .startObject("script")
                .field("source", "deterministic_source")
                .field("lang", "test")
                .endObject()
                .endObject();
        });

        MapperService mapperService = createMapperService(mapping);

        {
            SearchExecutionContext c = createSearchExecutionContext(mapperService);
            c.getFieldType("field").existsQuery(c);
            assertFalse(c.isCacheable());
        }

        {
            SearchExecutionContext c = createSearchExecutionContext(mapperService);
            c.getFieldType("field_source").existsQuery(c);
            assertTrue(c.isCacheable());
        }
    }

    protected final List<Object> blockLoaderReadValuesFromColumnAtATimeReader(DirectoryReader reader, MappedFieldType fieldType)
        throws IOException {
        BlockLoader loader = fieldType.blockLoader(blContext());
        List<Object> all = new ArrayList<>();
        for (LeafReaderContext ctx : reader.leaves()) {
            TestBlock block = (TestBlock) loader.columnAtATimeReader(ctx)
                .read(TestBlock.factory(ctx.reader().numDocs()), TestBlock.docs(ctx));
            for (int i = 0; i < block.size(); i++) {
                all.add(block.get(i));
            }
        }
        return all;
    }

    protected final List<Object> blockLoaderReadValuesFromRowStrideReader(DirectoryReader reader, MappedFieldType fieldType)
        throws IOException {
        BlockLoader loader = fieldType.blockLoader(blContext());
        List<Object> all = new ArrayList<>();
        for (LeafReaderContext ctx : reader.leaves()) {
            BlockLoader.RowStrideReader blockReader = loader.rowStrideReader(ctx);
            BlockLoader.Builder builder = loader.builder(TestBlock.factory(ctx.reader().numDocs()), ctx.reader().numDocs());
            for (int i = 0; i < ctx.reader().numDocs(); i++) {
                blockReader.read(i, null, builder);
            }
            TestBlock block = (TestBlock) builder.build();
            for (int i = 0; i < block.size(); i++) {
                all.add(block.get(i));
            }
        }
        return all;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    private MappedFieldType.BlockLoaderContext blContext() {
        return new MappedFieldType.BlockLoaderContext() {
            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public String indexName() {
                throw new UnsupportedOperationException();
            }

            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public IndexSettings indexSettings() {
                throw new UnsupportedOperationException();
            }

            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public MappedFieldType.FieldExtractPreference fieldExtractPreference() {
                return MappedFieldType.FieldExtractPreference.NONE;
            }

            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public SearchLookup lookup() {
                return mockContext().lookup();
            }

            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param name: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public Set<String> sourcePaths(String name) {
                throw new UnsupportedOperationException();
            }

            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param field: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public String parentField(String field) {
                throw new UnsupportedOperationException();
            }

            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public FieldNamesFieldMapper.FieldNamesFieldType fieldNames() {
                return FieldNamesFieldMapper.FieldNamesFieldType.get(true);
            }
        };
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param queryName: [Description]
     * @param buildQuery: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    private void assertQueryOnlyOnText(String queryName, ThrowingRunnable buildQuery) {
        Exception e = expectThrows(IllegalArgumentException.class, buildQuery);
        assertThat(
            e.getMessage(),
            equalTo(
                "Can only use "
                    + queryName
                    + " queries on text fields - not on [test] which is a runtime field of type ["
                    + typeName()
                    + "]"
            )
        );
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param reader: [Description]
     * @param docId: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected final String readSource(IndexReader reader, int docId) throws IOException {
        return reader.document(docId).getBinaryValue("_source").utf8ToString();
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param BiConsumer<MappedFieldType: [Description]
     * @param queryBuilder: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected final void checkExpensiveQuery(BiConsumer<MappedFieldType, SearchExecutionContext> queryBuilder) {
        Exception e = expectThrows(ElasticsearchException.class, () -> queryBuilder.accept(simpleMappedFieldType(), mockContext(false)));
        assertThat(
            e.getMessage(),
            equalTo("queries cannot be executed against runtime fields while [search.allow_expensive_queries] is set to [false].")
        );
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param BiConsumer<MappedFieldType: [Description]
     * @param queryBuilder: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected final void checkLoop(BiConsumer<MappedFieldType, SearchExecutionContext> queryBuilder) {
        Exception e = expectThrows(IllegalArgumentException.class, () -> queryBuilder.accept(loopFieldType(), mockContext()));
        assertThat(e.getMessage(), equalTo("Cyclic dependency detected while resolving runtime fields: test -> test"));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param b: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected final void minimalMapping(XContentBuilder b) throws IOException {
        b.field("type", typeName());
        b.startObject("script").field("source", "dummy_source").field("lang", "test").endObject();
    }

    protected abstract ScriptFactory parseFromSource();

    protected abstract ScriptFactory dummyScript();

    @Override
    @SuppressWarnings("unchecked")
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param script: [Description]
     * @param context: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected <T> T compileScript(Script script, ScriptContext<T> context) {
        boolean deterministicSource = "deterministic_source".equals(script.getIdOrCode());
        return deterministicSource ? (T) parseFromSource() : (T) dummyScript();
    }
}
