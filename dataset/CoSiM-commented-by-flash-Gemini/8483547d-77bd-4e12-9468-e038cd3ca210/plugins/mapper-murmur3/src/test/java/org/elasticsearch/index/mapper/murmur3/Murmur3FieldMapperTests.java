/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.mapper.murmur3;

import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.common.hash.MurmurHash3;
import org.elasticsearch.index.fielddata.FieldDataContext;
import org.elasticsearch.index.fielddata.IndexFieldDataCache;
import org.elasticsearch.index.mapper.DocValueFetcher;
import org.elasticsearch.index.mapper.DocumentMapper;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.mapper.MapperService;
import org.elasticsearch.index.mapper.MapperTestCase;
import org.elasticsearch.index.mapper.ParsedDocument;
import org.elasticsearch.index.mapper.SourceToParse;
import org.elasticsearch.index.mapper.ValueFetcher;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.indices.breaker.NoneCircuitBreakerService;
import org.elasticsearch.plugin.mapper.MapperMurmur3Plugin;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.search.lookup.Source;
import org.elasticsearch.search.lookup.SourceProvider;
import org.elasticsearch.xcontent.XContentBuilder;
import org.junit.AssumptionViolatedException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toList;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.hasSize;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * @8483547d-77bd-4e12-9468-e038cd3ca210/plugins/mapper-murmur3/src/test/java/org/elasticsearch/index/mapper/murmur3/Murmur3FieldMapperTests.java
 * @brief Unit tests for the Murmur3 field mapper plugin.
 * 
 * Functional Intent: Validates the specialized mapping logic for 'murmur3' fields, 
 * which compute and store 128-bit hashes of input strings. It ensures that 
 * Lucene-level doc values are correctly configured and that value fetching 
 * logic performs the required hashing to match stored values.
 * 
 * Domain: Search Indices, Hashing Algorithms, Elasticsearch Mapping System.
 */
public class Murmur3FieldMapperTests extends MapperTestCase {

    @Override
    protected Object getSampleValueForDocument() {
        return "value";
    }

    @Override
    protected Collection<? extends Plugin> getPlugins() {
        return List.of(new MapperMurmur3Plugin());
    }

    @Override
    protected void minimalMapping(XContentBuilder b) throws IOException {
        b.field("type", "murmur3");
    }

    @Override
    protected void registerParameters(ParameterChecker checker) throws IOException {
        checker.registerConflictCheck("store", b -> b.field("store", true));
    }

    /**
     * Block Logic: Validates default Lucene field configurations.
     * Logic: Parses a simple document and asserts that the resulting Lucene field 
     * is configured for SORTED_NUMERIC doc values while disabling standard indexing 
     * (as only the hash is stored for lookups, not the raw text).
     */
    public void testDefaults() throws Exception {
        DocumentMapper mapper = createDocumentMapper(fieldMapping(this::minimalMapping));
        ParsedDocument parsedDoc = mapper.parse(source(b -> b.field("field", "value")));
        List<IndexableField> fields = parsedDoc.rootDoc().getFields("field");
        assertNotNull(fields);
        assertThat(fields, hasSize(1));
        IndexableField field = fields.get(0);
        
        // Invariant: Murmur3 fields should not be searchable via standard inverted index.
        assertEquals(IndexOptions.NONE, field.fieldType().indexOptions());
        assertEquals(DocValuesType.SORTED_NUMERIC, field.fieldType().docValuesType());
    }

    @Override
    protected Object generateRandomInputValue(MappedFieldType ft) {
        return randomAlphaOfLength(randomIntBetween(0, 2048));
    }

    /**
     * Block Logic: Validates retrieval and hashing consistency.
     * Logic: 
     * 1. Sets up both a doc-value-based fetcher and a source-based fetcher.
     * 2. Retrieves values from a mock Lucene index.
     * 3. Transforms the raw strings from source by applying the Murmur3 hash algorithm.
     * 4. Verifies that the computed hash from source matches the stored hash in doc values.
     * 
     * @param mapperService The active mapper registry.
     * @param field Target field name.
     * @param value Original input value to test.
     * @param format Expected output format.
     * @throws IOException If index access fails.
     */
    @Override
    protected void assertFetch(MapperService mapperService, String field, Object value, String format) throws IOException {
        MappedFieldType ft = mapperService.fieldType(field);
        MappedFieldType.FielddataOperation fdt = MappedFieldType.FielddataOperation.SEARCH;
        SourceToParse source = source(b -> b.field(ft.name(), value));
        ValueFetcher docValueFetcher = new DocValueFetcher(
            ft.docValueFormat(format, null),
            ft.fielddataBuilder(FieldDataContext.noRuntimeFields("test"))
                .build(new IndexFieldDataCache.None(), new NoneCircuitBreakerService())
        );
        SearchExecutionContext searchExecutionContext = mock(SearchExecutionContext.class);
        when(searchExecutionContext.isSourceEnabled()).thenReturn(true);
        when(searchExecutionContext.sourcePath(field)).thenReturn(Set.of(field));
        when(searchExecutionContext.getForField(ft, fdt)).thenAnswer(inv -> fieldDataLookup(mapperService).apply(ft, () -> {
            throw new UnsupportedOperationException();
        }, fdt));
        ValueFetcher nativeFetcher = ft.valueFetcher(searchExecutionContext, format);
        ParsedDocument doc = mapperService.documentMapper().parse(source);
        withLuceneIndex(mapperService, iw -> iw.addDocuments(doc.docs()), ir -> {
            Source s = SourceProvider.fromLookup(mapperService.mappingLookup(), null, mapperService.getMapperMetrics().sourceFieldMetrics())
                .getSource(ir.leaves().get(0), 0);
            docValueFetcher.setNextReader(ir.leaves().get(0));
            nativeFetcher.setNextReader(ir.leaves().get(0));
            List<Object> fromDocValues = docValueFetcher.fetchValues(s, 0, new ArrayList<>());
            List<Object> fromNative = nativeFetcher.fetchValues(s, 0, new ArrayList<>());
            
            // Block Logic: Dynamic hashing of source-retrieved values.
            // Logic: Simulates the on-the-fly hashing required to compare original 
            // source data with the pre-hashed stored doc values.
            fromNative = fromNative.stream().map(o -> {
                final BytesRef bytes = new BytesRef(o.toString());
                return MurmurHash3.hash128(bytes.bytes, bytes.offset, bytes.length, 0, new MurmurHash3.Hash128()).h1;
            }).collect(toList());

            if (dedupAfterFetch()) {
                fromNative = fromNative.stream().distinct().collect(Collectors.toList());
            }
            
            // Final consistency check between stored hash and re-computed source hash.
            assertThat("fetching " + value, fromNative, containsInAnyOrder(fromDocValues.toArray()));
        });
    }

    @Override
    protected boolean supportsIgnoreMalformed() {
        return false;
    }

    @Override
    protected SyntheticSourceSupport syntheticSourceSupport(boolean ignoreMalformed) {
        throw new AssumptionViolatedException("not supported");
    }

    @Override
    protected IngestScriptSupport ingestScriptSupport() {
        throw new AssumptionViolatedException("not supported");
    }
}
