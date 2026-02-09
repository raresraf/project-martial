/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.mapper;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.index.IndexableField;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.mapper.DocumentMapper;
import org.elasticsearch.index.mapper.DocumentParsingException;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.mapper.MapperService;
import org.elasticsearch.index.mapper.MapperTestCase;
import org.elasticsearch.index.mapper.ParsedDocument;
import org.elasticsearch.index.mapper.SourceToParse;
import org.elasticsearch.index.mapper.ValueFetcher;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.search.lookup.Source;
import org.elasticsearch.search.lookup.SourceProvider;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.InferencePlugin;
import org.junit.AssumptionViolatedException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * @brief Functional description of the OffsetSourceFieldMapperTests class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class OffsetSourceFieldMapperTests extends MapperTestCase {
    @Override
    protected Collection<? extends Plugin> getPlugins() {
        return List.of(new InferencePlugin(Settings.EMPTY));
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param b: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected void minimalMapping(XContentBuilder b) throws IOException {
        b.field("type", "offset_source");
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected Object getSampleValueForDocument() {
        return getSampleObjectForDocument();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected Object getSampleObjectForDocument() {
        return Map.of("field", "foo", "start", 100, "end", 300);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param ft: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected Object generateRandomInputValue(MappedFieldType ft) {
        return new OffsetSourceFieldMapper.OffsetSource("field", randomIntBetween(0, 100), randomIntBetween(101, 1000));
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected IngestScriptSupport ingestScriptSupport() {
        throw new AssumptionViolatedException("not supported");
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param checker: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected void registerParameters(ParameterChecker checker) throws IOException {}

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param fieldType: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected void assertSearchable(MappedFieldType fieldType) {
        assertFalse(fieldType.isSearchable());
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected boolean supportsStoredFields() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected boolean supportsEmptyInputArray() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected boolean supportsCopyTo() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected boolean supportsIgnoreMalformed() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param ignoreMalformed: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected SyntheticSourceSupport syntheticSourceSupport(boolean ignoreMalformed) {
        return new SyntheticSourceSupport() {
            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param maxValues: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public SyntheticSourceExample example(int maxValues) {
                return new SyntheticSourceExample(getSampleValueForDocument(), getSampleValueForDocument(), b -> minimalMapping(b));
            }

            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public List<SyntheticSourceInvalidExample> invalidExample() {
                return List.of();
            }
        };
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testSyntheticSourceKeepArrays() {
        // This mapper doesn't support multiple values (array of objects).
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testDefaults() throws Exception {
        DocumentMapper mapper = createDocumentMapper(fieldMapping(this::minimalMapping));
        assertEquals(Strings.toString(fieldMapping(this::minimalMapping)), mapper.mappingSource().toString());

        ParsedDocument doc1 = mapper.parse(
            source(b -> b.startObject("field").field("field", "foo").field("start", 0).field("end", 128).endObject())
        );
        List<IndexableField> fields = doc1.rootDoc().getFields("field");
        assertEquals(1, fields.size());
        assertThat(fields.get(0), instanceOf(OffsetSourceField.class));
        OffsetSourceField offsetField1 = (OffsetSourceField) fields.get(0);

        ParsedDocument doc2 = mapper.parse(
            source(b -> b.startObject("field").field("field", "bar").field("start", 128).field("end", 512).endObject())
        );
        OffsetSourceField offsetField2 = (OffsetSourceField) doc2.rootDoc().getFields("field").get(0);

        assertTokenStream(offsetField1.tokenStream(null, null), "foo", 0, 128);
        assertTokenStream(offsetField2.tokenStream(null, null), "bar", 128, 512);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param tk: [Description]
     * @param expectedTerm: [Description]
     * @param expectedStartOffset: [Description]
     * @param expectedEndOffset: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    private void assertTokenStream(TokenStream tk, String expectedTerm, int expectedStartOffset, int expectedEndOffset) throws IOException {
        CharTermAttribute termAttribute = tk.addAttribute(CharTermAttribute.class);
        OffsetAttribute offsetAttribute = tk.addAttribute(OffsetAttribute.class);
        tk.reset();
        assertTrue(tk.incrementToken());
        assertThat(new String(termAttribute.buffer(), 0, termAttribute.length()), equalTo(expectedTerm));
        assertThat(offsetAttribute.startOffset(), equalTo(expectedStartOffset));
        assertThat(offsetAttribute.endOffset(), equalTo(expectedEndOffset));
        assertFalse(tk.incrementToken());
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param mapperService: [Description]
     * @param field: [Description]
     * @param value: [Description]
     * @param format: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected void assertFetch(MapperService mapperService, String field, Object value, String format) throws IOException {
        MappedFieldType ft = mapperService.fieldType(field);
        MappedFieldType.FielddataOperation fdt = MappedFieldType.FielddataOperation.SEARCH;
        SourceToParse source = source(b -> b.field(ft.name(), value));
        SearchExecutionContext searchExecutionContext = mock(SearchExecutionContext.class);
        when(searchExecutionContext.isSourceEnabled()).thenReturn(true);
        when(searchExecutionContext.sourcePath(field)).thenReturn(Set.of(field));
        when(searchExecutionContext.getForField(ft, fdt)).thenAnswer(inv -> fieldDataLookup(mapperService).apply(ft, () -> {
            throw new UnsupportedOperationException();
        }, fdt));
        ValueFetcher nativeFetcher = ft.valueFetcher(searchExecutionContext, format);
        ParsedDocument doc = mapperService.documentMapper().parse(source);
        withLuceneIndex(mapperService, iw -> iw.addDocuments(doc.docs()), ir -> {
            Source s = SourceProvider.fromStoredFields().getSource(ir.leaves().get(0), 0);
            nativeFetcher.setNextReader(ir.leaves().get(0));
            List<Object> fromNative = nativeFetcher.fetchValues(s, 0, new ArrayList<>());
            assertThat(fromNative.size(), equalTo(1));
            assertThat("fetching " + value, fromNative.get(0), equalTo(value));
        });
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param mapperService: [Description]
     * @param field: [Description]
     * @param value: [Description]
     * @param format: [Description]
     * @param count: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected void assertFetchMany(MapperService mapperService, String field, Object value, String format, int count) throws IOException {
        assumeFalse("[offset_source] currently don't support multiple values in the same field", false);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testInvalidCharset() {
        var exc = expectThrows(Exception.class, () -> createDocumentMapper(mapping(b -> {
            b.startObject("field").field("type", "offset_source").field("charset", "utf_8").endObject();
        })));
        assertThat(exc.getCause().getMessage(), containsString("Unknown value [utf_8] for field [charset]"));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testRejectMultiValuedFields() throws IOException {
        DocumentMapper mapper = createDocumentMapper(mapping(b -> { b.startObject("field").field("type", "offset_source").endObject(); }));

        DocumentParsingException exc = expectThrows(DocumentParsingException.class, () -> mapper.parse(source(b -> {
            b.startArray("field");
            {
                b.startObject().field("field", "bar1").field("start", 128).field("end", 512).endObject();
                b.startObject().field("field", "bar2").field("start", 128).field("end", 512).endObject();
            }
            b.endArray();
        })));
        assertThat(exc.getCause().getMessage(), containsString("[offset_source] fields do not support indexing multiple values"));
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testInvalidOffsets() throws IOException {
        DocumentMapper mapper = createDocumentMapper(mapping(b -> { b.startObject("field").field("type", "offset_source").endObject(); }));

        DocumentParsingException exc = expectThrows(DocumentParsingException.class, () -> mapper.parse(source(b -> {
            b.startArray("field");
            {
                b.startObject().field("field", "bar1").field("start", -1).field("end", 512).endObject();
            }
            b.endArray();
        })));
        assertThat(exc.getCause().getCause().getCause().getMessage(), containsString("Illegal offsets"));
    }
}
