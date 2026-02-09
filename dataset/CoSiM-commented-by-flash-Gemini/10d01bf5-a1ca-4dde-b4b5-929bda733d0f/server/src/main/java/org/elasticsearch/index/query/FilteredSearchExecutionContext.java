/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.query;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.similarities.Similarity;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.common.io.stream.NamedWriteableRegistry;
import org.elasticsearch.index.Index;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.analysis.IndexAnalyzers;
import org.elasticsearch.index.analysis.NamedAnalyzer;
import org.elasticsearch.index.fielddata.IndexFieldData;
import org.elasticsearch.index.mapper.DocumentParsingException;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.mapper.MappingLookup;
import org.elasticsearch.index.mapper.NestedLookup;
import org.elasticsearch.index.mapper.ParsedDocument;
import org.elasticsearch.index.mapper.SourceLoader;
import org.elasticsearch.index.mapper.SourceToParse;
import org.elasticsearch.index.query.support.NestedScope;
import org.elasticsearch.script.Script;
import org.elasticsearch.script.ScriptContext;
import org.elasticsearch.search.NestedDocuments;
import org.elasticsearch.search.aggregations.support.ValuesSourceRegistry;
import org.elasticsearch.search.lookup.LeafFieldLookupProvider;
import org.elasticsearch.search.lookup.SearchLookup;
import org.elasticsearch.search.lookup.SourceProvider;
import org.elasticsearch.xcontent.XContentParserConfiguration;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * This is NOT a simple clone of the SearchExecutionContext.
 * While it does "clone-esque" things, it delegates everything it can to the passed search execution context.
 *
 * Do NOT use this if you mean to clone the context as you are planning to make modifications
 */
/**
 * @brief Functional description of the FilteredSearchExecutionContext class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class FilteredSearchExecutionContext extends SearchExecutionContext {
    private final SearchExecutionContext in;

    public FilteredSearchExecutionContext(SearchExecutionContext in) {
        super(in);
        this.in = in;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Similarity getSearchSimilarity() {
        return in.getSearchSimilarity();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Similarity getDefaultSimilarity() {
        return in.getDefaultSimilarity();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public List<String> defaultFields() {
        return in.defaultFields();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean queryStringLenient() {
        return in.queryStringLenient();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean queryStringAnalyzeWildcard() {
        return in.queryStringAnalyzeWildcard();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean queryStringAllowLeadingWildcard() {
        return in.queryStringAllowLeadingWildcard();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param filter: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public BitSetProducer bitsetFilter(Query filter) {
        return in.bitsetFilter(filter);
    }

    @Override
    public <IFD extends IndexFieldData<?>> IFD getForField(
        MappedFieldType fieldType,
        MappedFieldType.FielddataOperation fielddataOperation
    ) {
        return in.getForField(fieldType, fielddataOperation);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param name: [Description]
     * @param query: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addNamedQuery(String name, Query query) {
        in.addNamedQuery(name, query);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Map<String, Query> copyNamedQueries() {
        return in.copyNamedQueries();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param source: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ParsedDocument parseDocument(SourceToParse source) throws DocumentParsingException {
        return in.parseDocument(source);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public NestedLookup nestedLookup() {
        return in.nestedLookup();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean hasMappings() {
        return in.hasMappings();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param name: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean isFieldMapped(String name) {
        return in.isFieldMapped(name);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param field: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean isMetadataField(String field) {
        return in.isMetadataField(field);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param field: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean isMultiField(String field) {
        return in.isMultiField(field);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param fullName: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Set<String> sourcePath(String fullName) {
        return in.sourcePath(fullName);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean isSourceEnabled() {
        return in.isSourceEnabled();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean isSourceSynthetic() {
        return in.isSourceSynthetic();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param forceSyntheticSource: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SourceLoader newSourceLoader(boolean forceSyntheticSource) {
        return in.newSourceLoader(forceSyntheticSource);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param type: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public MappedFieldType buildAnonymousFieldType(String type) {
        return in.buildAnonymousFieldType(type);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param Function<String: [Description]
     * @param unindexedFieldAnalyzer: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Analyzer getIndexAnalyzer(Function<String, NamedAnalyzer> unindexedFieldAnalyzer) {
        return in.getIndexAnalyzer(unindexedFieldAnalyzer);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param allowedFields: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void setAllowedFields(Predicate<String> allowedFields) {
        in.setAllowedFields(allowedFields);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param field: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean containsBrokenAnalysis(String field) {
        return in.containsBrokenAnalysis(field);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchLookup lookup() {
        return in.lookup();
    }

    @Override
    public void setLookupProviders(
        SourceProvider sourceProvider,
        Function<LeafReaderContext, LeafFieldLookupProvider> fieldLookupProvider
    ) {
        in.setLookupProviders(sourceProvider, fieldLookupProvider);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public NestedScope nestedScope() {
        return in.nestedScope();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IndexVersion indexVersionCreated() {
        return in.indexVersionCreated();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param field: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean indexSortedOnField(String field) {
        return in.indexSortedOnField(field);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param queryBuilder: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ParsedQuery toQuery(QueryBuilder queryBuilder) {
        return in.toQuery(queryBuilder);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Index index() {
        return in.index();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param script: [Description]
     * @param context: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public <FactoryType> FactoryType compile(Script script, ScriptContext<FactoryType> context) {
        return in.compile(script, context);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void disableCache() {
        in.disableCache();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param BiConsumer<Client: [Description]
     * @param asyncAction: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void registerAsyncAction(BiConsumer<Client, ActionListener<?>> asyncAction) {
        in.registerAsyncAction(asyncAction);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param listener: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void executeAsyncActions(ActionListener<Void> listener) {
        in.executeAsyncActions(listener);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int getShardId() {
        return in.getShardId();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int getShardRequestIndex() {
        return in.getShardRequestIndex();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public long nowInMillis() {
        return in.nowInMillis();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Client getClient() {
        return in.getClient();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IndexReader getIndexReader() {
        return in.getIndexReader();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IndexSearcher searcher() {
        return in.searcher();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param fieldname: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean fieldExistsInIndex(String fieldname) {
        return in.fieldExistsInIndex(fieldname);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public MappingLookup.CacheKey mappingCacheKey() {
        return in.mappingCacheKey();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public NestedDocuments getNestedDocuments() {
        return in.getNestedDocuments();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public XContentParserConfiguration getParserConfig() {
        return in.getParserConfig();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public CoordinatorRewriteContext convertToCoordinatorRewriteContext() {
        return in.convertToCoordinatorRewriteContext();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public QueryRewriteContext convertToIndexMetadataContext() {
        return in.convertToIndexMetadataContext();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public DataRewriteContext convertToDataRewriteContext() {
        return in.convertToDataRewriteContext();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param name: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public MappedFieldType getFieldType(String name) {
        return in.getFieldType(name);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param name: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    protected MappedFieldType fieldType(String name) {
        return in.fieldType(name);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IndexAnalyzers getIndexAnalyzers() {
        return in.getIndexAnalyzers();
    }

    @Override
    MappedFieldType failIfFieldMappingNotFound(String name, MappedFieldType fieldMapping) {
        return in.failIfFieldMappingNotFound(name, fieldMapping);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param allowUnmappedFields: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void setAllowUnmappedFields(boolean allowUnmappedFields) {
        in.setAllowUnmappedFields(allowUnmappedFields);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param mapUnmappedFieldAsString: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void setMapUnmappedFieldAsString(boolean mapUnmappedFieldAsString) {
        in.setMapUnmappedFieldAsString(mapUnmappedFieldAsString);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public NamedWriteableRegistry getWriteableRegistry() {
        return in.getWriteableRegistry();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ValuesSourceRegistry getValuesSourceRegistry() {
        return in.getValuesSourceRegistry();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean allowExpensiveQueries() {
        return in.allowExpensiveQueries();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean hasAsyncActions() {
        return in.hasAsyncActions();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Index getFullyQualifiedIndex() {
        return in.getFullyQualifiedIndex();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IndexSettings getIndexSettings() {
        return in.getIndexSettings();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param pattern: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean indexMatches(String pattern) {
        return in.indexMatches(pattern);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param pattern: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Set<String> getMatchingFieldNames(String pattern) {
        return in.getMatchingFieldNames(pattern);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void setRewriteToNamedQueries() {
        in.setRewriteToNamedQueries();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean rewriteToNamedQuery() {
        return in.rewriteToNamedQuery();
    }
}
