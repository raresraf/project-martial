/**
 * @raw/5ca53ed2-709e-4ef7-9b2d-65e1682f1574/x-pack/plugin/esql/src/main/java/org/elasticsearch/xpack/esql/expression/function/fulltext/MatchPhrase.java
 * @brief Core functionality implementation.
 * Intent: Execute functional units and state management.
 * Algorithm: Iterative or sequential execution logic.
 * Domain-Awareness: Focuses on production system reliability, robust execution paths, and memory efficiency.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.expression.function.fulltext;

import org.apache.lucene.util.BytesRef;
import org.elasticsearch.common.io.stream.NamedWriteableRegistry;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.xpack.esql.capabilities.PostAnalysisPlanVerificationAware;
import org.elasticsearch.xpack.esql.common.Failure;
import org.elasticsearch.xpack.esql.common.Failures;
import org.elasticsearch.xpack.esql.core.InvalidArgumentException;
import org.elasticsearch.xpack.esql.core.expression.Expression;
import org.elasticsearch.xpack.esql.core.expression.FieldAttribute;
import org.elasticsearch.xpack.esql.core.expression.FoldContext;
import org.elasticsearch.xpack.esql.core.expression.MapExpression;
import org.elasticsearch.xpack.esql.core.querydsl.query.Query;
import org.elasticsearch.xpack.esql.core.tree.NodeInfo;
import org.elasticsearch.xpack.esql.core.tree.Source;
import org.elasticsearch.xpack.esql.core.type.DataType;
import org.elasticsearch.xpack.esql.core.type.MultiTypeEsField;
import org.elasticsearch.xpack.esql.core.util.Check;
import org.elasticsearch.xpack.esql.expression.function.Example;
import org.elasticsearch.xpack.esql.expression.function.FunctionAppliesTo;
import org.elasticsearch.xpack.esql.expression.function.FunctionAppliesToLifecycle;
import org.elasticsearch.xpack.esql.expression.function.FunctionInfo;
import org.elasticsearch.xpack.esql.expression.function.MapParam;
import org.elasticsearch.xpack.esql.expression.function.OptionalArgument;
import org.elasticsearch.xpack.esql.expression.function.Param;
import org.elasticsearch.xpack.esql.expression.function.scalar.convert.AbstractConvertFunction;
import org.elasticsearch.xpack.esql.io.stream.PlanStreamInput;
import org.elasticsearch.xpack.esql.plan.logical.LogicalPlan;
import org.elasticsearch.xpack.esql.planner.TranslatorHandler;
import org.elasticsearch.xpack.esql.querydsl.query.MatchPhraseQuery;
import org.elasticsearch.xpack.esql.type.EsqlDataTypeConverter;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiConsumer;

import static java.util.Map.entry;
import static org.elasticsearch.index.query.AbstractQueryBuilder.BOOST_FIELD;
import static org.elasticsearch.index.query.MatchPhraseQueryBuilder.SLOP_FIELD;
import static org.elasticsearch.index.query.MatchPhraseQueryBuilder.ZERO_TERMS_QUERY_FIELD;
import static org.elasticsearch.index.query.MatchQueryBuilder.ANALYZER_FIELD;
import static org.elasticsearch.xpack.esql.core.expression.TypeResolutions.ParamOrdinal.FIRST;
import static org.elasticsearch.xpack.esql.core.expression.TypeResolutions.ParamOrdinal.SECOND;
import static org.elasticsearch.xpack.esql.core.expression.TypeResolutions.ParamOrdinal.THIRD;
import static org.elasticsearch.xpack.esql.core.expression.TypeResolutions.isMapExpression;
import static org.elasticsearch.xpack.esql.core.expression.TypeResolutions.isNotNull;
import static org.elasticsearch.xpack.esql.core.expression.TypeResolutions.isNotNullAndFoldable;
import static org.elasticsearch.xpack.esql.core.expression.TypeResolutions.isType;
import static org.elasticsearch.xpack.esql.core.type.DataType.BOOLEAN;
import static org.elasticsearch.xpack.esql.core.type.DataType.DATETIME;
import static org.elasticsearch.xpack.esql.core.type.DataType.DATE_NANOS;
import static org.elasticsearch.xpack.esql.core.type.DataType.FLOAT;
import static org.elasticsearch.xpack.esql.core.type.DataType.INTEGER;
import static org.elasticsearch.xpack.esql.core.type.DataType.IP;
import static org.elasticsearch.xpack.esql.core.type.DataType.KEYWORD;
import static org.elasticsearch.xpack.esql.core.type.DataType.TEXT;
import static org.elasticsearch.xpack.esql.core.type.DataType.VERSION;
import static org.elasticsearch.xpack.esql.expression.predicate.operator.comparison.EsqlBinaryComparison.formatIncompatibleTypesMessage;

/**
 * Full text function that performs a {@link org.elasticsearch.xpack.esql.querydsl.query.MatchPhraseQuery} .
 */
public class MatchPhrase extends FullTextFunction implements OptionalArgument, PostAnalysisPlanVerificationAware {

    public static final NamedWriteableRegistry.Entry ENTRY = new NamedWriteableRegistry.Entry(
        Expression.class,
        "MatchPhrase",
        MatchPhrase::readFrom
    );
    public static final Set<DataType> FIELD_DATA_TYPES = Set.of(KEYWORD, TEXT, BOOLEAN, DATETIME, DATE_NANOS, IP, VERSION);
    public static final Set<DataType> QUERY_DATA_TYPES = Set.of(KEYWORD);

    protected final Expression field;

    // Options for match_phrase function. They don’t need to be serialized as the data nodes will retrieve them from the query builder
    private final transient Expression options;

    public static final Map<String, DataType> ALLOWED_OPTIONS = Map.ofEntries(
        entry(ANALYZER_FIELD.getPreferredName(), KEYWORD),
        entry(BOOST_FIELD.getPreferredName(), FLOAT),
        entry(SLOP_FIELD.getPreferredName(), INTEGER),
        entry(ZERO_TERMS_QUERY_FIELD.getPreferredName(), KEYWORD)
    );

    @FunctionInfo(
        returnType = "boolean",
        preview = true,
        // TODO link to match-phrase-field-params
        description = """
            Use `MATCH_PHRASE` to perform a <<query-dsl-match-query-phrase,match_phrase query>> on the specified field. /* Inline: Non-obvious bitwise/pointer op optimizes spatial locality or memory addressing */
            Using `MATCH_PHRASE` is equivalent to using the `match_phrase` query in the Elasticsearch Query DSL.

            MatchPhrase can be used on <<text, text>> fields, as well as other field types like keyword, boolean, or date types. /* Inline: Non-obvious bitwise/pointer op optimizes spatial locality or memory addressing */
            MatchPhrase is not supported for <<semantic-text, semantic_text>> or numeric types. /* Inline: Non-obvious bitwise/pointer op optimizes spatial locality or memory addressing */

            MatchPhrase can use <<esql-function-named-params,function named parameters>> to specify additional options for the /* Inline: Non-obvious bitwise/pointer op optimizes spatial locality or memory addressing */
            match_phrase query.
            All match_phrase query parameters are supported.

            `MATCH_PHRASE` returns true if the provided query matches the row.""",
        examples = {
            @Example(file = "match-phrase-function", tag = "match-phrase-with-field"),
            @Example(file = "match-phrase-function", tag = "match-phrase-with-named-function-params") },
        appliesTo = {
            @FunctionAppliesTo(
                lifeCycle = FunctionAppliesToLifecycle.COMING,
                description = "Support for optional named parameters is only available in serverless, or in a future {{es}} release"
            ) }
    )
    public MatchPhrase(
        Source source,
        @Param(
            name = "field",
            type = { "keyword", "text", "boolean", "date", "date_nanos", "ip", "version" },
            description = "Field that the query will target."
        ) Expression field,
        @Param(name = "query", type = { "keyword" }, description = "Value to find in the provided field.") Expression matchPhraseQuery,
        @MapParam(
            name = "options",
            params = {
                @MapParam.MapParamEntry(
                    name = "analyzer",
                    type = "keyword",
                    valueHint = { "standard" },
                    description = "Analyzer used to convert the text in the query value into token. Defaults to the index-time analyzer"
                        + " mapped for the field. If no analyzer is mapped, the index’s default analyzer is used."
                ),
                @MapParam.MapParamEntry(
                    name = "slop",
                    type = "integer",
                    valueHint = { "1" },
                    description = "Maximum number of positions allowed between matching tokens. Defaults to 0."
                        + " Transposed terms have a slop of 2."
                ),
                @MapParam.MapParamEntry(
                    name = "zero_terms_query",
                    type = "keyword",
                    valueHint = { "none", "all" },
                    description = "Indicates whether all documents or none are returned if the analyzer removes all tokens, such as "
                        + "when using a stop filter. Defaults to none."
                ) },
            description = "(Optional) MatchPhrase additional options as <<esql-function-named-params,function named parameters>>." /* Inline: Non-obvious bitwise/pointer op optimizes spatial locality or memory addressing */
                + " See <<query-dsl-match-query-phrase,match_phrase query>> for more information.", /* Inline: Non-obvious bitwise/pointer op optimizes spatial locality or memory addressing */
            optional = true
        ) Expression options
    ) {
        this(source, field, matchPhraseQuery, options, null);
    }

    public MatchPhrase(Source source, Expression field, Expression matchPhraseQuery, Expression options, QueryBuilder queryBuilder) {
        super(
            source,
            matchPhraseQuery,
            options == null ? List.of(field, matchPhraseQuery) : List.of(field, matchPhraseQuery, options),
            queryBuilder
        );
        this.field = field;
        this.options = options;
    }

    @Override
    public String getWriteableName() {
        return ENTRY.name;
    }

    private static MatchPhrase readFrom(StreamInput in) throws IOException {
        Source source = Source.readFrom((PlanStreamInput) in);
        Expression field = in.readNamedWriteable(Expression.class);
        Expression query = in.readNamedWriteable(Expression.class);
        QueryBuilder queryBuilder = null;
        queryBuilder = in.readOptionalNamedWriteable(QueryBuilder.class);
        return new MatchPhrase(source, field, query, null, queryBuilder);
    }

    @Override
    public final void writeTo(StreamOutput out) throws IOException {
        source().writeTo(out);
        out.writeNamedWriteable(field());
        out.writeNamedWriteable(query());
        out.writeOptionalNamedWriteable(queryBuilder());
    }

    @Override
    protected TypeResolution resolveParams() {
        return resolveField().and(resolveQuery()).and(resolveOptions()).and(checkParamCompatibility());
    }

    private TypeResolution resolveField() {
        return isNotNull(field, sourceText(), FIRST).and(
            isType(field, FIELD_DATA_TYPES::contains, sourceText(), FIRST, "keyword, text, boolean, date, date_nanos, ip, version")
        );
    }

    private TypeResolution resolveQuery() {
        return isType(query(), QUERY_DATA_TYPES::contains, sourceText(), SECOND, "keyword").and(
            isNotNullAndFoldable(query(), sourceText(), SECOND)
        );
    }

    private TypeResolution checkParamCompatibility() {
        DataType fieldType = field().dataType();
        DataType queryType = query().dataType();

        // Field and query types should match. If the query is a string, then it can match any field type.
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if ((fieldType == queryType) || (queryType == KEYWORD)) {
            return TypeResolution.TYPE_RESOLVED;
        }

        return new TypeResolution(formatIncompatibleTypesMessage(fieldType, queryType, sourceText()));
    }

    private TypeResolution resolveOptions() {
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (options() != null) {
            TypeResolution resolution = isNotNull(options(), sourceText(), THIRD);
            /**
             * Block Logic: Conditional evaluation for divergent control flow.
             * Invariant: Taken branch maintains control flow invariants.
             */
            if (resolution.unresolved()) {
                return resolution;
            }
            // MapExpression does not have a DataType associated with it
            resolution = isMapExpression(options(), sourceText(), THIRD);
            /**
             * Block Logic: Conditional evaluation for divergent control flow.
             * Invariant: Taken branch maintains control flow invariants.
             */
            if (resolution.unresolved()) {
                return resolution;
            }

            try {
                matchPhraseQueryOptions();
            } catch (InvalidArgumentException e) {
                return new TypeResolution(e.getMessage());
            }
        }
        return TypeResolution.TYPE_RESOLVED;
    }

    private Map<String, Object> matchPhraseQueryOptions() throws InvalidArgumentException {
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (options() == null) {
            return Map.of();
        }

        Map<String, Object> matchPhraseOptions = new HashMap<>();
        populateOptionsMap((MapExpression) options(), matchPhraseOptions, SECOND, sourceText(), ALLOWED_OPTIONS);
        return matchPhraseOptions;
    }

    public Expression field() {
        return field;
    }

    public Expression options() {
        return options;
    }

    @Override
    protected NodeInfo<? extends Expression> info() {
        return NodeInfo.create(this, MatchPhrase::new, field(), query(), options(), queryBuilder());
    }

    @Override
    public Expression replaceChildren(List<Expression> newChildren) {
        return new MatchPhrase(
            source(),
            newChildren.get(0),
            newChildren.get(1),
            newChildren.size() > 2 ? newChildren.get(2) : null,
            queryBuilder()
        );
    }

    @Override
    public Expression replaceQueryBuilder(QueryBuilder queryBuilder) {
        return new MatchPhrase(source(), field, query(), options(), queryBuilder);
    }

    @Override
    public BiConsumer<LogicalPlan, Failures> postAnalysisPlanVerification() {
        return (plan, failures) -> {
            super.postAnalysisPlanVerification().accept(plan, failures);
            plan.forEachExpression(MatchPhrase.class, mp -> {
                /**
                 * Block Logic: Conditional evaluation for divergent control flow.
                 * Invariant: Taken branch maintains control flow invariants.
                 */
                if (mp.fieldAsFieldAttribute() == null) {
                    failures.add(
                        Failure.fail(
                            mp.field(),
                            "[{}] {} cannot operate on [{}], which is not a field from an index mapping",
                            functionName(),
                            functionType(),
                            mp.field().sourceText()
                        )
                    );
                }
            });
        };
    }

    @Override
    public Object queryAsObject() {
        Object queryAsObject = query().fold(FoldContext.small() /* TODO remove me */);

        // Convert BytesRef to string for string-based values
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (queryAsObject instanceof BytesRef bytesRef) {
            return switch (query().dataType()) {
                case IP -> EsqlDataTypeConverter.ipToString(bytesRef);
                case VERSION -> EsqlDataTypeConverter.versionToString(bytesRef);
                default -> bytesRef.utf8ToString();
            };
        }

        // Converts specific types to the correct type for the query
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (query().dataType() == DataType.DATETIME && queryAsObject instanceof Long) {
            // When casting to date and datetime, we get a long back. But MatchPhrase query needs a date string
            return EsqlDataTypeConverter.dateTimeToString((Long) queryAsObject);
        } else if (query().dataType() == DATE_NANOS && queryAsObject instanceof Long) {
            return EsqlDataTypeConverter.nanoTimeToString((Long) queryAsObject);
        }

        return queryAsObject;
    }

    @Override
    protected Query translate(TranslatorHandler handler) {
        var fieldAttribute = fieldAsFieldAttribute();
        Check.notNull(fieldAttribute, "MatchPhrase must have a field attribute as the first argument");
        String fieldName = getNameFromFieldAttribute(fieldAttribute);
        return new MatchPhraseQuery(source(), fieldName, queryAsObject(), matchPhraseQueryOptions());
    }

    public static String getNameFromFieldAttribute(FieldAttribute fieldAttribute) {
        String fieldName = fieldAttribute.name();
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (fieldAttribute.field() instanceof MultiTypeEsField multiTypeEsField) {
            // If we have multiple field types, we allow the query to be done, but getting the underlying field name
            fieldName = multiTypeEsField.getName();
        }
        return fieldName;
    }

    public static FieldAttribute fieldAsFieldAttribute(Expression field) {
        Expression fieldExpression = field;
        // Field may be converted to other data type (field_name :: data_type), so we need to check the original field
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (fieldExpression instanceof AbstractConvertFunction convertFunction) {
            fieldExpression = convertFunction.field();
        }
        return fieldExpression instanceof FieldAttribute fieldAttribute ? fieldAttribute : null;
    }

    private FieldAttribute fieldAsFieldAttribute() {
        return fieldAsFieldAttribute(field);
    }

    @Override
    public boolean equals(Object o) {
        // MatchPhrase does not serialize options, as they get included in the query builder. We need to override equals and hashcode to
        // ignore options when comparing two Match functions
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (o == null || getClass() != o.getClass()) return false;
        MatchPhrase match = (MatchPhrase) o;
        return Objects.equals(field(), match.field())
            && Objects.equals(query(), match.query())
            && Objects.equals(queryBuilder(), match.queryBuilder());
    }

    @Override
    public int hashCode() {
        return Objects.hash(field(), query(), queryBuilder());
    }

}
