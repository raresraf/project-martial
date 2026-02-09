/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.mapper;

import org.apache.lucene.document.DoubleField;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatField;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongField;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.sandbox.document.HalfFloatPoint;
import org.apache.lucene.search.IndexOrDocValuesQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;
import org.elasticsearch.common.Explicit;
import org.elasticsearch.common.Numbers;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.common.settings.Setting.Property;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.IndexMode;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.fielddata.FieldDataContext;
import org.elasticsearch.index.fielddata.IndexFieldData;
import org.elasticsearch.index.fielddata.IndexNumericFieldData.NumericType;
import org.elasticsearch.index.fielddata.SourceValueFetcherSortedDoubleIndexFieldData;
import org.elasticsearch.index.fielddata.SourceValueFetcherSortedNumericIndexFieldData;
import org.elasticsearch.index.fielddata.plain.SortedDoublesIndexFieldData;
import org.elasticsearch.index.fielddata.plain.SortedNumericIndexFieldData;
import org.elasticsearch.index.mapper.TimeSeriesParams.MetricType;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.lucene.document.NumericField;
import org.elasticsearch.lucene.search.XIndexSortSortedNumericDocValuesRangeQuery;
import org.elasticsearch.script.DoubleFieldScript;
import org.elasticsearch.script.LongFieldScript;
import org.elasticsearch.script.Script;
import org.elasticsearch.script.ScriptCompiler;
import org.elasticsearch.script.field.ByteDocValuesField;
import org.elasticsearch.script.field.DoubleDocValuesField;
import org.elasticsearch.script.field.FloatDocValuesField;
import org.elasticsearch.script.field.HalfFloatDocValuesField;
import org.elasticsearch.script.field.IntegerDocValuesField;
import org.elasticsearch.script.field.LongDocValuesField;
import org.elasticsearch.script.field.ShortDocValuesField;
import org.elasticsearch.search.DocValueFormat;
import org.elasticsearch.search.aggregations.support.TimeSeriesValuesSourceType;
import org.elasticsearch.search.aggregations.support.ValuesSourceType;
import org.elasticsearch.search.lookup.FieldValues;
import org.elasticsearch.search.lookup.SearchLookup;
import org.elasticsearch.search.lookup.SourceProvider;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xcontent.XContentParser.Token;

import java.io.IOException;
import java.math.BigDecimal;
import java.time.ZoneId;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;

/** A {@link FieldMapper} for numeric types: byte, short, int, long, float and double. */
/**
 * @brief Functional description of the NumberFieldMapper class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class NumberFieldMapper extends FieldMapper {

    public static final Setting<Boolean> COERCE_SETTING = Setting.boolSetting("index.mapping.coerce", true, Property.IndexScope);

    /**
     * @brief [Functional Utility for toType]: Describe purpose here.
     * @param in: [Description]
     * @return [ReturnType]: [Description]
     */
    private static NumberFieldMapper toType(FieldMapper in) {
        return (NumberFieldMapper) in;
    }

    public static final class Builder extends FieldMapper.DimensionBuilder {

    /**
     * @brief [Functional description for field indexed]: Describe purpose here.
     */
        private final Parameter<Boolean> indexed;
        private final Parameter<Boolean> hasDocValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, true);
        private final Parameter<Boolean> stored = Parameter.storeParam(m -> toType(m).stored, false);

    /**
     * @brief [Functional description for field ignoreMalformed]: Describe purpose here.
     */
        private final Parameter<Explicit<Boolean>> ignoreMalformed;
    /**
     * @brief [Functional description for field coerce]: Describe purpose here.
     */
        private final Parameter<Explicit<Boolean>> coerce;

    /**
     * @brief [Functional description for field nullValue]: Describe purpose here.
     */
        private final Parameter<Number> nullValue;

        private final Parameter<Script> script = Parameter.scriptParam(m -> toType(m).script);
        private final Parameter<OnScriptError> onScriptErrorParam = Parameter.onScriptErrorParam(
            m -> toType(m).builderParams.onScriptError(),
            script
        );

        /**
         * Parameter that marks this field as a time series dimension.
         */
        private final Parameter<Boolean> dimension;

        /**
         * Parameter that marks this field as a time series metric defining its time series metric type.
         * For the numeric fields gauge and counter metric types are
         * supported
         */
        private final Parameter<MetricType> metric;

        private final Parameter<Map<String, String>> meta = Parameter.metaParam();

    /**
     * @brief [Functional description for field scriptCompiler]: Describe purpose here.
     */
        private final ScriptCompiler scriptCompiler;
    /**
     * @brief [Functional description for field type]: Describe purpose here.
     */
        private final NumberType type;

    /**
     * @brief [Functional description for field allowMultipleValues]: Describe purpose here.
     */
        private boolean allowMultipleValues = true;
    /**
     * @brief [Functional description for field indexCreatedVersion]: Describe purpose here.
     */
        private final IndexVersion indexCreatedVersion;

    /**
     * @brief [Functional description for field indexMode]: Describe purpose here.
     */
        private final IndexMode indexMode;

        public Builder(
            String name,
            NumberType type,
            ScriptCompiler compiler,
            Settings settings,
            IndexVersion indexCreatedVersion,
            IndexMode mode
        ) {
            this(name, type, compiler, IGNORE_MALFORMED_SETTING.get(settings), COERCE_SETTING.get(settings), indexCreatedVersion, mode);
        }

    /**
     * @brief [Functional Utility for docValuesOnly]: Describe purpose here.
     * @param name: [Description]
     * @param type: [Description]
     * @param indexCreatedVersion: [Description]
     * @return [ReturnType]: [Description]
     */
        public static Builder docValuesOnly(String name, NumberType type, IndexVersion indexCreatedVersion) {
            Builder builder = new Builder(name, type, ScriptCompiler.NONE, false, false, indexCreatedVersion, null);
            builder.indexed.setValue(false);
            builder.dimension.setValue(false);
    /**
     * @brief [Functional description for field builder]: Describe purpose here.
     */
            return builder;
        }

        public Builder(
            String name,
            NumberType type,
            ScriptCompiler compiler,
            boolean ignoreMalformedByDefault,
            boolean coerceByDefault,
            IndexVersion indexCreatedVersion,
            IndexMode mode
        ) {
            super(name);
            this.type = type;
            this.scriptCompiler = Objects.requireNonNull(compiler);
            this.indexCreatedVersion = Objects.requireNonNull(indexCreatedVersion);

            this.ignoreMalformed = Parameter.explicitBoolParam(
                "ignore_malformed",
                true,
                m -> toType(m).ignoreMalformed,
                ignoreMalformedByDefault
            );
            this.coerce = Parameter.explicitBoolParam("coerce", true, m -> toType(m).coerce, coerceByDefault);
            this.nullValue = new Parameter<>(
                "null_value",
                false,
                () -> null,
                (n, c, o) -> o == null ? null : type.parse(o, false),
                m -> toType(m).nullValue,
                XContentBuilder::field,
                Objects::toString
            ).acceptsNull();
            this.indexMode = mode;
            this.indexed = Parameter.indexParam(m -> toType(m).indexed, () -> {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (indexMode == IndexMode.TIME_SERIES) {
                    var metricType = getMetric().getValue();
                    return metricType != MetricType.COUNTER && metricType != MetricType.GAUGE;
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
    /**
     * @brief [Functional description for field true]: Describe purpose here.
     */
                    return true;
                }
            });
            this.dimension = TimeSeriesParams.dimensionParam(m -> toType(m).dimension).addValidator(v -> {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (v && (indexed.getValue() == false || hasDocValues.getValue() == false)) {
                    throw new IllegalArgumentException(
                        "Field ["
                            + TimeSeriesParams.TIME_SERIES_DIMENSION_PARAM
                            + "] requires that ["
                            + indexed.name
                            + "] and ["
                            + hasDocValues.name
                            + "] are true"
                    );
                }
            });

            this.metric = TimeSeriesParams.metricParam(m -> toType(m).metricType, MetricType.GAUGE, MetricType.COUNTER).addValidator(v -> {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (v != null && hasDocValues.getValue() == false) {
                    throw new IllegalArgumentException(
                        "Field [" + TimeSeriesParams.TIME_SERIES_METRIC_PARAM + "] requires that [" + hasDocValues.name + "] is true"
                    );
                }
            }).precludesParameters(dimension);

            this.script.precludesParameters(ignoreMalformed, coerce, nullValue);
            addScriptValidation(script, indexed, hasDocValues);
        }

    /**
     * @brief [Functional Utility for nullValue]: Describe purpose here.
     * @param number: [Description]
     * @return [ReturnType]: [Description]
     */
        Builder nullValue(Number number) {
            this.nullValue.setValue(number);
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
            return this;
        }

    /**
     * @brief [Functional Utility for docValues]: Describe purpose here.
     * @param hasDocValues: [Description]
     * @return [ReturnType]: [Description]
     */
        public Builder docValues(boolean hasDocValues) {
            this.hasDocValues.setValue(hasDocValues);
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
            return this;
        }

    /**
     * @brief [Functional Utility for scriptValues]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        private FieldValues<Number> scriptValues() {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (this.script.get() == null) {
    /**
     * @brief [Functional description for field null]: Describe purpose here.
     */
                return null;
            }
            return type.compile(leafName(), script.get(), scriptCompiler);
        }

    /**
     * @brief [Functional Utility for dimension]: Describe purpose here.
     * @param dimension: [Description]
     * @return [ReturnType]: [Description]
     */
        public Builder dimension(boolean dimension) {
            this.dimension.setValue(dimension);
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
            return this;
        }

    /**
     * @brief [Functional Utility for metric]: Describe purpose here.
     * @param metric: [Description]
     * @return [ReturnType]: [Description]
     */
        public Builder metric(MetricType metric) {
            this.metric.setValue(metric);
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
            return this;
        }

    /**
     * @brief [Functional Utility for getMetric]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        private Parameter<MetricType> getMetric() {
    /**
     * @brief [Functional description for field metric]: Describe purpose here.
     */
            return metric;
        }

    /**
     * @brief [Functional Utility for allowMultipleValues]: Describe purpose here.
     * @param allowMultipleValues: [Description]
     * @return [ReturnType]: [Description]
     */
        public Builder allowMultipleValues(boolean allowMultipleValues) {
            this.allowMultipleValues = allowMultipleValues;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
            return this;
        }

        @Override
        protected Parameter<?>[] getParameters() {
            return new Parameter<?>[] {
                indexed,
                hasDocValues,
                stored,
                ignoreMalformed,
                coerce,
                nullValue,
                script,
                onScriptErrorParam,
                meta,
                dimension,
                metric };
        }

        @Override
    /**
     * @brief [Functional Utility for build]: Describe purpose here.
     * @param context: [Description]
     * @return [ReturnType]: [Description]
     */
        public NumberFieldMapper build(MapperBuilderContext context) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (inheritDimensionParameterFromParentObject(context)) {
                dimension.setValue(true);
            }

            MappedFieldType ft = new NumberFieldType(context.buildFullName(leafName()), this);
            hasScript = script.get() != null;
            onScriptError = onScriptErrorParam.getValue();
            return new NumberFieldMapper(leafName(), ft, builderParams(this, context), context.isSourceSynthetic(), this);
        }
    }

    public enum NumberType {
        HALF_FLOAT("half_float", NumericType.HALF_FLOAT) {
            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param value: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     */
            public Float parse(Object value, boolean coerce) {
                final float result = parseToFloat(value);
                validateFiniteValue(result);
                // Reduce the precision to what we actually index
                return HalfFloatPoint.sortableShortToHalfFloat(HalfFloatPoint.halfFloatToSortableShort(result));
            }

            @Override
    /**
     * @brief [Functional Utility for reduceToStoredPrecision]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            public double reduceToStoredPrecision(double value) {
                return parse(value, false).doubleValue();
            }

            /**
             * Parse a query parameter or {@code _source} value to a float,
             * keeping float precision. Used by queries which do need to validate
             * against infinite values, but need more precise control over their
             * rounding behavior that {@link #parse(Object, boolean)} provides.
             */
            private static float parseToFloat(Object value) {
    /**
     * @brief [Functional description for field result]: Describe purpose here.
     */
                final float result;

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (value instanceof Number) {
                    result = ((Number) value).floatValue();
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (value instanceof BytesRef) {
                        value = ((BytesRef) value).utf8ToString();
                    }
                    result = Float.parseFloat(value.toString());
                }
    /**
     * @brief [Functional description for field result]: Describe purpose here.
     */
                return result;
            }

            @Override
    /**
     * @brief [Functional Utility for parsePoint]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            public Number parsePoint(byte[] value) {
                return HalfFloatPoint.decodeDimension(value, 0);
            }

            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param parser: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
            public Float parse(XContentParser parser, boolean coerce) throws IOException {
                float parsed = parser.floatValue(coerce);
                validateFiniteValue(parsed);
    /**
     * @brief [Functional description for field parsed]: Describe purpose here.
     */
                return parsed;
            }

            @Override
    /**
     * @brief [Functional Utility for termQuery]: Describe purpose here.
     * @param field: [Description]
     * @param value: [Description]
     * @param isIndexed: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termQuery(String field, Object value, boolean isIndexed) {
                float v = parseToFloat(value);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (Float.isFinite(HalfFloatPoint.sortableShortToHalfFloat(HalfFloatPoint.halfFloatToSortableShort(v))) == false) {
                    return Queries.newMatchNoDocsQuery("Value [" + value + "] is out of range");
                }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isIndexed) {
                    return HalfFloatPoint.newExactQuery(field, v);
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    return SortedNumericDocValuesField.newSlowExactQuery(field, HalfFloatPoint.halfFloatToSortableShort(v));
                }
            }

            @Override
    /**
     * @brief [Functional Utility for termsQuery]: Describe purpose here.
     * @param field: [Description]
     * @param values: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termsQuery(String field, Collection<?> values) {
                float[] v = new float[values.size()];
    /**
     * @brief [Functional description for field pos]: Describe purpose here.
     */
                int pos = 0;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                for (Object value : values) {
                    float float_value = parseToFloat(value);
                    validateFiniteValue(float_value);
                    v[pos++] = float_value;
                }
                return HalfFloatPoint.newSetQuery(field, v);
            }

            @Override
            public Query rangeQuery(
                String field,
                Object lowerTerm,
                Object upperTerm,
                boolean includeLower,
                boolean includeUpper,
                boolean hasDocValues,
                SearchExecutionContext context,
                boolean isIndexed
            ) {
    /**
     * @brief [Functional description for field l]: Describe purpose here.
     */
                float l = Float.NEGATIVE_INFINITY;
    /**
     * @brief [Functional description for field u]: Describe purpose here.
     */
                float u = Float.POSITIVE_INFINITY;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (lowerTerm != null) {
                    l = parseToFloat(lowerTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (includeLower) {
                        l = HalfFloatPoint.nextDown(l);
                    }
                    l = HalfFloatPoint.nextUp(l);
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (upperTerm != null) {
                    u = parseToFloat(upperTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (includeUpper) {
                        u = HalfFloatPoint.nextUp(u);
                    }
                    u = HalfFloatPoint.nextDown(u);
                }
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                Query query;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isIndexed) {
                    query = HalfFloatPoint.newRangeQuery(field, l, u);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (hasDocValues) {
                        Query dvQuery = SortedNumericDocValuesField.newSlowRangeQuery(
                            field,
                            HalfFloatPoint.halfFloatToSortableShort(l),
                            HalfFloatPoint.halfFloatToSortableShort(u)
                        );
                        query = new IndexOrDocValuesQuery(query, dvQuery);
                    }
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    query = SortedNumericDocValuesField.newSlowRangeQuery(
                        field,
                        HalfFloatPoint.halfFloatToSortableShort(l),
                        HalfFloatPoint.halfFloatToSortableShort(u)
                    );
                }
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                return query;
            }

            @Override
    /**
     * @brief [Functional Utility for addFields]: Describe purpose here.
     * @param document: [Description]
     * @param name: [Description]
     * @param value: [Description]
     * @param indexed: [Description]
     * @param docValued: [Description]
     * @param stored: [Description]
     * @return [ReturnType]: [Description]
     */
            public void addFields(LuceneDocument document, String name, Number value, boolean indexed, boolean docValued, boolean stored) {
                final float f = value.floatValue();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (indexed) {
                    document.add(new HalfFloatPoint(name, f));
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (docValued) {
                    document.add(new SortedNumericDocValuesField(name, HalfFloatPoint.halfFloatToSortableShort(f)));
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (stored) {
                    document.add(new StoredField(name, f));
                }
            }

            @Override
    /**
     * @brief [Functional Utility for getFieldDataBuilder]: Describe purpose here.
     * @param ft: [Description]
     * @param valuesSourceType: [Description]
     * @return [ReturnType]: [Description]
     */
            public IndexFieldData.Builder getFieldDataBuilder(MappedFieldType ft, ValuesSourceType valuesSourceType) {
                return new SortedDoublesIndexFieldData.Builder(
                    ft.name(),
                    numericType(),
                    valuesSourceType,
                    HalfFloatDocValuesField::new,
                    ft.isIndexed()
                );
            }

            @Override
            public IndexFieldData.Builder getValueFetcherFieldDataBuilder(
                String name,
                ValuesSourceType valuesSourceType,
                SourceProvider sourceProvider,
                ValueFetcher valueFetcher
            ) {
                return new SourceValueFetcherSortedDoubleIndexFieldData.Builder(
                    name,
                    valuesSourceType,
                    valueFetcher,
                    sourceProvider,
                    HalfFloatDocValuesField::new
                );
            }

    /**
     * @brief [Functional Utility for validateFiniteValue]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            private static void validateFiniteValue(float value) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (Float.isFinite(HalfFloatPoint.sortableShortToHalfFloat(HalfFloatPoint.halfFloatToSortableShort(value))) == false) {
                    throw new IllegalArgumentException("[half_float] supports only finite values, but got [" + value + "]");
                }
            }

            @Override
    /**
     * @brief [Functional Utility for syntheticFieldLoader]: Describe purpose here.
     * @param fieldName: [Description]
     * @param fieldSimpleName: [Description]
     * @param ignoreMalformed: [Description]
     * @return [ReturnType]: [Description]
     */
            SourceLoader.SyntheticFieldLoader syntheticFieldLoader(String fieldName, String fieldSimpleName, boolean ignoreMalformed) {
                return new SortedNumericDocValuesSyntheticFieldLoader(fieldName, fieldSimpleName, ignoreMalformed) {
                    @Override
    /**
     * @brief [Functional Utility for writeValue]: Describe purpose here.
     * @param b: [Description]
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
                    protected void writeValue(XContentBuilder b, long value) throws IOException {
                        b.value(HalfFloatPoint.sortableShortToHalfFloat((short) value));
                    }
                };
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromDocValues]: Describe purpose here.
     * @param fieldName: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromDocValues(String fieldName) {
                return new BlockDocValuesReader.DoublesBlockLoader(fieldName, l -> HalfFloatPoint.sortableShortToHalfFloat((short) l));
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromSource]: Describe purpose here.
     * @param sourceValueFetcher: [Description]
     * @param lookup: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromSource(SourceValueFetcher sourceValueFetcher, BlockSourceReader.LeafIteratorLookup lookup) {
                return new BlockSourceReader.DoublesBlockLoader(sourceValueFetcher, lookup);
            }
        },
        FLOAT("float", NumericType.FLOAT) {
            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param value: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     */
            public Float parse(Object value, boolean coerce) {
                final float result = parseToFloat(value);
                validateFiniteValue(result);
    /**
     * @brief [Functional description for field result]: Describe purpose here.
     */
                return result;
            }

            /**
             * Parse a query parameter or {@code _source} value to a float,
             * keeping float precision. Used by queries which do need validate
             * against infinite values like {@link #parse(Object, boolean)} does.
             */
            private static float parseToFloat(Object value) {
    /**
     * @brief [Functional description for field result]: Describe purpose here.
     */
                final float result;

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (value instanceof Number) {
                    result = ((Number) value).floatValue();
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (value instanceof BytesRef) {
                        value = ((BytesRef) value).utf8ToString();
                    }
                    result = Float.parseFloat(value.toString());
                }
    /**
     * @brief [Functional description for field result]: Describe purpose here.
     */
                return result;
            }

            @Override
    /**
     * @brief [Functional Utility for reduceToStoredPrecision]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            public double reduceToStoredPrecision(double value) {
                return parse(value, false).doubleValue();
            }

            @Override
    /**
     * @brief [Functional Utility for parsePoint]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            public Number parsePoint(byte[] value) {
                return FloatPoint.decodeDimension(value, 0);
            }

            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param parser: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
            public Float parse(XContentParser parser, boolean coerce) throws IOException {
                float parsed = parser.floatValue(coerce);
                validateFiniteValue(parsed);
    /**
     * @brief [Functional description for field parsed]: Describe purpose here.
     */
                return parsed;
            }

            @Override
    /**
     * @brief [Functional Utility for termQuery]: Describe purpose here.
     * @param field: [Description]
     * @param value: [Description]
     * @param isIndexed: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termQuery(String field, Object value, boolean isIndexed) {
                float v = parseToFloat(value);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (Float.isFinite(v) == false) {
                    return new MatchNoDocsQuery("Value [" + value + "] is out of range");
                }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isIndexed) {
                    return FloatPoint.newExactQuery(field, v);
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    return SortedNumericDocValuesField.newSlowExactQuery(field, NumericUtils.floatToSortableInt(v));
                }
            }

            @Override
    /**
     * @brief [Functional Utility for termsQuery]: Describe purpose here.
     * @param field: [Description]
     * @param values: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termsQuery(String field, Collection<?> values) {
                float[] v = new float[values.size()];
    /**
     * @brief [Functional description for field pos]: Describe purpose here.
     */
                int pos = 0;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                for (Object value : values) {
                    v[pos++] = parse(value, false);
                }
                return FloatPoint.newSetQuery(field, v);
            }

            @Override
            public Query rangeQuery(
                String field,
                Object lowerTerm,
                Object upperTerm,
                boolean includeLower,
                boolean includeUpper,
                boolean hasDocValues,
                SearchExecutionContext context,
                boolean isIndexed
            ) {
    /**
     * @brief [Functional description for field l]: Describe purpose here.
     */
                float l = Float.NEGATIVE_INFINITY;
    /**
     * @brief [Functional description for field u]: Describe purpose here.
     */
                float u = Float.POSITIVE_INFINITY;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (lowerTerm != null) {
                    l = parseToFloat(lowerTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (includeLower) {
                        l = FloatPoint.nextDown(l);
                    }
                    l = FloatPoint.nextUp(l);
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (upperTerm != null) {
                    u = parseToFloat(upperTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (includeUpper) {
                        u = FloatPoint.nextUp(u);
                    }
                    u = FloatPoint.nextDown(u);
                }
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                Query query;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isIndexed) {
                    query = FloatPoint.newRangeQuery(field, l, u);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (hasDocValues) {
                        Query dvQuery = SortedNumericDocValuesField.newSlowRangeQuery(
                            field,
                            NumericUtils.floatToSortableInt(l),
                            NumericUtils.floatToSortableInt(u)
                        );
                        query = new IndexOrDocValuesQuery(query, dvQuery);
                    }
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    query = SortedNumericDocValuesField.newSlowRangeQuery(
                        field,
                        NumericUtils.floatToSortableInt(l),
                        NumericUtils.floatToSortableInt(u)
                    );
                }
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                return query;
            }

            @Override
    /**
     * @brief [Functional Utility for addFields]: Describe purpose here.
     * @param document: [Description]
     * @param name: [Description]
     * @param value: [Description]
     * @param indexed: [Description]
     * @param docValued: [Description]
     * @param stored: [Description]
     * @return [ReturnType]: [Description]
     */
            public void addFields(LuceneDocument document, String name, Number value, boolean indexed, boolean docValued, boolean stored) {
                final float f = value.floatValue();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (indexed && docValued) {
                    document.add(new FloatField(name, f, Field.Store.NO));
        // Block Logic: [Describe purpose of this else/else if block]
                } else if (docValued) {
                    document.add(new SortedNumericDocValuesField(name, NumericUtils.floatToSortableInt(f)));
        // Block Logic: [Describe purpose of this else/else if block]
                } else if (indexed) {
                    document.add(new FloatPoint(name, f));
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (stored) {
                    document.add(new StoredField(name, f));
                }
            }

            @Override
    /**
     * @brief [Functional Utility for getFieldDataBuilder]: Describe purpose here.
     * @param ft: [Description]
     * @param valuesSourceType: [Description]
     * @return [ReturnType]: [Description]
     */
            public IndexFieldData.Builder getFieldDataBuilder(MappedFieldType ft, ValuesSourceType valuesSourceType) {
                return new SortedDoublesIndexFieldData.Builder(
                    ft.name(),
                    numericType(),
                    valuesSourceType,
                    FloatDocValuesField::new,
                    ft.isIndexed()
                );
            }

            @Override
            public IndexFieldData.Builder getValueFetcherFieldDataBuilder(
                String name,
                ValuesSourceType valuesSourceType,
                SourceProvider sourceProvider,
                ValueFetcher valueFetcher
            ) {
                return new SourceValueFetcherSortedDoubleIndexFieldData.Builder(
                    name,
                    valuesSourceType,
                    valueFetcher,
                    sourceProvider,
                    FloatDocValuesField::new
                );
            }

    /**
     * @brief [Functional Utility for validateFiniteValue]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            private static void validateFiniteValue(float value) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (Float.isFinite(value) == false) {
                    throw new IllegalArgumentException("[float] supports only finite values, but got [" + value + "]");
                }
            }

            @Override
    /**
     * @brief [Functional Utility for syntheticFieldLoader]: Describe purpose here.
     * @param fieldName: [Description]
     * @param fieldSimpleName: [Description]
     * @param ignoreMalformed: [Description]
     * @return [ReturnType]: [Description]
     */
            SourceLoader.SyntheticFieldLoader syntheticFieldLoader(String fieldName, String fieldSimpleName, boolean ignoreMalformed) {
                return new SortedNumericDocValuesSyntheticFieldLoader(fieldName, fieldSimpleName, ignoreMalformed) {
                    @Override
    /**
     * @brief [Functional Utility for writeValue]: Describe purpose here.
     * @param b: [Description]
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
                    protected void writeValue(XContentBuilder b, long value) throws IOException {
                        b.value(NumericUtils.sortableIntToFloat((int) value));
                    }
                };
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromDocValues]: Describe purpose here.
     * @param fieldName: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromDocValues(String fieldName) {
                return new BlockDocValuesReader.DoublesBlockLoader(fieldName, l -> NumericUtils.sortableIntToFloat((int) l));
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromSource]: Describe purpose here.
     * @param sourceValueFetcher: [Description]
     * @param lookup: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromSource(SourceValueFetcher sourceValueFetcher, BlockSourceReader.LeafIteratorLookup lookup) {
                return new BlockSourceReader.DoublesBlockLoader(sourceValueFetcher, lookup);
            }
        },
        DOUBLE("double", NumericType.DOUBLE) {
            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param value: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     */
            public Double parse(Object value, boolean coerce) {
                double parsed = objectToDouble(value);
                validateParsed(parsed);
    /**
     * @brief [Functional description for field parsed]: Describe purpose here.
     */
                return parsed;
            }

            @Override
    /**
     * @brief [Functional Utility for parsePoint]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            public Number parsePoint(byte[] value) {
                return DoublePoint.decodeDimension(value, 0);
            }

            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param parser: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
            public Double parse(XContentParser parser, boolean coerce) throws IOException {
                double parsed = parser.doubleValue(coerce);
                validateParsed(parsed);
    /**
     * @brief [Functional description for field parsed]: Describe purpose here.
     */
                return parsed;
            }

            @Override
    /**
     * @brief [Functional Utility for compile]: Describe purpose here.
     * @param fieldName: [Description]
     * @param script: [Description]
     * @param compiler: [Description]
     * @return [ReturnType]: [Description]
     */
            public FieldValues<Number> compile(String fieldName, Script script, ScriptCompiler compiler) {
                DoubleFieldScript.Factory scriptFactory = compiler.compile(script, DoubleFieldScript.CONTEXT);
                return (lookup, ctx, doc, consumer) -> scriptFactory.newFactory(fieldName, script.getParams(), lookup, OnScriptError.FAIL)
                    .newInstance(ctx)
                    .runForDoc(doc, consumer::accept);
            }

            @Override
    /**
     * @brief [Functional Utility for termQuery]: Describe purpose here.
     * @param field: [Description]
     * @param value: [Description]
     * @param isIndexed: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termQuery(String field, Object value, boolean isIndexed) {
                double v = objectToDouble(value);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (Double.isFinite(v) == false) {
                    return Queries.newMatchNoDocsQuery("Value [" + value + "] has a decimal part");
                }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isIndexed) {
                    return DoublePoint.newExactQuery(field, v);
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    return SortedNumericDocValuesField.newSlowExactQuery(field, NumericUtils.doubleToSortableLong(v));
                }
            }

            @Override
    /**
     * @brief [Functional Utility for termsQuery]: Describe purpose here.
     * @param field: [Description]
     * @param values: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termsQuery(String field, Collection<?> values) {
                double[] v = values.stream().mapToDouble(value -> parse(value, false)).toArray();
                return DoublePoint.newSetQuery(field, v);
            }

            @Override
            public Query rangeQuery(
                String field,
                Object lowerTerm,
                Object upperTerm,
                boolean includeLower,
                boolean includeUpper,
                boolean hasDocValues,
                SearchExecutionContext context,
                boolean isIndexed
            ) {
                return doubleRangeQuery(lowerTerm, upperTerm, includeLower, includeUpper, (l, u) -> {
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                    Query query;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (isIndexed) {
                        query = DoublePoint.newRangeQuery(field, l, u);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                        if (hasDocValues) {
                            Query dvQuery = SortedNumericDocValuesField.newSlowRangeQuery(
                                field,
                                NumericUtils.doubleToSortableLong(l),
                                NumericUtils.doubleToSortableLong(u)
                            );
                            query = new IndexOrDocValuesQuery(query, dvQuery);
                        }
        // Block Logic: [Describe purpose of this else/else if block]
                    } else {
                        query = SortedNumericDocValuesField.newSlowRangeQuery(
                            field,
                            NumericUtils.doubleToSortableLong(l),
                            NumericUtils.doubleToSortableLong(u)
                        );
                    }
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                    return query;
                });
            }

            @Override
    /**
     * @brief [Functional Utility for addFields]: Describe purpose here.
     * @param document: [Description]
     * @param name: [Description]
     * @param value: [Description]
     * @param indexed: [Description]
     * @param docValued: [Description]
     * @param stored: [Description]
     * @return [ReturnType]: [Description]
     */
            public void addFields(LuceneDocument document, String name, Number value, boolean indexed, boolean docValued, boolean stored) {
                final double d = value.doubleValue();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (indexed && docValued) {
                    document.add(new DoubleField(name, d, Field.Store.NO));
        // Block Logic: [Describe purpose of this else/else if block]
                } else if (docValued) {
                    document.add(new SortedNumericDocValuesField(name, NumericUtils.doubleToSortableLong(d)));
        // Block Logic: [Describe purpose of this else/else if block]
                } else if (indexed) {
                    document.add(new DoublePoint(name, d));
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (stored) {
                    document.add(new StoredField(name, d));
                }
            }

            @Override
    /**
     * @brief [Functional Utility for getFieldDataBuilder]: Describe purpose here.
     * @param ft: [Description]
     * @param valuesSourceType: [Description]
     * @return [ReturnType]: [Description]
     */
            public IndexFieldData.Builder getFieldDataBuilder(MappedFieldType ft, ValuesSourceType valuesSourceType) {
                return new SortedDoublesIndexFieldData.Builder(
                    ft.name(),
                    numericType(),
                    valuesSourceType,
                    DoubleDocValuesField::new,
                    ft.isIndexed()
                );
            }

            @Override
            public IndexFieldData.Builder getValueFetcherFieldDataBuilder(
                String name,
                ValuesSourceType valuesSourceType,
                SourceProvider sourceProvider,
                ValueFetcher valueFetcher
            ) {
                return new SourceValueFetcherSortedDoubleIndexFieldData.Builder(
                    name,
                    valuesSourceType,
                    valueFetcher,
                    sourceProvider,
                    DoubleDocValuesField::new
                );
            }

    /**
     * @brief [Functional Utility for validateParsed]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            private static void validateParsed(double value) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (Double.isFinite(value) == false) {
                    throw new IllegalArgumentException("[double] supports only finite values, but got [" + value + "]");
                }
            }

            @Override
    /**
     * @brief [Functional Utility for syntheticFieldLoader]: Describe purpose here.
     * @param fieldName: [Description]
     * @param fieldSimpleName: [Description]
     * @param ignoreMalformed: [Description]
     * @return [ReturnType]: [Description]
     */
            SourceLoader.SyntheticFieldLoader syntheticFieldLoader(String fieldName, String fieldSimpleName, boolean ignoreMalformed) {
                return new SortedNumericDocValuesSyntheticFieldLoader(fieldName, fieldSimpleName, ignoreMalformed) {
                    @Override
    /**
     * @brief [Functional Utility for writeValue]: Describe purpose here.
     * @param b: [Description]
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
                    protected void writeValue(XContentBuilder b, long value) throws IOException {
                        b.value(NumericUtils.sortableLongToDouble(value));
                    }
                };
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromDocValues]: Describe purpose here.
     * @param fieldName: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromDocValues(String fieldName) {
                return new BlockDocValuesReader.DoublesBlockLoader(fieldName, NumericUtils::sortableLongToDouble);
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromSource]: Describe purpose here.
     * @param sourceValueFetcher: [Description]
     * @param lookup: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromSource(SourceValueFetcher sourceValueFetcher, BlockSourceReader.LeafIteratorLookup lookup) {
                return new BlockSourceReader.DoublesBlockLoader(sourceValueFetcher, lookup);
            }
        },
        BYTE("byte", NumericType.BYTE) {
            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param value: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     */
            public Byte parse(Object value, boolean coerce) {
                double doubleValue = objectToDouble(value);

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (doubleValue < Byte.MIN_VALUE || doubleValue > Byte.MAX_VALUE) {
                    throw new IllegalArgumentException("Value [" + value + "] is out of range for a byte");
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (coerce == false && doubleValue % 1 != 0) {
                    throw new IllegalArgumentException("Value [" + value + "] has a decimal part");
                }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (value instanceof Number) {
                    return ((Number) value).byteValue();
                }

                return (byte) doubleValue;
            }

            @Override
    /**
     * @brief [Functional Utility for parsePoint]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            public Number parsePoint(byte[] value) {
                return INTEGER.parsePoint(value).byteValue();
            }

            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param parser: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
            public Short parse(XContentParser parser, boolean coerce) throws IOException {
                int value = parser.intValue(coerce);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (value < Byte.MIN_VALUE || value > Byte.MAX_VALUE) {
                    throw new IllegalArgumentException("Value [" + value + "] is out of range for a byte");
                }
                return (short) value;
            }

            @Override
    /**
     * @brief [Functional Utility for termQuery]: Describe purpose here.
     * @param field: [Description]
     * @param value: [Description]
     * @param isIndexed: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termQuery(String field, Object value, boolean isIndexed) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isOutOfRange(value)) {
                    return new MatchNoDocsQuery("Value [" + value + "] is out of range");
                }

                return INTEGER.termQuery(field, value, isIndexed);
            }

            @Override
    /**
     * @brief [Functional Utility for termsQuery]: Describe purpose here.
     * @param field: [Description]
     * @param values: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termsQuery(String field, Collection<?> values) {
                return INTEGER.termsQuery(field, values);
            }

            @Override
            public Query rangeQuery(
                String field,
                Object lowerTerm,
                Object upperTerm,
                boolean includeLower,
                boolean includeUpper,
                boolean hasDocValues,
                SearchExecutionContext context,
                boolean isIndexed
            ) {
                return INTEGER.rangeQuery(field, lowerTerm, upperTerm, includeLower, includeUpper, hasDocValues, context, isIndexed);
            }

            @Override
    /**
     * @brief [Functional Utility for addFields]: Describe purpose here.
     * @param document: [Description]
     * @param name: [Description]
     * @param value: [Description]
     * @param indexed: [Description]
     * @param docValued: [Description]
     * @param stored: [Description]
     * @return [ReturnType]: [Description]
     */
            public void addFields(LuceneDocument document, String name, Number value, boolean indexed, boolean docValued, boolean stored) {
                INTEGER.addFields(document, name, value, indexed, docValued, stored);
            }

            @Override
    /**
     * @brief [Functional Utility for valueForSearch]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            Number valueForSearch(Number value) {
                return value.byteValue();
            }

            @Override
    /**
     * @brief [Functional Utility for getFieldDataBuilder]: Describe purpose here.
     * @param ft: [Description]
     * @param valuesSourceType: [Description]
     * @return [ReturnType]: [Description]
     */
            public IndexFieldData.Builder getFieldDataBuilder(MappedFieldType ft, ValuesSourceType valuesSourceType) {
                return new SortedNumericIndexFieldData.Builder(
                    ft.name(),
                    numericType(),
                    valuesSourceType,
                    ByteDocValuesField::new,
                    ft.isIndexed()
                );
            }

            @Override
            public IndexFieldData.Builder getValueFetcherFieldDataBuilder(
                String name,
                ValuesSourceType valuesSourceType,
                SourceProvider sourceProvider,
                ValueFetcher valueFetcher
            ) {
                return new SourceValueFetcherSortedNumericIndexFieldData.Builder(
                    name,
                    valuesSourceType,
                    valueFetcher,
                    sourceProvider,
                    ByteDocValuesField::new
                );
            }

            @Override
    /**
     * @brief [Functional Utility for syntheticFieldLoader]: Describe purpose here.
     * @param fieldName: [Description]
     * @param fieldSimpleName: [Description]
     * @param ignoreMalformed: [Description]
     * @return [ReturnType]: [Description]
     */
            SourceLoader.SyntheticFieldLoader syntheticFieldLoader(String fieldName, String fieldSimpleName, boolean ignoreMalformed) {
                return NumberType.syntheticLongFieldLoader(fieldName, fieldSimpleName, ignoreMalformed);
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromDocValues]: Describe purpose here.
     * @param fieldName: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromDocValues(String fieldName) {
                return new BlockDocValuesReader.IntsBlockLoader(fieldName);
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromSource]: Describe purpose here.
     * @param sourceValueFetcher: [Description]
     * @param lookup: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromSource(SourceValueFetcher sourceValueFetcher, BlockSourceReader.LeafIteratorLookup lookup) {
                return new BlockSourceReader.IntsBlockLoader(sourceValueFetcher, lookup);
            }

    /**
     * @brief [Functional Utility for isOutOfRange]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            private boolean isOutOfRange(Object value) {
                double doubleValue = objectToDouble(value);
                return doubleValue < Byte.MIN_VALUE || doubleValue > Byte.MAX_VALUE;
            }
        },
        SHORT("short", NumericType.SHORT) {
            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param value: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     */
            public Short parse(Object value, boolean coerce) {
                double doubleValue = objectToDouble(value);

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (doubleValue < Short.MIN_VALUE || doubleValue > Short.MAX_VALUE) {
                    throw new IllegalArgumentException("Value [" + value + "] is out of range for a short");
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (coerce == false && doubleValue % 1 != 0) {
                    throw new IllegalArgumentException("Value [" + value + "] has a decimal part");
                }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (value instanceof Number) {
                    return ((Number) value).shortValue();
                }

                return (short) doubleValue;
            }

            @Override
    /**
     * @brief [Functional Utility for parsePoint]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            public Number parsePoint(byte[] value) {
                return INTEGER.parsePoint(value).shortValue();
            }

            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param parser: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
            public Short parse(XContentParser parser, boolean coerce) throws IOException {
                return parser.shortValue(coerce);
            }

            @Override
    /**
     * @brief [Functional Utility for termQuery]: Describe purpose here.
     * @param field: [Description]
     * @param value: [Description]
     * @param isIndexed: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termQuery(String field, Object value, boolean isIndexed) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isOutOfRange(value)) {
                    return Queries.newMatchNoDocsQuery("Value [" + value + "] is out of range");
                }
                return INTEGER.termQuery(field, value, isIndexed);
            }

            @Override
    /**
     * @brief [Functional Utility for termsQuery]: Describe purpose here.
     * @param field: [Description]
     * @param values: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termsQuery(String field, Collection<?> values) {
                return INTEGER.termsQuery(field, values);
            }

            @Override
            public Query rangeQuery(
                String field,
                Object lowerTerm,
                Object upperTerm,
                boolean includeLower,
                boolean includeUpper,
                boolean hasDocValues,
                SearchExecutionContext context,
                boolean isIndexed
            ) {
                return INTEGER.rangeQuery(field, lowerTerm, upperTerm, includeLower, includeUpper, hasDocValues, context, isIndexed);
            }

            @Override
    /**
     * @brief [Functional Utility for addFields]: Describe purpose here.
     * @param document: [Description]
     * @param name: [Description]
     * @param value: [Description]
     * @param indexed: [Description]
     * @param docValued: [Description]
     * @param stored: [Description]
     * @return [ReturnType]: [Description]
     */
            public void addFields(LuceneDocument document, String name, Number value, boolean indexed, boolean docValued, boolean stored) {
                INTEGER.addFields(document, name, value, indexed, docValued, stored);
            }

            @Override
    /**
     * @brief [Functional Utility for valueForSearch]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            Number valueForSearch(Number value) {
                return value.shortValue();
            }

            @Override
    /**
     * @brief [Functional Utility for getFieldDataBuilder]: Describe purpose here.
     * @param ft: [Description]
     * @param valuesSourceType: [Description]
     * @return [ReturnType]: [Description]
     */
            public IndexFieldData.Builder getFieldDataBuilder(MappedFieldType ft, ValuesSourceType valuesSourceType) {
                return new SortedNumericIndexFieldData.Builder(
                    ft.name(),
                    numericType(),
                    valuesSourceType,
                    ShortDocValuesField::new,
                    ft.isIndexed()
                );
            }

            @Override
            public IndexFieldData.Builder getValueFetcherFieldDataBuilder(
                String name,
                ValuesSourceType valuesSourceType,
                SourceProvider sourceProvider,
                ValueFetcher valueFetcher
            ) {
                return new SourceValueFetcherSortedNumericIndexFieldData.Builder(
                    name,
                    valuesSourceType,
                    valueFetcher,
                    sourceProvider,
                    ShortDocValuesField::new
                );
            }

            @Override
    /**
     * @brief [Functional Utility for syntheticFieldLoader]: Describe purpose here.
     * @param fieldName: [Description]
     * @param fieldSimpleName: [Description]
     * @param ignoreMalformed: [Description]
     * @return [ReturnType]: [Description]
     */
            SourceLoader.SyntheticFieldLoader syntheticFieldLoader(String fieldName, String fieldSimpleName, boolean ignoreMalformed) {
                return NumberType.syntheticLongFieldLoader(fieldName, fieldSimpleName, ignoreMalformed);
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromDocValues]: Describe purpose here.
     * @param fieldName: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromDocValues(String fieldName) {
                return new BlockDocValuesReader.IntsBlockLoader(fieldName);
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromSource]: Describe purpose here.
     * @param sourceValueFetcher: [Description]
     * @param lookup: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromSource(SourceValueFetcher sourceValueFetcher, BlockSourceReader.LeafIteratorLookup lookup) {
                return new BlockSourceReader.IntsBlockLoader(sourceValueFetcher, lookup);
            }

    /**
     * @brief [Functional Utility for isOutOfRange]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            private boolean isOutOfRange(Object value) {
                double doubleValue = objectToDouble(value);
                return doubleValue < Short.MIN_VALUE || doubleValue > Short.MAX_VALUE;
            }
        },
        INTEGER("integer", NumericType.INT) {
            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param value: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     */
            public Integer parse(Object value, boolean coerce) {
                double doubleValue = objectToDouble(value);

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isOutOfRange(doubleValue)) {
                    throw new IllegalArgumentException("Value [" + value + "] is out of range for an integer");
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (coerce == false && doubleValue % 1 != 0) {
                    throw new IllegalArgumentException("Value [" + value + "] has a decimal part");
                }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (value instanceof Number) {
                    return ((Number) value).intValue();
                }
                return (int) doubleValue;
            }

    /**
     * @brief [Functional Utility for isOutOfRange]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            private boolean isOutOfRange(double value) {
                return value < Integer.MIN_VALUE || value > Integer.MAX_VALUE;
            }

            @Override
    /**
     * @brief [Functional Utility for parsePoint]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            public Number parsePoint(byte[] value) {
                return IntPoint.decodeDimension(value, 0);
            }

            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param parser: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
            public Integer parse(XContentParser parser, boolean coerce) throws IOException {
                return parser.intValue(coerce);
            }

            @Override
    /**
     * @brief [Functional Utility for termQuery]: Describe purpose here.
     * @param field: [Description]
     * @param value: [Description]
     * @param isIndexed: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termQuery(String field, Object value, boolean isIndexed) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (hasDecimalPart(value)) {
                    return Queries.newMatchNoDocsQuery("Value [" + value + "] has a decimal part");
                }
                double doubleValue = objectToDouble(value);

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isOutOfRange(doubleValue)) {
                    return Queries.newMatchNoDocsQuery("Value [" + value + "] is out of range");
                }
                int v = parse(value, true);

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isIndexed) {
                    return IntPoint.newExactQuery(field, v);
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    return SortedNumericDocValuesField.newSlowExactQuery(field, v);
                }
            }

            @Override
    /**
     * @brief [Functional Utility for termsQuery]: Describe purpose here.
     * @param field: [Description]
     * @param values: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termsQuery(String field, Collection<?> values) {
                int[] v = new int[values.size()];
    /**
     * @brief [Functional description for field upTo]: Describe purpose here.
     */
                int upTo = 0;

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                for (Object value : values) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (hasDecimalPart(value) == false) {
                        v[upTo++] = parse(value, true);
                    }
                }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (upTo == 0) {
                    return Queries.newMatchNoDocsQuery("All values have a decimal part");
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (upTo != v.length) {
                    v = Arrays.copyOf(v, upTo);
                }
                return IntPoint.newSetQuery(field, v);
            }

            @Override
            public Query rangeQuery(
                String field,
                Object lowerTerm,
                Object upperTerm,
                boolean includeLower,
                boolean includeUpper,
                boolean hasDocValues,
                SearchExecutionContext context,
                boolean isIndexed
            ) {
    /**
     * @brief [Functional description for field l]: Describe purpose here.
     */
                int l = Integer.MIN_VALUE;
    /**
     * @brief [Functional description for field u]: Describe purpose here.
     */
                int u = Integer.MAX_VALUE;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (lowerTerm != null) {
                    l = parse(lowerTerm, true);
                    // if the lower bound is decimal:
                    // - if the bound is positive then we increment it:
                    // if lowerTerm=1.5 then the (inclusive) bound becomes 2
                    // - if the bound is negative then we leave it as is:
                    // if lowerTerm=-1.5 then the (inclusive) bound becomes -1 due to the call to longValue
                    boolean lowerTermHasDecimalPart = hasDecimalPart(lowerTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if ((lowerTermHasDecimalPart == false && includeLower == false) || (lowerTermHasDecimalPart && signum(lowerTerm) > 0)) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                        if (l == Integer.MAX_VALUE) {
                            return new MatchNoDocsQuery();
                        }
                        ++l;
                    }
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (upperTerm != null) {
                    u = parse(upperTerm, true);
                    boolean upperTermHasDecimalPart = hasDecimalPart(upperTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if ((upperTermHasDecimalPart == false && includeUpper == false) || (upperTermHasDecimalPart && signum(upperTerm) < 0)) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                        if (u == Integer.MIN_VALUE) {
                            return new MatchNoDocsQuery();
                        }
                        --u;
                    }
                }
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                Query query;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isIndexed) {
                    query = IntPoint.newRangeQuery(field, l, u);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (hasDocValues) {
                        Query dvQuery = SortedNumericDocValuesField.newSlowRangeQuery(field, l, u);
                        query = new IndexOrDocValuesQuery(query, dvQuery);
                    }
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    query = SortedNumericDocValuesField.newSlowRangeQuery(field, l, u);
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (hasDocValues && context.indexSortedOnField(field)) {
                    query = new XIndexSortSortedNumericDocValuesRangeQuery(field, l, u, query);
                }
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                return query;
            }

            @Override
    /**
     * @brief [Functional Utility for addFields]: Describe purpose here.
     * @param document: [Description]
     * @param name: [Description]
     * @param value: [Description]
     * @param indexed: [Description]
     * @param docValued: [Description]
     * @param stored: [Description]
     * @return [ReturnType]: [Description]
     */
            public void addFields(LuceneDocument document, String name, Number value, boolean indexed, boolean docValued, boolean stored) {
                final int i = value.intValue();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (indexed && docValued) {
                    document.add(new IntField(name, i, Field.Store.NO));
        // Block Logic: [Describe purpose of this else/else if block]
                } else if (docValued) {
                    document.add(new SortedNumericDocValuesField(name, i));
        // Block Logic: [Describe purpose of this else/else if block]
                } else if (indexed) {
                    document.add(new IntPoint(name, i));
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (stored) {
                    document.add(new StoredField(name, i));
                }
            }

            @Override
    /**
     * @brief [Functional Utility for getFieldDataBuilder]: Describe purpose here.
     * @param ft: [Description]
     * @param valuesSourceType: [Description]
     * @return [ReturnType]: [Description]
     */
            public IndexFieldData.Builder getFieldDataBuilder(MappedFieldType ft, ValuesSourceType valuesSourceType) {
                return new SortedNumericIndexFieldData.Builder(
                    ft.name(),
                    numericType(),
                    valuesSourceType,
                    IntegerDocValuesField::new,
                    ft.isIndexed()
                );
            }

            @Override
            public IndexFieldData.Builder getValueFetcherFieldDataBuilder(
                String name,
                ValuesSourceType valuesSourceType,
                SourceProvider sourceProvider,
                ValueFetcher valueFetcher
            ) {
                return new SourceValueFetcherSortedNumericIndexFieldData.Builder(
                    name,
                    valuesSourceType,
                    valueFetcher,
                    sourceProvider,
                    IntegerDocValuesField::new
                );
            }

            @Override
    /**
     * @brief [Functional Utility for syntheticFieldLoader]: Describe purpose here.
     * @param fieldName: [Description]
     * @param fieldSimpleName: [Description]
     * @param ignoreMalformed: [Description]
     * @return [ReturnType]: [Description]
     */
            SourceLoader.SyntheticFieldLoader syntheticFieldLoader(String fieldName, String fieldSimpleName, boolean ignoreMalformed) {
                return NumberType.syntheticLongFieldLoader(fieldName, fieldSimpleName, ignoreMalformed);
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromDocValues]: Describe purpose here.
     * @param fieldName: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromDocValues(String fieldName) {
                return new BlockDocValuesReader.IntsBlockLoader(fieldName);
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromSource]: Describe purpose here.
     * @param sourceValueFetcher: [Description]
     * @param lookup: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromSource(SourceValueFetcher sourceValueFetcher, BlockSourceReader.LeafIteratorLookup lookup) {
                return new BlockSourceReader.IntsBlockLoader(sourceValueFetcher, lookup);
            }
        },
        LONG("long", NumericType.LONG) {
            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param value: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     */
            public Long parse(Object value, boolean coerce) {
                return objectToLong(value, coerce);
            }

            @Override
    /**
     * @brief [Functional Utility for parsePoint]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            public Number parsePoint(byte[] value) {
                return LongPoint.decodeDimension(value, 0);
            }

            @Override
    /**
     * @brief [Functional Utility for parse]: Describe purpose here.
     * @param parser: [Description]
     * @param coerce: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
            public Long parse(XContentParser parser, boolean coerce) throws IOException {
                return parser.longValue(coerce);
            }

            @Override
    /**
     * @brief [Functional Utility for compile]: Describe purpose here.
     * @param fieldName: [Description]
     * @param script: [Description]
     * @param compiler: [Description]
     * @return [ReturnType]: [Description]
     */
            public FieldValues<Number> compile(String fieldName, Script script, ScriptCompiler compiler) {
                final LongFieldScript.Factory scriptFactory = compiler.compile(script, LongFieldScript.CONTEXT);
                return (lookup, ctx, doc, consumer) -> scriptFactory.newFactory(fieldName, script.getParams(), lookup, OnScriptError.FAIL)
                    .newInstance(ctx)
                    .runForDoc(doc, consumer::accept);
            }

            @Override
    /**
     * @brief [Functional Utility for termQuery]: Describe purpose here.
     * @param field: [Description]
     * @param value: [Description]
     * @param isIndexed: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termQuery(String field, Object value, boolean isIndexed) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (hasDecimalPart(value)) {
                    return Queries.newMatchNoDocsQuery("Value [" + value + "] has a decimal part");
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isOutOfRange(value)) {
                    return Queries.newMatchNoDocsQuery("Value [" + value + "] is out of range");
                }

                long v = parse(value, true);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (isIndexed) {
                    return LongPoint.newExactQuery(field, v);
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    return SortedNumericDocValuesField.newSlowExactQuery(field, v);
                }
            }

            @Override
    /**
     * @brief [Functional Utility for termsQuery]: Describe purpose here.
     * @param field: [Description]
     * @param values: [Description]
     * @return [ReturnType]: [Description]
     */
            public Query termsQuery(String field, Collection<?> values) {
                long[] v = new long[values.size()];
    /**
     * @brief [Functional description for field upTo]: Describe purpose here.
     */
                int upTo = 0;

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                for (Object value : values) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (hasDecimalPart(value) == false) {
                        v[upTo++] = parse(value, true);
                    }
                }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (upTo == 0) {
                    return Queries.newMatchNoDocsQuery("All values have a decimal part");
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (upTo != v.length) {
                    v = Arrays.copyOf(v, upTo);
                }
                return LongPoint.newSetQuery(field, v);
            }

            @Override
            public Query rangeQuery(
                String field,
                Object lowerTerm,
                Object upperTerm,
                boolean includeLower,
                boolean includeUpper,
                boolean hasDocValues,
                SearchExecutionContext context,
                boolean isIndexed
            ) {
                return longRangeQuery(lowerTerm, upperTerm, includeLower, includeUpper, (l, u) -> {
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                    Query query;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (isIndexed) {
                        query = LongPoint.newRangeQuery(field, l, u);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                        if (hasDocValues) {
                            Query dvQuery = SortedNumericDocValuesField.newSlowRangeQuery(field, l, u);
                            query = new IndexOrDocValuesQuery(query, dvQuery);
                        }
        // Block Logic: [Describe purpose of this else/else if block]
                    } else {
                        query = SortedNumericDocValuesField.newSlowRangeQuery(field, l, u);
                    }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (hasDocValues && context.indexSortedOnField(field)) {
                        query = new XIndexSortSortedNumericDocValuesRangeQuery(field, l, u, query);
                    }
    /**
     * @brief [Functional description for field query]: Describe purpose here.
     */
                    return query;
                });
            }

            @Override
    /**
     * @brief [Functional Utility for addFields]: Describe purpose here.
     * @param document: [Description]
     * @param name: [Description]
     * @param value: [Description]
     * @param indexed: [Description]
     * @param docValued: [Description]
     * @param stored: [Description]
     * @return [ReturnType]: [Description]
     */
            public void addFields(LuceneDocument document, String name, Number value, boolean indexed, boolean docValued, boolean stored) {
                final long l = value.longValue();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (indexed && docValued) {
                    document.add(new LongField(name, l, Field.Store.NO));
        // Block Logic: [Describe purpose of this else/else if block]
                } else if (docValued) {
                    document.add(new SortedNumericDocValuesField(name, l));
        // Block Logic: [Describe purpose of this else/else if block]
                } else if (indexed) {
                    document.add(new LongPoint(name, l));
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (stored) {
                    document.add(new StoredField(name, l));
                }
            }

            @Override
    /**
     * @brief [Functional Utility for getFieldDataBuilder]: Describe purpose here.
     * @param ft: [Description]
     * @param valuesSourceType: [Description]
     * @return [ReturnType]: [Description]
     */
            public IndexFieldData.Builder getFieldDataBuilder(MappedFieldType ft, ValuesSourceType valuesSourceType) {
                return new SortedNumericIndexFieldData.Builder(
                    ft.name(),
                    numericType(),
                    valuesSourceType,
                    LongDocValuesField::new,
                    ft.isIndexed()
                );
            }

            @Override
            public IndexFieldData.Builder getValueFetcherFieldDataBuilder(
                String name,
                ValuesSourceType valuesSourceType,
                SourceProvider sourceProvider,
                ValueFetcher valueFetcher
            ) {
                return new SourceValueFetcherSortedNumericIndexFieldData.Builder(
                    name,
                    valuesSourceType,
                    valueFetcher,
                    sourceProvider,
                    LongDocValuesField::new
                );
            }

            @Override
    /**
     * @brief [Functional Utility for syntheticFieldLoader]: Describe purpose here.
     * @param fieldName: [Description]
     * @param fieldSimpleName: [Description]
     * @param ignoreMalformed: [Description]
     * @return [ReturnType]: [Description]
     */
            SourceLoader.SyntheticFieldLoader syntheticFieldLoader(String fieldName, String fieldSimpleName, boolean ignoreMalformed) {
                return syntheticLongFieldLoader(fieldName, fieldSimpleName, ignoreMalformed);
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromDocValues]: Describe purpose here.
     * @param fieldName: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromDocValues(String fieldName) {
                return new BlockDocValuesReader.LongsBlockLoader(fieldName);
            }

            @Override
    /**
     * @brief [Functional Utility for blockLoaderFromSource]: Describe purpose here.
     * @param sourceValueFetcher: [Description]
     * @param lookup: [Description]
     * @return [ReturnType]: [Description]
     */
            BlockLoader blockLoaderFromSource(SourceValueFetcher sourceValueFetcher, BlockSourceReader.LeafIteratorLookup lookup) {
                return new BlockSourceReader.LongsBlockLoader(sourceValueFetcher, lookup);
            }

    /**
     * @brief [Functional Utility for isOutOfRange]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
            private boolean isOutOfRange(Object value) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (value instanceof Long) {
    /**
     * @brief [Functional description for field false]: Describe purpose here.
     */
                    return false;
                }
                String stringValue = (value instanceof BytesRef) ? ((BytesRef) value).utf8ToString() : value.toString();
                BigDecimal bigDecimalValue = new BigDecimal(stringValue);
                return bigDecimalValue.compareTo(BigDecimal.valueOf(Long.MAX_VALUE)) > 0
                    || bigDecimalValue.compareTo(BigDecimal.valueOf(Long.MIN_VALUE)) < 0;
            }
        };

    /**
     * @brief [Functional description for field name]: Describe purpose here.
     */
        private final String name;
    /**
     * @brief [Functional description for field numericType]: Describe purpose here.
     */
        private final NumericType numericType;
    /**
     * @brief [Functional description for field parser]: Describe purpose here.
     */
        private final TypeParser parser;

        NumberType(String name, NumericType numericType) {
            this.name = name;
            this.numericType = numericType;
            this.parser = createTypeParserWithLegacySupport(
                (n, c) -> new Builder(n, this, c.scriptCompiler(), c.getSettings(), c.indexVersionCreated(), c.getIndexSettings().getMode())
            );
        }

        /** Get the associated type name. */
    /**
     * @brief [Functional Utility for typeName]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public final String typeName() {
    /**
     * @brief [Functional description for field name]: Describe purpose here.
     */
            return name;
        }

        /** Get the associated numeric type */
    /**
     * @brief [Functional Utility for numericType]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public final NumericType numericType() {
    /**
     * @brief [Functional description for field numericType]: Describe purpose here.
     */
            return numericType;
        }

    /**
     * @brief [Functional Utility for parser]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public final TypeParser parser() {
    /**
     * @brief [Functional description for field parser]: Describe purpose here.
     */
            return parser;
        }

        public abstract Query termQuery(String field, Object value, boolean isIndexed);

        public abstract Query termsQuery(String field, Collection<?> values);

        public abstract Query rangeQuery(
            String field,
            Object lowerTerm,
            Object upperTerm,
            boolean includeLower,
            boolean includeUpper,
            boolean hasDocValues,
            SearchExecutionContext context,
            boolean isIndexed
        );

        public abstract Number parse(XContentParser parser, boolean coerce) throws IOException;

        public abstract Number parse(Object value, boolean coerce);

        public abstract Number parsePoint(byte[] value);

        /**
         * Maps the given {@code value} to one or more Lucene field values ands them to the given {@code document} under the given
         * {@code name}.
         *
         * @param document document to add fields to
         * @param name field name
         * @param value value to map
         * @param indexed whether or not the field is indexed
         * @param docValued whether or not doc values should be added
         * @param stored whether or not the field is stored
         */
        public abstract void addFields(
            LuceneDocument document,
            String name,
            Number value,
            boolean indexed,
            boolean docValued,
            boolean stored
        );

    /**
     * @brief [Functional Utility for compile]: Describe purpose here.
     * @param fieldName: [Description]
     * @param script: [Description]
     * @param compiler: [Description]
     * @return [ReturnType]: [Description]
     */
        public FieldValues<Number> compile(String fieldName, Script script, ScriptCompiler compiler) {
            // only implemented for long and double fields
            throw new IllegalArgumentException("Unknown parameter [script] for mapper [" + fieldName + "]");
        }

    /**
     * @brief [Functional Utility for valueForSearch]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
        Number valueForSearch(Number value) {
    /**
     * @brief [Functional description for field value]: Describe purpose here.
     */
            return value;
        }

        /**
         * Returns true if the object is a number and has a decimal part
         */
        public static boolean hasDecimalPart(Object number) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (number instanceof Byte || number instanceof Short || number instanceof Integer || number instanceof Long) {
    /**
     * @brief [Functional description for field false]: Describe purpose here.
     */
                return false;
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (number instanceof Number) {
                double doubleValue = ((Number) number).doubleValue();
                return doubleValue % 1 != 0;
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (number instanceof BytesRef) {
                number = ((BytesRef) number).utf8ToString();
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (number instanceof String) {
                return Double.parseDouble((String) number) % 1 != 0;
            }
    /**
     * @brief [Functional description for field false]: Describe purpose here.
     */
            return false;
        }

        /**
         * Returns -1, 0, or 1 if the value is lower than, equal to, or greater than 0
         */
        static double signum(Object value) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (value instanceof Number) {
                double doubleValue = ((Number) value).doubleValue();
                return Math.signum(doubleValue);
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (value instanceof BytesRef) {
                value = ((BytesRef) value).utf8ToString();
            }
            return Math.signum(Double.parseDouble(value.toString()));
        }

        /**
         * Converts an Object to a double by checking it against known types first
         */
        public static double objectToDouble(Object value) {
    /**
     * @brief [Functional description for field doubleValue]: Describe purpose here.
     */
            double doubleValue;

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (value instanceof Number) {
                doubleValue = ((Number) value).doubleValue();
        // Block Logic: [Describe purpose of this else/else if block]
            } else if (value instanceof BytesRef) {
                doubleValue = Double.parseDouble(((BytesRef) value).utf8ToString());
        // Block Logic: [Describe purpose of this else/else if block]
            } else {
                doubleValue = Double.parseDouble(value.toString());
            }

    /**
     * @brief [Functional description for field doubleValue]: Describe purpose here.
     */
            return doubleValue;
        }

        /**
         * Converts an Object to a {@code long} by checking it against known
         * types and checking its range.
         */
        public static long objectToLong(Object value, boolean coerce) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (value instanceof Long) {
                return (Long) value;
            }

            double doubleValue = objectToDouble(value);
            // this check does not guarantee that value is inside MIN_VALUE/MAX_VALUE because values up to 9223372036854776832 will
            // be equal to Long.MAX_VALUE after conversion to double. More checks ahead.
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (doubleValue < Long.MIN_VALUE || doubleValue > Long.MAX_VALUE) {
                throw new IllegalArgumentException("Value [" + value + "] is out of range for a long");
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (coerce == false && doubleValue % 1 != 0) {
                throw new IllegalArgumentException("Value [" + value + "] has a decimal part");
            }

            // longs need special handling so we don't lose precision while parsing
            String stringValue = (value instanceof BytesRef) ? ((BytesRef) value).utf8ToString() : value.toString();
            return Numbers.toLong(stringValue, coerce);
        }

        public static Query doubleRangeQuery(
            Object lowerTerm,
            Object upperTerm,
            boolean includeLower,
            boolean includeUpper,
            BiFunction<Double, Double, Query> builder
        ) {
    /**
     * @brief [Functional description for field l]: Describe purpose here.
     */
            double l = Double.NEGATIVE_INFINITY;
    /**
     * @brief [Functional description for field u]: Describe purpose here.
     */
            double u = Double.POSITIVE_INFINITY;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (lowerTerm != null) {
                l = objectToDouble(lowerTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (includeLower == false) {
                    l = DoublePoint.nextUp(l);
                }
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (upperTerm != null) {
                u = objectToDouble(upperTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (includeUpper == false) {
                    u = DoublePoint.nextDown(u);
                }
            }
            return builder.apply(l, u);
        }

        /**
         * Processes query bounds into {@code long}s and delegates the
         * provided {@code builder} to build a range query.
         */
        public static Query longRangeQuery(
            Object lowerTerm,
            Object upperTerm,
            boolean includeLower,
            boolean includeUpper,
            BiFunction<Long, Long, Query> builder
        ) {
    /**
     * @brief [Functional description for field l]: Describe purpose here.
     */
            long l = Long.MIN_VALUE;
    /**
     * @brief [Functional description for field u]: Describe purpose here.
     */
            long u = Long.MAX_VALUE;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (lowerTerm != null) {
                l = objectToLong(lowerTerm, true);
                // if the lower bound is decimal:
                // - if the bound is positive then we increment it:
                // if lowerTerm=1.5 then the (inclusive) bound becomes 2
                // - if the bound is negative then we leave it as is:
                // if lowerTerm=-1.5 then the (inclusive) bound becomes -1 due to the call to longValue
                boolean lowerTermHasDecimalPart = hasDecimalPart(lowerTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if ((lowerTermHasDecimalPart == false && includeLower == false) || (lowerTermHasDecimalPart && signum(lowerTerm) > 0)) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (l == Long.MAX_VALUE) {
                        return new MatchNoDocsQuery();
                    }
                    ++l;
                }
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (upperTerm != null) {
                u = objectToLong(upperTerm, true);
                boolean upperTermHasDecimalPart = hasDecimalPart(upperTerm);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if ((upperTermHasDecimalPart == false && includeUpper == false) || (upperTermHasDecimalPart && signum(upperTerm) < 0)) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (u == Long.MIN_VALUE) {
                        return new MatchNoDocsQuery();
                    }
                    --u;
                }
            }
            return builder.apply(l, u);
        }

        public abstract IndexFieldData.Builder getFieldDataBuilder(MappedFieldType ft, ValuesSourceType valuesSourceType);

        public IndexFieldData.Builder getValueFetcherFieldDataBuilder(
            String name,
            ValuesSourceType valuesSourceType,
            SourceProvider sourceProvider,
            ValueFetcher valueFetcher
        ) {
            throw new UnsupportedOperationException("not supported for source fallback");
        }

        /**
         * Adjusts a value to the value it would have been had it been parsed by that mapper
         * and then cast up to a double. This is meant to be an entry point to manipulate values
         * before the actual value is parsed.
         *
         * @param value the value to reduce to the field stored value
         * @return the double value
         */
        public double reduceToStoredPrecision(double value) {
            return ((Number) value).doubleValue();
        }

        abstract SourceLoader.SyntheticFieldLoader syntheticFieldLoader(String fieldName, String fieldSimpleName, boolean ignoreMalformed);

        private static SourceLoader.SyntheticFieldLoader syntheticLongFieldLoader(
            String fieldName,
            String fieldSimpleName,
            boolean ignoreMalformed
        ) {
            return new SortedNumericDocValuesSyntheticFieldLoader(fieldName, fieldSimpleName, ignoreMalformed) {
                @Override
    /**
     * @brief [Functional Utility for writeValue]: Describe purpose here.
     * @param b: [Description]
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
                protected void writeValue(XContentBuilder b, long value) throws IOException {
                    b.value(value);
                }
            };
        }

        abstract BlockLoader blockLoaderFromDocValues(String fieldName);

        abstract BlockLoader blockLoaderFromSource(SourceValueFetcher sourceValueFetcher, BlockSourceReader.LeafIteratorLookup lookup);
    }

    public static class NumberFieldType extends SimpleMappedFieldType {

    /**
     * @brief [Functional description for field type]: Describe purpose here.
     */
        private final NumberType type;
    /**
     * @brief [Functional description for field coerce]: Describe purpose here.
     */
        private final boolean coerce;
    /**
     * @brief [Functional description for field nullValue]: Describe purpose here.
     */
        private final Number nullValue;
    /**
     * @brief [Functional description for field scriptValues]: Describe purpose here.
     */
        private final FieldValues<Number> scriptValues;
    /**
     * @brief [Functional description for field isDimension]: Describe purpose here.
     */
        private final boolean isDimension;
    /**
     * @brief [Functional description for field metricType]: Describe purpose here.
     */
        private final MetricType metricType;
    /**
     * @brief [Functional description for field indexMode]: Describe purpose here.
     */
        private final IndexMode indexMode;

        public NumberFieldType(
            String name,
            NumberType type,
            boolean isIndexed,
            boolean isStored,
            boolean hasDocValues,
            boolean coerce,
            Number nullValue,
            Map<String, String> meta,
            FieldValues<Number> script,
            boolean isDimension,
            MetricType metricType,
            IndexMode indexMode
        ) {
            super(name, isIndexed, isStored, hasDocValues, TextSearchInfo.SIMPLE_MATCH_WITHOUT_TERMS, meta);
            this.type = Objects.requireNonNull(type);
            this.coerce = coerce;
            this.nullValue = nullValue;
            this.scriptValues = script;
            this.isDimension = isDimension;
            this.metricType = metricType;
            this.indexMode = indexMode;
        }

        NumberFieldType(String name, Builder builder) {
            this(
                name,
                builder.type,
                builder.indexed.getValue() && builder.indexCreatedVersion.isLegacyIndexVersion() == false,
                builder.stored.getValue(),
                builder.hasDocValues.getValue(),
                builder.coerce.getValue().value(),
                builder.nullValue.getValue(),
                builder.meta.getValue(),
                builder.scriptValues(),
                builder.dimension.getValue(),
                builder.metric.getValue(),
                builder.indexMode
            );
        }

    /**
     * @brief [Functional Utility for NumberFieldType]: Describe purpose here.
     * @param name: [Description]
     * @param type: [Description]
     * @return [ReturnType]: [Description]
     */
        public NumberFieldType(String name, NumberType type) {
            this(name, type, true);
        }

    /**
     * @brief [Functional Utility for NumberFieldType]: Describe purpose here.
     * @param name: [Description]
     * @param type: [Description]
     * @param isIndexed: [Description]
     * @return [ReturnType]: [Description]
     */
        public NumberFieldType(String name, NumberType type, boolean isIndexed) {
            this(name, type, isIndexed, false, true, true, null, Collections.emptyMap(), null, false, null, null);
        }

        @Override
    /**
     * @brief [Functional Utility for typeName]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public String typeName() {
            return type.name;
        }

        /**
         * This method reinterprets a double precision value based on the maximum precision of the stored number field.  Mostly this
         * corrects for unrepresentable values which have different approximations when cast from floats than when parsed as doubles.
         * It may seem strange to convert a double to a double, and it is.  This function's goal is to reduce the precision
         * on the double in the case that the backing number type would have parsed the value differently.  This is to address
         * the problem where (e.g.) 0.04F &lt; 0.04D, which causes problems for range aggregations.
         */
        public double reduceToStoredPrecision(double value) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (Double.isInfinite(value)) {
                // Trying to parse infinite values into ints/longs throws. Understandably.
    /**
     * @brief [Functional description for field value]: Describe purpose here.
     */
                return value;
            }
            return type.reduceToStoredPrecision(value);
        }

    /**
     * @brief [Functional Utility for numericType]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public NumericType numericType() {
            return type.numericType();
        }

        @Override
    /**
     * @brief [Functional Utility for mayExistInIndex]: Describe purpose here.
     * @param context: [Description]
     * @return [ReturnType]: [Description]
     */
        public boolean mayExistInIndex(SearchExecutionContext context) {
            return context.fieldExistsInIndex(this.name());
        }

    /**
     * @brief [Functional Utility for isSearchable]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public boolean isSearchable() {
            return isIndexed() || hasDocValues();
        }

        @Override
    /**
     * @brief [Functional Utility for termQuery]: Describe purpose here.
     * @param value: [Description]
     * @param context: [Description]
     * @return [ReturnType]: [Description]
     */
        public Query termQuery(Object value, SearchExecutionContext context) {
            failIfNotIndexedNorDocValuesFallback(context);
            return type.termQuery(name(), value, isIndexed());
        }

        @Override
    /**
     * @brief [Functional Utility for termsQuery]: Describe purpose here.
     * @param values: [Description]
     * @param context: [Description]
     * @return [ReturnType]: [Description]
     */
        public Query termsQuery(Collection<?> values, SearchExecutionContext context) {
            failIfNotIndexedNorDocValuesFallback(context);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (isIndexed()) {
                return type.termsQuery(name(), values);
        // Block Logic: [Describe purpose of this else/else if block]
            } else {
                return super.termsQuery(values, context);
            }
        }

        @Override
        public Query rangeQuery(
            Object lowerTerm,
            Object upperTerm,
            boolean includeLower,
            boolean includeUpper,
            SearchExecutionContext context
        ) {
            failIfNotIndexedNorDocValuesFallback(context);
            return type.rangeQuery(name(), lowerTerm, upperTerm, includeLower, includeUpper, hasDocValues(), context, isIndexed());
        }

        @Override
        public Function<byte[], Number> pointReaderIfPossible() {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (isIndexed()) {
                return this::parsePoint;
            }
    /**
     * @brief [Functional description for field null]: Describe purpose here.
     */
            return null;
        }

        @Override
    /**
     * @brief [Functional Utility for blockLoader]: Describe purpose here.
     * @param blContext: [Description]
     * @return [ReturnType]: [Description]
     */
        public BlockLoader blockLoader(BlockLoaderContext blContext) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (hasDocValues()) {
                return type.blockLoaderFromDocValues(name());
            }
            BlockSourceReader.LeafIteratorLookup lookup = isStored() || isIndexed()
                ? BlockSourceReader.lookupFromFieldNames(blContext.fieldNames(), name())
                : BlockSourceReader.lookupMatchingAll();
            return type.blockLoaderFromSource(sourceValueFetcher(blContext.sourcePaths(name())), lookup);
        }

        @Override
    /**
     * @brief [Functional Utility for fielddataBuilder]: Describe purpose here.
     * @param fieldDataContext: [Description]
     * @return [ReturnType]: [Description]
     */
        public IndexFieldData.Builder fielddataBuilder(FieldDataContext fieldDataContext) {
            FielddataOperation operation = fieldDataContext.fielddataOperation();

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (fieldDataContext.fielddataOperation() == FielddataOperation.SEARCH) {
                failIfNoDocValues();
            }

            ValuesSourceType valuesSourceType = indexMode == IndexMode.TIME_SERIES && metricType == TimeSeriesParams.MetricType.COUNTER
                ? TimeSeriesValuesSourceType.COUNTER
                : type.numericType.getValuesSourceType();

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if ((operation == FielddataOperation.SEARCH || operation == FielddataOperation.SCRIPT) && hasDocValues()) {
                return type.getFieldDataBuilder(this, valuesSourceType);
            }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (operation == FielddataOperation.SCRIPT) {
                SearchLookup searchLookup = fieldDataContext.lookupSupplier().get();
                Set<String> sourcePaths = fieldDataContext.sourcePathsLookup().apply(name());
                return type.getValueFetcherFieldDataBuilder(name(), valuesSourceType, searchLookup, sourceValueFetcher(sourcePaths));
            }

            throw new IllegalStateException("unknown field data type [" + operation.name() + "]");
        }

        @Override
    /**
     * @brief [Functional Utility for valueForDisplay]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
        public Object valueForDisplay(Object value) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (value == null) {
    /**
     * @brief [Functional description for field null]: Describe purpose here.
     */
                return null;
            }
            return type.valueForSearch((Number) value);
        }

        @Override
    /**
     * @brief [Functional Utility for valueFetcher]: Describe purpose here.
     * @param context: [Description]
     * @param format: [Description]
     * @return [ReturnType]: [Description]
     */
        public ValueFetcher valueFetcher(SearchExecutionContext context, String format) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (format != null) {
                throw new IllegalArgumentException("Field [" + name() + "] of type [" + typeName() + "] doesn't support formats.");
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (this.scriptValues != null) {
                return FieldValues.valueFetcher(this.scriptValues, context);
            }
            return sourceValueFetcher(context.isSourceEnabled() ? context.sourcePath(name()) : Collections.emptySet());
        }

    /**
     * @brief [Functional Utility for sourceValueFetcher]: Describe purpose here.
     * @param sourcePaths: [Description]
     * @return [ReturnType]: [Description]
     */
        private SourceValueFetcher sourceValueFetcher(Set<String> sourcePaths) {
            return new SourceValueFetcher(sourcePaths, nullValue) {
                @Override
    /**
     * @brief [Functional Utility for parseSourceValue]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
                protected Object parseSourceValue(Object value) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (value.equals("")) {
    /**
     * @brief [Functional description for field nullValue]: Describe purpose here.
     */
                        return nullValue;
                    }
                    return type.parse(value, coerce);
                }
            };
        }

        @Override
    /**
     * @brief [Functional Utility for docValueFormat]: Describe purpose here.
     * @param format: [Description]
     * @param timeZone: [Description]
     * @return [ReturnType]: [Description]
     */
        public DocValueFormat docValueFormat(String format, ZoneId timeZone) {
            checkNoTimeZone(timeZone);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (format == null) {
                return DocValueFormat.RAW;
            }
            return new DocValueFormat.Decimal(format);
        }

    /**
     * @brief [Functional Utility for parsePoint]: Describe purpose here.
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
        public Number parsePoint(byte[] value) {
            return type.parsePoint(value);
        }

        @Override
    /**
     * @brief [Functional Utility for collapseType]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public CollapseType collapseType() {
            return CollapseType.NUMERIC;
        }

        @Override
    /**
     * @brief [Functional Utility for isDimension]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public boolean isDimension() {
    /**
     * @brief [Functional description for field isDimension]: Describe purpose here.
     */
            return isDimension;
        }

        @Override
    /**
     * @brief [Functional Utility for hasScriptValues]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public boolean hasScriptValues() {
            return scriptValues != null;
        }

        /**
         * If field is a time series metric field, returns its metric type
         * @return the metric type or null
         */
        public MetricType getMetricType() {
    /**
     * @brief [Functional description for field metricType]: Describe purpose here.
     */
            return metricType;
        }
    }

    /**
     * @brief [Functional description for field type]: Describe purpose here.
     */
    private final NumberType type;
    /**
     * @brief [Functional description for field indexed]: Describe purpose here.
     */
    private final boolean indexed;
    /**
     * @brief [Functional description for field hasDocValues]: Describe purpose here.
     */
    private final boolean hasDocValues;
    /**
     * @brief [Functional description for field stored]: Describe purpose here.
     */
    private final boolean stored;
    /**
     * @brief [Functional description for field ignoreMalformed]: Describe purpose here.
     */
    private final Explicit<Boolean> ignoreMalformed;
    /**
     * @brief [Functional description for field coerce]: Describe purpose here.
     */
    private final Explicit<Boolean> coerce;
    /**
     * @brief [Functional description for field nullValue]: Describe purpose here.
     */
    private final Number nullValue;
    /**
     * @brief [Functional description for field scriptValues]: Describe purpose here.
     */
    private final FieldValues<Number> scriptValues;
    /**
     * @brief [Functional description for field ignoreMalformedByDefault]: Describe purpose here.
     */
    private final boolean ignoreMalformedByDefault;
    /**
     * @brief [Functional description for field coerceByDefault]: Describe purpose here.
     */
    private final boolean coerceByDefault;
    /**
     * @brief [Functional description for field dimension]: Describe purpose here.
     */
    private final boolean dimension;
    /**
     * @brief [Functional description for field scriptCompiler]: Describe purpose here.
     */
    private final ScriptCompiler scriptCompiler;
    /**
     * @brief [Functional description for field script]: Describe purpose here.
     */
    private final Script script;
    /**
     * @brief [Functional description for field metricType]: Describe purpose here.
     */
    private final MetricType metricType;
    /**
     * @brief [Functional description for field allowMultipleValues]: Describe purpose here.
     */
    private boolean allowMultipleValues;
    /**
     * @brief [Functional description for field indexCreatedVersion]: Describe purpose here.
     */
    private final IndexVersion indexCreatedVersion;
    /**
     * @brief [Functional description for field storeMalformedFields]: Describe purpose here.
     */
    private final boolean storeMalformedFields;

    /**
     * @brief [Functional description for field indexMode]: Describe purpose here.
     */
    private final IndexMode indexMode;

    private NumberFieldMapper(
        String simpleName,
        MappedFieldType mappedFieldType,
        BuilderParams builderParams,
        boolean storeMalformedFields,
        Builder builder
    ) {
        super(simpleName, mappedFieldType, builderParams);
        this.type = builder.type;
        this.indexed = builder.indexed.getValue();
        this.hasDocValues = builder.hasDocValues.getValue();
        this.stored = builder.stored.getValue();
        this.ignoreMalformed = builder.ignoreMalformed.getValue();
        this.coerce = builder.coerce.getValue();
        this.nullValue = builder.nullValue.getValue();
        this.ignoreMalformedByDefault = builder.ignoreMalformed.getDefaultValue().value();
        this.coerceByDefault = builder.coerce.getDefaultValue().value();
        this.scriptValues = builder.scriptValues();
        this.dimension = builder.dimension.getValue();
        this.scriptCompiler = builder.scriptCompiler;
        this.script = builder.script.getValue();
        this.metricType = builder.metric.getValue();
        this.allowMultipleValues = builder.allowMultipleValues;
        this.indexCreatedVersion = builder.indexCreatedVersion;
        this.storeMalformedFields = storeMalformedFields;
        this.indexMode = builder.indexMode;
    }

    /**
     * @brief [Functional Utility for coerce]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    boolean coerce() {
        return coerce.value();
    }

    @Override
    /**
     * @brief [Functional Utility for ignoreMalformed]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public boolean ignoreMalformed() {
        return ignoreMalformed.value();
    }

    @Override
    /**
     * @brief [Functional Utility for fieldType]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public NumberFieldType fieldType() {
        return (NumberFieldType) super.fieldType();
    }

    /**
     * @brief [Functional Utility for type]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public NumberType type() {
    /**
     * @brief [Functional description for field type]: Describe purpose here.
     */
        return type;
    }

    @Override
    /**
     * @brief [Functional Utility for contentType]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    protected String contentType() {
        return fieldType().type.typeName();
    }

    @Override
    /**
     * @brief [Functional Utility for parseCreateField]: Describe purpose here.
     * @param context: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    protected void parseCreateField(DocumentParserContext context) throws IOException {
    /**
     * @brief [Functional description for field value]: Describe purpose here.
     */
        Number value;
        try {
            value = value(context.parser());
        } catch (IllegalArgumentException e) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (ignoreMalformed.value() && context.parser().currentToken().isValue()) {
                context.addIgnoredField(mappedFieldType.name());
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (storeMalformedFields) {
                    // Save a copy of the field so synthetic source can load it
                    context.doc().add(IgnoreMalformedStoredValues.storedField(fullPath(), context.parser()));
                }
                return;
        // Block Logic: [Describe purpose of this else/else if block]
            } else {
    /**
     * @brief [Functional description for field e]: Describe purpose here.
     */
                throw e;
            }
        }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (value != null) {
            indexValue(context, value);
        }
    }

    /**
     * Read the value at the current position of the parser. For numeric fields
     * this is called by {@link #parseCreateField} but it is public so it can
     * be used by other fields that want to share the behavior of numeric fields.
     * @throws IllegalArgumentException if there was an error parsing the value from the json
     * @throws IOException if there was any other IO error
     */
    public Number value(XContentParser parser) throws IllegalArgumentException, IOException {
        final Token currentToken = parser.currentToken();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (currentToken == Token.VALUE_NULL) {
    /**
     * @brief [Functional description for field nullValue]: Describe purpose here.
     */
            return nullValue;
        }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (coerce() && currentToken == Token.VALUE_STRING && parser.textLength() == 0) {
    /**
     * @brief [Functional description for field nullValue]: Describe purpose here.
     */
            return nullValue;
        }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (currentToken == Token.START_OBJECT) {
            throw new IllegalArgumentException("Cannot parse object as number");
        }
        return type.parse(parser, coerce());
    }

    /**
     * Index a value for this field. For numeric fields this is called by
     * {@link #parseCreateField} but it is public so it can be used by other
     * fields that want to share the behavior of numeric fields.
     */
    public void indexValue(DocumentParserContext context, Number numericValue) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (dimension && numericValue != null) {
            context.getRoutingFields().addLong(fieldType().name(), numericValue.longValue());
        }
        fieldType().type.addFields(context.doc(), fieldType().name(), numericValue, indexed, hasDocValues, stored);

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (false == allowMultipleValues && (indexed || hasDocValues || stored)) {
            // the last field is the current field, Add to the key map, so that we can validate if it has been added
            List<IndexableField> fields = context.doc().getFields();
            IndexableField last = fields.get(fields.size() - 1);
            assert last.name().equals(fieldType().name())
                : "last field name [" + last.name() + "] mis match field name [" + fieldType().name() + "]";
            context.doc().onlyAddKey(fieldType().name(), fields.get(fields.size() - 1));
        }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (hasDocValues == false && (stored || indexed)) {
            context.addToFieldNames(fieldType().name());
        }
    }

    @Override
    protected void indexScriptValues(
        SearchLookup searchLookup,
        LeafReaderContext readerContext,
        int doc,
        DocumentParserContext documentParserContext
    ) {
        this.scriptValues.valuesForDoc(searchLookup, readerContext, doc, value -> indexValue(documentParserContext, value));
    }

    @Override
    /**
     * @brief [Functional Utility for getMergeBuilder]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public FieldMapper.Builder getMergeBuilder() {
        return new Builder(leafName(), type, scriptCompiler, ignoreMalformedByDefault, coerceByDefault, indexCreatedVersion, indexMode)
            .dimension(dimension)
            .metric(metricType)
            .allowMultipleValues(allowMultipleValues)
            .init(this);
    }

    @Override
    /**
     * @brief [Functional Utility for doValidate]: Describe purpose here.
     * @param lookup: [Description]
     * @return [ReturnType]: [Description]
     */
    public void doValidate(MappingLookup lookup) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (dimension && null != lookup.nestedLookup().getNestedParent(fullPath())) {
            throw new IllegalArgumentException(
                TimeSeriesParams.TIME_SERIES_DIMENSION_PARAM + " can't be configured in nested field [" + fullPath() + "]"
            );
        }
    }

    @Override
    /**
     * @brief [Functional Utility for syntheticSourceSupport]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    protected SyntheticSourceSupport syntheticSourceSupport() {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (hasDocValues) {
            return new SyntheticSourceSupport.Native(() -> type.syntheticFieldLoader(fullPath(), leafName(), ignoreMalformed.value()));
        }

        return super.syntheticSourceSupport();
    }

    // For testing only:
    /**
     * @brief [Functional Utility for setAllowMultipleValues]: Describe purpose here.
     * @param allowMultipleValues: [Description]
     * @return [ReturnType]: [Description]
     */
    void setAllowMultipleValues(boolean allowMultipleValues) {
        this.allowMultipleValues = allowMultipleValues;
    }
}
