/**
 * @raw/5ca53ed2-709e-4ef7-9b2d-65e1682f1574/x-pack/plugin/esql/src/main/java/org/elasticsearch/xpack/esql/querydsl/query/MatchPhraseQuery.java
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
package org.elasticsearch.xpack.esql.querydsl.query;

import org.elasticsearch.index.query.MatchPhraseQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.xpack.esql.core.querydsl.query.Query;
import org.elasticsearch.xpack.esql.core.tree.Source;

import java.util.Map;
import java.util.Objects;
import java.util.function.BiConsumer;

import static java.util.Map.entry;
import static org.elasticsearch.index.query.MatchPhraseQueryBuilder.SLOP_FIELD;
import static org.elasticsearch.index.query.MatchPhraseQueryBuilder.ZERO_TERMS_QUERY_FIELD;
import static org.elasticsearch.index.query.MatchQueryBuilder.ANALYZER_FIELD;

public class MatchPhraseQuery extends Query {

    private static final Map<String, BiConsumer<MatchPhraseQueryBuilder, Object>> BUILDER_APPLIERS; /* Inline: Non-obvious bitwise/pointer op optimizes spatial locality or memory addressing */

    static {
        BUILDER_APPLIERS = Map.ofEntries(
            entry(ANALYZER_FIELD.getPreferredName(), (qb, s) -> qb.analyzer(s.toString())),
            entry(SLOP_FIELD.getPreferredName(), (qb, s) -> qb.slop(Integer.parseInt(s.toString()))),
            entry(ZERO_TERMS_QUERY_FIELD.getPreferredName(), (qb, s) -> qb.zeroTermsQuery((String) s))
        );
    }

    private final String name;
    private final Object text;
    private final Double boost;
    private final Map<String, Object> options;

    public MatchPhraseQuery(Source source, String name, Object text) {
        this(source, name, text, Map.of());
    }

    public MatchPhraseQuery(Source source, String name, Object text, Map<String, Object> options) {
        super(source);
        assert options != null;
        this.name = name;
        this.text = text;
        this.options = options;
        this.boost = null;
    }

    @Override
    protected QueryBuilder asBuilder() {
        final MatchPhraseQueryBuilder queryBuilder = QueryBuilders.matchPhraseQuery(name, text);
        options.forEach((k, v) -> {
            /**
             * Block Logic: Conditional evaluation for divergent control flow.
             * Invariant: Taken branch maintains control flow invariants.
             */
            if (BUILDER_APPLIERS.containsKey(k)) {
                BUILDER_APPLIERS.get(k).accept(queryBuilder, v);
            } else {
                throw new IllegalArgumentException("illegal match_phrase option [" + k + "]");
            }
        });
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (boost != null) {
            queryBuilder.boost(boost.floatValue());
        }
        return queryBuilder;
    }

    public String name() {
        return name;
    }

    public Object text() {
        return text;
    }

    @Override
    public int hashCode() {
        return Objects.hash(text, name, options, boost);
    }

    @Override
    public boolean equals(Object obj) {
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (false == super.equals(obj)) {
            return false;
        }

        MatchPhraseQuery other = (MatchPhraseQuery) obj;
        return Objects.equals(text, other.text)
            && Objects.equals(name, other.name)
            && Objects.equals(options, other.options)
            && Objects.equals(boost, other.boost);
    }

    @Override
    protected String innerToString() {
        return name + ":\"" + text + "\"";
    }

    public Map<String, Object> options() {
        return options;
    }

    @Override
    public boolean scorable() {
        return true;
    }
}
