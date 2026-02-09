/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.fetch.subphase.highlight;

import org.apache.lucene.search.Query;
import org.elasticsearch.common.util.Maps;
import org.elasticsearch.search.fetch.subphase.highlight.HighlightBuilder.BoundaryScannerType;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * @brief Functional description of the SearchHighlightContext class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class SearchHighlightContext {

    private final Map<String, Field> fields;

    /**
     * @brief [Functional Utility for SearchHighlightContext]: Describe purpose here.
     * @param fields: [Description]
     * @return [ReturnType]: [Description]
     */
    public SearchHighlightContext(Collection<Field> fields) {
        assert fields != null;
        this.fields = Maps.newLinkedHashMapWithExpectedSize(fields.size());
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        for (Field field : fields) {
            this.fields.put(field.field, field);
        }
    }

    /**
     * @brief [Functional Utility for fields]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public Collection<Field> fields() {
        return fields.values();
    }

    public static class Field {
    /**
     * @brief [Functional description for field field]: Describe purpose here.
     */
        private final String field;
    /**
     * @brief [Functional description for field fieldOptions]: Describe purpose here.
     */
        private final FieldOptions fieldOptions;

    /**
     * @brief [Functional Utility for Field]: Describe purpose here.
     * @param field: [Description]
     * @param fieldOptions: [Description]
     * @return [ReturnType]: [Description]
     */
        public Field(String field, FieldOptions fieldOptions) {
            assert field != null;
            assert fieldOptions != null;
            this.field = field;
            this.fieldOptions = fieldOptions;
        }

    /**
     * @brief [Functional Utility for field]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public String field() {
    /**
     * @brief [Functional description for field field]: Describe purpose here.
     */
            return field;
        }

    /**
     * @brief [Functional Utility for fieldOptions]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public FieldOptions fieldOptions() {
    /**
     * @brief [Functional description for field fieldOptions]: Describe purpose here.
     */
            return fieldOptions;
        }
    }

    public static class FieldOptions {

        // Field options that default to null or -1 are often set to their real default in HighlighterParseElement#parse
    /**
     * @brief [Functional description for field fragmentCharSize]: Describe purpose here.
     */
        private int fragmentCharSize = -1;

    /**
     * @brief [Functional description for field numberOfFragments]: Describe purpose here.
     */
        private int numberOfFragments = -1;

    /**
     * @brief [Functional description for field fragmentOffset]: Describe purpose here.
     */
        private int fragmentOffset = -1;

    /**
     * @brief [Functional description for field encoder]: Describe purpose here.
     */
        private String encoder;

    /**
     * @brief [Functional description for field preTags]: Describe purpose here.
     */
        private String[] preTags;

    /**
     * @brief [Functional description for field postTags]: Describe purpose here.
     */
        private String[] postTags;

    /**
     * @brief [Functional description for field scoreOrdered]: Describe purpose here.
     */
        private Boolean scoreOrdered;

    /**
     * @brief [Functional description for field highlightFilter]: Describe purpose here.
     */
        private Boolean highlightFilter;

    /**
     * @brief [Functional description for field requireFieldMatch]: Describe purpose here.
     */
        private Boolean requireFieldMatch;

    /**
     * @brief [Functional description for field maxAnalyzedOffset]: Describe purpose here.
     */
        private Integer maxAnalyzedOffset;

    /**
     * @brief [Functional description for field highlighterType]: Describe purpose here.
     */
        private String highlighterType;

    /**
     * @brief [Functional description for field fragmenter]: Describe purpose here.
     */
        private String fragmenter;

    /**
     * @brief [Functional description for field boundaryScannerType]: Describe purpose here.
     */
        private BoundaryScannerType boundaryScannerType;

    /**
     * @brief [Functional description for field boundaryMaxScan]: Describe purpose here.
     */
        private int boundaryMaxScan = -1;

    /**
     * @brief [Functional description for field boundaryChars]: Describe purpose here.
     */
        private char[] boundaryChars = null;

    /**
     * @brief [Functional description for field boundaryScannerLocale]: Describe purpose here.
     */
        private Locale boundaryScannerLocale;

    /**
     * @brief [Functional description for field highlightQuery]: Describe purpose here.
     */
        private Query highlightQuery;

    /**
     * @brief [Functional description for field noMatchSize]: Describe purpose here.
     */
        private int noMatchSize = -1;

    /**
     * @brief [Functional description for field matchedFields]: Describe purpose here.
     */
        private Set<String> matchedFields;

        private Map<String, Object> options;

    /**
     * @brief [Functional description for field phraseLimit]: Describe purpose here.
     */
        private int phraseLimit = -1;

    /**
     * @brief [Functional Utility for fragmentCharSize]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public int fragmentCharSize() {
    /**
     * @brief [Functional description for field fragmentCharSize]: Describe purpose here.
     */
            return fragmentCharSize;
        }

    /**
     * @brief [Functional Utility for numberOfFragments]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public int numberOfFragments() {
    /**
     * @brief [Functional description for field numberOfFragments]: Describe purpose here.
     */
            return numberOfFragments;
        }

    /**
     * @brief [Functional Utility for fragmentOffset]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public int fragmentOffset() {
    /**
     * @brief [Functional description for field fragmentOffset]: Describe purpose here.
     */
            return fragmentOffset;
        }

    /**
     * @brief [Functional Utility for encoder]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public String encoder() {
    /**
     * @brief [Functional description for field encoder]: Describe purpose here.
     */
            return encoder;
        }

    /**
     * @brief [Functional Utility for preTags]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public String[] preTags() {
    /**
     * @brief [Functional description for field preTags]: Describe purpose here.
     */
            return preTags;
        }

    /**
     * @brief [Functional Utility for postTags]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public String[] postTags() {
    /**
     * @brief [Functional description for field postTags]: Describe purpose here.
     */
            return postTags;
        }

    /**
     * @brief [Functional Utility for scoreOrdered]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public Boolean scoreOrdered() {
    /**
     * @brief [Functional description for field scoreOrdered]: Describe purpose here.
     */
            return scoreOrdered;
        }

    /**
     * @brief [Functional Utility for highlightFilter]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public Boolean highlightFilter() {
    /**
     * @brief [Functional description for field highlightFilter]: Describe purpose here.
     */
            return highlightFilter;
        }

    /**
     * @brief [Functional Utility for requireFieldMatch]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public Boolean requireFieldMatch() {
    /**
     * @brief [Functional description for field requireFieldMatch]: Describe purpose here.
     */
            return requireFieldMatch;
        }

    /**
     * @brief [Functional Utility for maxAnalyzedOffset]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public Integer maxAnalyzedOffset() {
    /**
     * @brief [Functional description for field maxAnalyzedOffset]: Describe purpose here.
     */
            return maxAnalyzedOffset;
        }

    /**
     * @brief [Functional Utility for highlighterType]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public String highlighterType() {
    /**
     * @brief [Functional description for field highlighterType]: Describe purpose here.
     */
            return highlighterType;
        }

    /**
     * @brief [Functional Utility for fragmenter]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public String fragmenter() {
    /**
     * @brief [Functional description for field fragmenter]: Describe purpose here.
     */
            return fragmenter;
        }

    /**
     * @brief [Functional Utility for boundaryScannerType]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public BoundaryScannerType boundaryScannerType() {
    /**
     * @brief [Functional description for field boundaryScannerType]: Describe purpose here.
     */
            return boundaryScannerType;
        }

    /**
     * @brief [Functional Utility for boundaryMaxScan]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public int boundaryMaxScan() {
    /**
     * @brief [Functional description for field boundaryMaxScan]: Describe purpose here.
     */
            return boundaryMaxScan;
        }

    /**
     * @brief [Functional Utility for boundaryChars]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public char[] boundaryChars() {
    /**
     * @brief [Functional description for field boundaryChars]: Describe purpose here.
     */
            return boundaryChars;
        }

    /**
     * @brief [Functional Utility for boundaryScannerLocale]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public Locale boundaryScannerLocale() {
    /**
     * @brief [Functional description for field boundaryScannerLocale]: Describe purpose here.
     */
            return boundaryScannerLocale;
        }

    /**
     * @brief [Functional Utility for highlightQuery]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public Query highlightQuery() {
    /**
     * @brief [Functional description for field highlightQuery]: Describe purpose here.
     */
            return highlightQuery;
        }

    /**
     * @brief [Functional Utility for noMatchSize]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public int noMatchSize() {
    /**
     * @brief [Functional description for field noMatchSize]: Describe purpose here.
     */
            return noMatchSize;
        }

    /**
     * @brief [Functional Utility for phraseLimit]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public int phraseLimit() {
    /**
     * @brief [Functional description for field phraseLimit]: Describe purpose here.
     */
            return phraseLimit;
        }

    /**
     * @brief [Functional Utility for matchedFields]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public Set<String> matchedFields() {
    /**
     * @brief [Functional description for field matchedFields]: Describe purpose here.
     */
            return matchedFields;
        }

        public Map<String, Object> options() {
    /**
     * @brief [Functional description for field options]: Describe purpose here.
     */
            return options;
        }

        static class Builder {

            private final FieldOptions fieldOptions = new FieldOptions();

    /**
     * @brief [Functional Utility for fragmentCharSize]: Describe purpose here.
     * @param fragmentCharSize: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder fragmentCharSize(int fragmentCharSize) {
                fieldOptions.fragmentCharSize = fragmentCharSize;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for numberOfFragments]: Describe purpose here.
     * @param numberOfFragments: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder numberOfFragments(int numberOfFragments) {
                fieldOptions.numberOfFragments = numberOfFragments;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for fragmentOffset]: Describe purpose here.
     * @param fragmentOffset: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder fragmentOffset(int fragmentOffset) {
                fieldOptions.fragmentOffset = fragmentOffset;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for encoder]: Describe purpose here.
     * @param encoder: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder encoder(String encoder) {
                fieldOptions.encoder = encoder;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for preTags]: Describe purpose here.
     * @param preTags: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder preTags(String[] preTags) {
                fieldOptions.preTags = preTags;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for postTags]: Describe purpose here.
     * @param postTags: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder postTags(String[] postTags) {
                fieldOptions.postTags = postTags;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for scoreOrdered]: Describe purpose here.
     * @param scoreOrdered: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder scoreOrdered(boolean scoreOrdered) {
                fieldOptions.scoreOrdered = scoreOrdered;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for highlightFilter]: Describe purpose here.
     * @param highlightFilter: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder highlightFilter(boolean highlightFilter) {
                fieldOptions.highlightFilter = highlightFilter;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for requireFieldMatch]: Describe purpose here.
     * @param requireFieldMatch: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder requireFieldMatch(boolean requireFieldMatch) {
                fieldOptions.requireFieldMatch = requireFieldMatch;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for maxAnalyzedOffset]: Describe purpose here.
     * @param maxAnalyzedOffset: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder maxAnalyzedOffset(Integer maxAnalyzedOffset) {
                fieldOptions.maxAnalyzedOffset = maxAnalyzedOffset;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for highlighterType]: Describe purpose here.
     * @param type: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder highlighterType(String type) {
                fieldOptions.highlighterType = type;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for fragmenter]: Describe purpose here.
     * @param fragmenter: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder fragmenter(String fragmenter) {
                fieldOptions.fragmenter = fragmenter;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for boundaryScannerType]: Describe purpose here.
     * @param boundaryScanner: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder boundaryScannerType(BoundaryScannerType boundaryScanner) {
                fieldOptions.boundaryScannerType = boundaryScanner;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for boundaryMaxScan]: Describe purpose here.
     * @param boundaryMaxScan: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder boundaryMaxScan(int boundaryMaxScan) {
                fieldOptions.boundaryMaxScan = boundaryMaxScan;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for boundaryChars]: Describe purpose here.
     * @param boundaryChars: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder boundaryChars(char[] boundaryChars) {
                fieldOptions.boundaryChars = boundaryChars;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for boundaryScannerLocale]: Describe purpose here.
     * @param boundaryScannerLocale: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder boundaryScannerLocale(Locale boundaryScannerLocale) {
                fieldOptions.boundaryScannerLocale = boundaryScannerLocale;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for highlightQuery]: Describe purpose here.
     * @param highlightQuery: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder highlightQuery(Query highlightQuery) {
                fieldOptions.highlightQuery = highlightQuery;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for noMatchSize]: Describe purpose here.
     * @param noMatchSize: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder noMatchSize(int noMatchSize) {
                fieldOptions.noMatchSize = noMatchSize;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for phraseLimit]: Describe purpose here.
     * @param phraseLimit: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder phraseLimit(int phraseLimit) {
                fieldOptions.phraseLimit = phraseLimit;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for matchedFields]: Describe purpose here.
     * @param matchedFields: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder matchedFields(Set<String> matchedFields) {
                fieldOptions.matchedFields = matchedFields;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for options]: Describe purpose here.
     * @param Map<String: [Description]
     * @param options: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder options(Map<String, Object> options) {
                fieldOptions.options = options;
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }

    /**
     * @brief [Functional Utility for build]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
            FieldOptions build() {
    /**
     * @brief [Functional description for field fieldOptions]: Describe purpose here.
     */
                return fieldOptions;
            }

    /**
     * @brief [Functional Utility for merge]: Describe purpose here.
     * @param globalOptions: [Description]
     * @return [ReturnType]: [Description]
     */
            Builder merge(FieldOptions globalOptions) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.preTags == null && globalOptions.preTags != null) {
                    fieldOptions.preTags = Arrays.copyOf(globalOptions.preTags, globalOptions.preTags.length);
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.postTags == null && globalOptions.postTags != null) {
                    fieldOptions.postTags = Arrays.copyOf(globalOptions.postTags, globalOptions.postTags.length);
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.highlightFilter == null) {
                    fieldOptions.highlightFilter = globalOptions.highlightFilter;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.scoreOrdered == null) {
                    fieldOptions.scoreOrdered = globalOptions.scoreOrdered;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.fragmentCharSize == -1) {
                    fieldOptions.fragmentCharSize = globalOptions.fragmentCharSize;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.numberOfFragments == -1) {
                    fieldOptions.numberOfFragments = globalOptions.numberOfFragments;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.encoder == null) {
                    fieldOptions.encoder = globalOptions.encoder;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.requireFieldMatch == null) {
                    fieldOptions.requireFieldMatch = globalOptions.requireFieldMatch;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.maxAnalyzedOffset == null) {
                    fieldOptions.maxAnalyzedOffset = globalOptions.maxAnalyzedOffset;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.boundaryScannerType == null) {
                    fieldOptions.boundaryScannerType = globalOptions.boundaryScannerType;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.boundaryMaxScan == -1) {
                    fieldOptions.boundaryMaxScan = globalOptions.boundaryMaxScan;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.boundaryChars == null && globalOptions.boundaryChars != null) {
                    fieldOptions.boundaryChars = Arrays.copyOf(globalOptions.boundaryChars, globalOptions.boundaryChars.length);
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.boundaryScannerLocale == null) {
                    fieldOptions.boundaryScannerLocale = globalOptions.boundaryScannerLocale;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.highlighterType == null) {
                    fieldOptions.highlighterType = globalOptions.highlighterType;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.fragmenter == null) {
                    fieldOptions.fragmenter = globalOptions.fragmenter;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if ((fieldOptions.options == null || fieldOptions.options.size() == 0) && globalOptions.options != null) {
                    fieldOptions.options = new HashMap<>(globalOptions.options);
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.highlightQuery == null && globalOptions.highlightQuery != null) {
                    fieldOptions.highlightQuery = globalOptions.highlightQuery;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.noMatchSize == -1) {
                    fieldOptions.noMatchSize = globalOptions.noMatchSize;
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (fieldOptions.phraseLimit == -1) {
                    fieldOptions.phraseLimit = globalOptions.phraseLimit;
                }
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
                return this;
            }
        }
    }
}
