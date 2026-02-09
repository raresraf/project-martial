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

    public SearchHighlightContext(Collection<Field> fields) {
        assert fields != null;
        this.fields = Maps.newLinkedHashMapWithExpectedSize(fields.size());
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (Field field : fields) {
            this.fields.put(field.field, field);
        }
    }

    public Collection<Field> fields() {
        return fields.values();
    }

    public static class Field {
        private final String field;
        private final FieldOptions fieldOptions;

        Field(String field, FieldOptions fieldOptions) {
            assert field != null;
            assert fieldOptions != null;
            this.field = field;
            this.fieldOptions = fieldOptions;
        }

        public String field() {
            return field;
        }

        public FieldOptions fieldOptions() {
            return fieldOptions;
        }
    }

    public static class FieldOptions {

        // Field options that default to null or -1 are often set to their real default in HighlighterParseElement#parse
        private int fragmentCharSize = -1;

        private int numberOfFragments = -1;

        private int fragmentOffset = -1;

        private String encoder;

        private String[] preTags;

        private String[] postTags;

        private Boolean scoreOrdered;

        private Boolean highlightFilter;

        private Boolean requireFieldMatch;

        private Integer maxAnalyzedOffset;

        private String highlighterType;

        private String fragmenter;

        private BoundaryScannerType boundaryScannerType;

        private int boundaryMaxScan = -1;

        private char[] boundaryChars = null;

        private Locale boundaryScannerLocale;

        private Query highlightQuery;

        private int noMatchSize = -1;

        private Set<String> matchedFields;

        private Map<String, Object> options;

        private int phraseLimit = -1;

        public int fragmentCharSize() {
            return fragmentCharSize;
        }

        public int numberOfFragments() {
            return numberOfFragments;
        }

        public int fragmentOffset() {
            return fragmentOffset;
        }

        public String encoder() {
            return encoder;
        }

        public String[] preTags() {
            return preTags;
        }

        public String[] postTags() {
            return postTags;
        }

        public Boolean scoreOrdered() {
            return scoreOrdered;
        }

        public Boolean highlightFilter() {
            return highlightFilter;
        }

        public Boolean requireFieldMatch() {
            return requireFieldMatch;
        }

        public Integer maxAnalyzedOffset() {
            return maxAnalyzedOffset;
        }

        public String highlighterType() {
            return highlighterType;
        }

        public String fragmenter() {
            return fragmenter;
        }

        public BoundaryScannerType boundaryScannerType() {
            return boundaryScannerType;
        }

        public int boundaryMaxScan() {
            return boundaryMaxScan;
        }

        public char[] boundaryChars() {
            return boundaryChars;
        }

        public Locale boundaryScannerLocale() {
            return boundaryScannerLocale;
        }

        public Query highlightQuery() {
            return highlightQuery;
        }

        public int noMatchSize() {
            return noMatchSize;
        }

        public int phraseLimit() {
            return phraseLimit;
        }

        public Set<String> matchedFields() {
            return matchedFields;
        }

        public Map<String, Object> options() {
            return options;
        }

        static class Builder {

            private final FieldOptions fieldOptions = new FieldOptions();

            Builder fragmentCharSize(int fragmentCharSize) {
                fieldOptions.fragmentCharSize = fragmentCharSize;
                return this;
            }

            Builder numberOfFragments(int numberOfFragments) {
                fieldOptions.numberOfFragments = numberOfFragments;
                return this;
            }

            Builder fragmentOffset(int fragmentOffset) {
                fieldOptions.fragmentOffset = fragmentOffset;
                return this;
            }

            Builder encoder(String encoder) {
                fieldOptions.encoder = encoder;
                return this;
            }

            Builder preTags(String[] preTags) {
                fieldOptions.preTags = preTags;
                return this;
            }

            Builder postTags(String[] postTags) {
                fieldOptions.postTags = postTags;
                return this;
            }

            Builder scoreOrdered(boolean scoreOrdered) {
                fieldOptions.scoreOrdered = scoreOrdered;
                return this;
            }

            Builder highlightFilter(boolean highlightFilter) {
                fieldOptions.highlightFilter = highlightFilter;
                return this;
            }

            Builder requireFieldMatch(boolean requireFieldMatch) {
                fieldOptions.requireFieldMatch = requireFieldMatch;
                return this;
            }

            Builder maxAnalyzedOffset(Integer maxAnalyzedOffset) {
                fieldOptions.maxAnalyzedOffset = maxAnalyzedOffset;
                return this;
            }

            Builder highlighterType(String type) {
                fieldOptions.highlighterType = type;
                return this;
            }

            Builder fragmenter(String fragmenter) {
                fieldOptions.fragmenter = fragmenter;
                return this;
            }

            Builder boundaryScannerType(BoundaryScannerType boundaryScanner) {
                fieldOptions.boundaryScannerType = boundaryScanner;
                return this;
            }

            Builder boundaryMaxScan(int boundaryMaxScan) {
                fieldOptions.boundaryMaxScan = boundaryMaxScan;
                return this;
            }

            Builder boundaryChars(char[] boundaryChars) {
                fieldOptions.boundaryChars = boundaryChars;
                return this;
            }

            Builder boundaryScannerLocale(Locale boundaryScannerLocale) {
                fieldOptions.boundaryScannerLocale = boundaryScannerLocale;
                return this;
            }

            Builder highlightQuery(Query highlightQuery) {
                fieldOptions.highlightQuery = highlightQuery;
                return this;
            }

            Builder noMatchSize(int noMatchSize) {
                fieldOptions.noMatchSize = noMatchSize;
                return this;
            }

            Builder phraseLimit(int phraseLimit) {
                fieldOptions.phraseLimit = phraseLimit;
                return this;
            }

            Builder matchedFields(Set<String> matchedFields) {
                fieldOptions.matchedFields = matchedFields;
                return this;
            }

            Builder options(Map<String, Object> options) {
                fieldOptions.options = options;
                return this;
            }

            FieldOptions build() {
                return fieldOptions;
            }

            Builder merge(FieldOptions globalOptions) {
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.preTags == null && globalOptions.preTags != null) {
                    fieldOptions.preTags = Arrays.copyOf(globalOptions.preTags, globalOptions.preTags.length);
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.postTags == null && globalOptions.postTags != null) {
                    fieldOptions.postTags = Arrays.copyOf(globalOptions.postTags, globalOptions.postTags.length);
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.highlightFilter == null) {
                    fieldOptions.highlightFilter = globalOptions.highlightFilter;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.scoreOrdered == null) {
                    fieldOptions.scoreOrdered = globalOptions.scoreOrdered;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.fragmentCharSize == -1) {
                    fieldOptions.fragmentCharSize = globalOptions.fragmentCharSize;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.numberOfFragments == -1) {
                    fieldOptions.numberOfFragments = globalOptions.numberOfFragments;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.encoder == null) {
                    fieldOptions.encoder = globalOptions.encoder;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.requireFieldMatch == null) {
                    fieldOptions.requireFieldMatch = globalOptions.requireFieldMatch;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.maxAnalyzedOffset == null) {
                    fieldOptions.maxAnalyzedOffset = globalOptions.maxAnalyzedOffset;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.boundaryScannerType == null) {
                    fieldOptions.boundaryScannerType = globalOptions.boundaryScannerType;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.boundaryMaxScan == -1) {
                    fieldOptions.boundaryMaxScan = globalOptions.boundaryMaxScan;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.boundaryChars == null && globalOptions.boundaryChars != null) {
                    fieldOptions.boundaryChars = Arrays.copyOf(globalOptions.boundaryChars, globalOptions.boundaryChars.length);
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.boundaryScannerLocale == null) {
                    fieldOptions.boundaryScannerLocale = globalOptions.boundaryScannerLocale;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.highlighterType == null) {
                    fieldOptions.highlighterType = globalOptions.highlighterType;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.fragmenter == null) {
                    fieldOptions.fragmenter = globalOptions.fragmenter;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if ((fieldOptions.options == null || fieldOptions.options.size() == 0) && globalOptions.options != null) {
                    fieldOptions.options = new HashMap<>(globalOptions.options);
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.highlightQuery == null && globalOptions.highlightQuery != null) {
                    fieldOptions.highlightQuery = globalOptions.highlightQuery;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.noMatchSize == -1) {
                    fieldOptions.noMatchSize = globalOptions.noMatchSize;
                }
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (fieldOptions.phraseLimit == -1) {
                    fieldOptions.phraseLimit = globalOptions.phraseLimit;
                }
                return this;
            }
        }
    }
}
