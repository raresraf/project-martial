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
import org.apache.lucene.search.vectorhighlight.SimpleBoundaryScanner;
import org.elasticsearch.TransportVersions;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.common.util.set.Sets;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.index.query.Rewriteable;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.search.fetch.subphase.highlight.SearchHighlightContext.FieldOptions;
import org.elasticsearch.xcontent.ObjectParser;
import org.elasticsearch.xcontent.ObjectParser.NamedObjectParser;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiFunction;

import static org.elasticsearch.xcontent.ObjectParser.fromList;

/**
 * A builder for search highlighting. Settings can control how large fields
 * are summarized to show only selected snippets ("fragments") containing search terms.
 *
 * This class extends {@link AbstractHighlighterBuilder} to provide a flexible and comprehensive
 * way to define highlighting behavior for various fields in Elasticsearch search results.
 * It encapsulates options related to fragment size, number of fragments, pre/post tags,
 * boundary scanning, and more, allowing users to customize how matched text is presented.
 *
 * @see org.elasticsearch.search.builder.SearchSourceBuilder#highlight()
 */
public final class HighlightBuilder extends AbstractHighlighterBuilder<HighlightBuilder> {
    /** default for whether a field should be highlighted only if a query matches that field */
    public static final boolean DEFAULT_REQUIRE_FIELD_MATCH = true;
    /** default for whether to stop highlighting at the defined max_analyzed_offset to avoid exceptions for longer texts */
    public static final Integer DEFAULT_MAX_ANALYZED_OFFSET = null;
    /** default for whether {@code fvh} should provide highlighting on filter clauses */
    public static final boolean DEFAULT_HIGHLIGHT_FILTER = false;
    /** default for highlight fragments being ordered by score */
    public static final boolean DEFAULT_SCORE_ORDERED = false;
    /** the default encoder setting */
    public static final String DEFAULT_ENCODER = "default";
    /** default for the maximum number of phrases the fvh will consider */
    public static final int DEFAULT_PHRASE_LIMIT = 256;
    /** default for fragment size when there are no matches */
    public static final int DEFAULT_NO_MATCH_SIZE = 0;
    /** the default number of fragments for highlighting */
    public static final int DEFAULT_NUMBER_OF_FRAGMENTS = 5;
    /** the default number of fragments size in characters */
    public static final int DEFAULT_FRAGMENT_CHAR_SIZE = 100;
    /** the default opening tag  */
    static final String[] DEFAULT_PRE_TAGS = new String[] { "<em>" };
    /** the default closing tag  */
    static final String[] DEFAULT_POST_TAGS = new String[] { "</em>" };

    /** the default opening tags when {@code tag_schema = "styled"}  */
    public static final String[] DEFAULT_STYLED_PRE_TAG = {
        "<em class=\"hlt1\">",
        "<em class=\"hlt2\">",
        "<em class=\"hlt3\">",
        "<em class=\"hlt4\">",
        "<em class=\"hlt5\">",
        "<em class=\"hlt6\">",
        "<em class=\"hlt7\">",
        "<em class=\"hlt8\">",
        "<em class=\"hlt9\">",
        "<em class=\"hlt10\">" };
    /** the default closing tags when {@code tag_schema = "styled"}  */
    public static final String[] DEFAULT_STYLED_POST_TAGS = { "</em>" };

    /**
     * a {@link FieldOptions} with default settings
     */
    static final FieldOptions defaultOptions = new SearchHighlightContext.FieldOptions.Builder().preTags(DEFAULT_PRE_TAGS)
        .postTags(DEFAULT_POST_TAGS)
        .scoreOrdered(DEFAULT_SCORE_ORDERED)
        .highlightFilter(DEFAULT_HIGHLIGHT_FILTER)
        .requireFieldMatch(DEFAULT_REQUIRE_FIELD_MATCH)
        .maxAnalyzedOffset(DEFAULT_MAX_ANALYZED_OFFSET)
        .fragmentCharSize(DEFAULT_FRAGMENT_CHAR_SIZE)
        .numberOfFragments(DEFAULT_NUMBER_OF_FRAGMENTS)
        .encoder(DEFAULT_ENCODER)
        .boundaryMaxScan(SimpleBoundaryScanner.DEFAULT_MAX_SCAN)
        .boundaryChars(SimpleBoundaryScanner.DEFAULT_BOUNDARY_CHARS)
        .boundaryScannerLocale(Locale.ROOT)
        .noMatchSize(DEFAULT_NO_MATCH_SIZE)
        .phraseLimit(DEFAULT_PHRASE_LIMIT)
        .build();

    private final List<Field> fields;

    private boolean useExplicitFieldOrder = false;

    public HighlightBuilder() {
        // Functional Utility: Initializes a new, empty HighlightBuilder instance.
        // The list of fields to be highlighted is initialized as an empty ArrayList.
        fields = new ArrayList<>();
    }

    public HighlightBuilder(HighlightBuilder template, QueryBuilder highlightQuery, List<Field> fields) {
        // Functional Utility: Initializes a HighlightBuilder by copying settings from a template.
        // This constructor is used during the rewrite process to create a new builder with potentially rewritten queries or fields.
        super(template, highlightQuery);
        this.useExplicitFieldOrder = template.useExplicitFieldOrder;
        this.fields = fields;
    }

    /**
     * Read from a stream.
     */
    public HighlightBuilder(StreamInput in) throws IOException {
        // Functional Utility: Reconstructs a HighlightBuilder instance from a StreamInput.
        // This constructor is used for deserialization, reading various highlighting options from the stream.
        super(in);
        // Conditional Logic: Read encoder if transport version is before 8.14.0.
        if (in.getTransportVersion().before(TransportVersions.V_8_14_0)) {
            encoder(in.readOptionalString());
        }
        useExplicitFieldOrder(in.readBoolean());
        this.fields = in.readCollectionAsList(Field::new);
        // Block Logic: Assertion to ensure copy constructor works correctly after deserialization.
        assert this.equals(new HighlightBuilder(this, highlightQuery, fields)) : "copy constructor is broken";
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        // Functional Utility: Writes the state of the HighlightBuilder to a StreamOutput.
        // This method is used for serializing the object for network transfer or persistence,
        // specifically handling version-dependent writing of the encoder and other fields.
        // Conditional Logic: Write encoder if transport version is before 8.14.0.
        if (out.getTransportVersion().before(TransportVersions.V_8_14_0)) {
            out.writeOptionalString(encoder);
        }
        out.writeBoolean(useExplicitFieldOrder);
        out.writeCollection(fields);
    }

    /**
     * Adds a field to be highlighted with default fragment size of 100 characters, and
     * default number of fragments of 5 using the default encoder
     *
     * @param name The field to highlight
     */
    public HighlightBuilder field(String name) {
        // Functional Utility: Adds a new field to the highlighting configuration with default options.
        return field(new Field(name));
    }

    /**
     * Adds a field to be highlighted with a provided fragment size (in characters), and
     * a provided (maximum) number of fragments.
     *
     * @param name              The field to highlight
     * @param fragmentSize      The size of a fragment in characters
     * @param numberOfFragments The (maximum) number of fragments
     */
    public HighlightBuilder field(String name, int fragmentSize, int numberOfFragments) {
        // Functional Utility: Adds a new field to the highlighting configuration with specified fragment size and number of fragments.
        return field(new Field(name).fragmentSize(fragmentSize).numOfFragments(numberOfFragments));
    }

    /**
     * Adds a field to be highlighted with a provided fragment size (in characters), and
     * a provided (maximum) number of fragments.
     *
     * @param name              The field to highlight
     * @param fragmentSize      The size of a fragment in characters
     * @param numberOfFragments The (maximum) number of fragments
     * @param fragmentOffset    The offset from the start of the fragment to the start of the highlight
     */
    public HighlightBuilder field(String name, int fragmentSize, int numberOfFragments, int fragmentOffset) {
        // Functional Utility: Adds a new field to the highlighting configuration with specified fragment size, number of fragments, and fragment offset.
        return field(new Field(name).fragmentSize(fragmentSize).numOfFragments(numberOfFragments).fragmentOffset(fragmentOffset));
    }

    public HighlightBuilder field(Field field) {
        // Functional Utility: Adds a pre-configured Field object to the list of fields to be highlighted.
        fields.add(field);
        return this;
    }

    void fields(List<Field> fields) {
        // Functional Utility: Adds a collection of pre-configured Field objects to the list of fields to be highlighted.
        this.fields.addAll(fields);
    }

    public List<Field> fields() {
        // Functional Utility: Returns the list of Field objects configured for highlighting.
        return this.fields;
    }

    /**
     * Send the fields to be highlighted using a syntax that is specific about the order in which they should be highlighted.
     * @return this for chaining
     */
    public HighlightBuilder useExplicitFieldOrder(boolean useExplicitFieldOrder) {
        // Functional Utility: Sets whether the order of fields in the highlighting configuration should be explicitly maintained.
        this.useExplicitFieldOrder = useExplicitFieldOrder;
        return this;
    }

    /**
     * Gets value set with {@link #useExplicitFieldOrder(boolean)}
     */
    public Boolean useExplicitFieldOrder() {
        // Functional Utility: Returns whether explicit field order is enabled for highlighting.
        return this.useExplicitFieldOrder;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        // Functional Utility: Converts the HighlightBuilder instance into XContent (e.g., JSON).
        // This method writes the highlighting configuration, including common and field-specific options.
        builder.startObject();
        innerXContent(builder);
        builder.endObject();
        return builder;
    }

    private static final BiFunction<XContentParser, HighlightBuilder, HighlightBuilder> PARSER;
    static {
        // Block Logic: Initialize ObjectParser for HighlightBuilder, declaring how to parse fields and set properties.
        ObjectParser<HighlightBuilder, Void> parser = new ObjectParser<>("highlight");
        // Block Logic: Declare parsing for highlight fields, setting explicit field order if multiple fields are present.
        parser.declareNamedObjects(
            HighlightBuilder::fields,
            Field.PARSER,
            (HighlightBuilder hb) -> hb.useExplicitFieldOrder(true),
            FIELDS_FIELD
        );
        PARSER = setupParser(parser);
    }

    public static HighlightBuilder fromXContent(XContentParser p) {
        // Functional Utility: Creates a HighlightBuilder instance from XContent parsed by an XContentParser.
        return PARSER.apply(p, new HighlightBuilder());
    }

    public SearchHighlightContext build(SearchExecutionContext context) throws IOException {
        // Functional Utility: Constructs a SearchHighlightContext from the HighlightBuilder.
        // This context contains all the necessary options for performing highlighting during a search.
        // create template global options that are later merged with any partial field options
        final SearchHighlightContext.FieldOptions.Builder globalOptionsBuilder = new SearchHighlightContext.FieldOptions.Builder();

        transferOptions(this, globalOptionsBuilder, context);

        // overwrite unset global options by default values
        globalOptionsBuilder.merge(defaultOptions);

        // create field options
        // Block Logic: Iterate through configured fields and build their individual highlighting options.
        Collection<SearchHighlightContext.Field> fieldOptions = new ArrayList<>();
        for (Field field : this.fields) {
            final SearchHighlightContext.FieldOptions.Builder fieldOptionsBuilder = new SearchHighlightContext.FieldOptions.Builder();
            fieldOptionsBuilder.fragmentOffset(field.fragmentOffset);
            // Conditional Logic: Handle matched fields.
            if (field.matchedFields != null) {
                Set<String> matchedFields = Sets.newHashSetWithExpectedSize(field.matchedFields.length);
                Collections.addAll(matchedFields, field.matchedFields);
                fieldOptionsBuilder.matchedFields(matchedFields);
            }
            transferOptions(field, fieldOptionsBuilder, context);
            fieldOptions.add(
                new SearchHighlightContext.Field(field.name(), fieldOptionsBuilder.merge(globalOptionsBuilder.build()).build())
            );
        }
        // Block Logic: Return a new SearchHighlightContext with the compiled field options.
        return new SearchHighlightContext(fieldOptions);
    }

    /**
     * Transfers field options present in the input {@link AbstractHighlighterBuilder} to the receiving
     * {@link FieldOptions.Builder}, effectively overwriting existing settings
     * @param targetOptionsBuilder the receiving options builder
     * @param highlighterBuilder highlight builder with the input options
     * @param context needed to convert {@link QueryBuilder} to {@link Query}
     * @throws IOException on errors parsing any optional nested highlight query
     */
    @SuppressWarnings({ "rawtypes", "unchecked" })
    private static void transferOptions(
        AbstractHighlighterBuilder highlighterBuilder,
        SearchHighlightContext.FieldOptions.Builder targetOptionsBuilder,
        SearchExecutionContext context
    ) throws IOException {
        // Block Logic: Conditionally transfer various highlighting options from the source builder to the target builder.
        // This ensures that specific settings from the HighlightBuilder (or its Field sub-builders)
        // are applied to the FieldOptions.Builder, overriding global defaults if set.
        if (highlighterBuilder.preTags != null) {
            targetOptionsBuilder.preTags(highlighterBuilder.preTags);
        }
        if (highlighterBuilder.postTags != null) {
            targetOptionsBuilder.postTags(highlighterBuilder.postTags);
        }
        if (highlighterBuilder.order != null) {
            targetOptionsBuilder.scoreOrdered(highlighterBuilder.order == Order.SCORE);
        }
        if (highlighterBuilder.highlightFilter != null) {
            targetOptionsBuilder.highlightFilter(highlighterBuilder.highlightFilter);
        }
        if (highlighterBuilder.fragmentSize != null) {
            targetOptionsBuilder.fragmentCharSize(highlighterBuilder.fragmentSize);
        }
        if (highlighterBuilder.numOfFragments != null) {
            targetOptionsBuilder.numberOfFragments(highlighterBuilder.numOfFragments);
        }
        if (highlighterBuilder.encoder != null) {
            targetOptionsBuilder.encoder(highlighterBuilder.encoder);
        }
        if (highlighterBuilder.requireFieldMatch != null) {
            targetOptionsBuilder.requireFieldMatch(highlighterBuilder.requireFieldMatch);
        }
        if (highlighterBuilder.maxAnalyzedOffset != null) {
            targetOptionsBuilder.maxAnalyzedOffset(highlighterBuilder.maxAnalyzedOffset);
        }
        if (highlighterBuilder.boundaryScannerType != null) {
            targetOptionsBuilder.boundaryScannerType(highlighterBuilder.boundaryScannerType);
        }
        if (highlighterBuilder.boundaryMaxScan != null) {
            targetOptionsBuilder.boundaryMaxScan(highlighterBuilder.boundaryMaxScan);
        }
        if (highlighterBuilder.boundaryChars != null) {
            targetOptionsBuilder.boundaryChars(highlighterBuilder.boundaryChars);
        }
        if (highlighterBuilder.boundaryScannerLocale != null) {
            targetOptionsBuilder.boundaryScannerLocale(highlighterBuilder.boundaryScannerLocale);
        }
        if (highlighterBuilder.highlighterType != null) {
            targetOptionsBuilder.highlighterType(highlighterBuilder.highlighterType);
        }
        if (highlighterBuilder.fragmenter != null) {
            targetOptionsBuilder.fragmenter(highlighterBuilder.fragmenter);
        }
        if (highlighterBuilder.noMatchSize != null) {
            targetOptionsBuilder.noMatchSize(highlighterBuilder.noMatchSize);
        }
        if (highlighterBuilder.phraseLimit != null) {
            targetOptionsBuilder.phraseLimit(highlighterBuilder.phraseLimit);
        }
        if (highlighterBuilder.options != null) {
            targetOptionsBuilder.options(highlighterBuilder.options);
        }
        if (highlighterBuilder.highlightQuery != null) {
            targetOptionsBuilder.highlightQuery(highlighterBuilder.highlightQuery.toQuery(context));
        }
    }

    @Override
    public void innerXContent(XContentBuilder builder) throws IOException {
        // Functional Utility: Writes the internal XContent representation of the HighlightBuilder.
        // This method handles writing common highlighting options and then iterates through
        // the configured fields, writing their individual XContent.
        // first write common options
        commonOptionsToXContent(builder);
        // special options for top-level highlighter
        // Conditional Logic: Write fields as an array if explicit order is used, otherwise as an object.
        if (fields.size() > 0) {
            if (useExplicitFieldOrder) {
                builder.startArray(FIELDS_FIELD.getPreferredName());
            } else {
                builder.startObject(FIELDS_FIELD.getPreferredName());
            }
            // Block Logic: Iterate through fields and write their XContent.
            for (Field field : fields) {
                // Conditional Logic: Start/end object for each field if explicit order is used.
                if (useExplicitFieldOrder) {
                    builder.startObject();
                }
                field.innerXContent(builder);
                if (useExplicitFieldOrder) {
                    builder.endObject();
                }
            }
            if (useExplicitFieldOrder) {
                builder.endArray();
            } else {
                builder.endObject();
            }
        }
    }

    @Override
    protected int doHashCode() {
        // Functional Utility: Computes a subclass-specific hash code for HighlightBuilder.
        // It combines the hash codes of `useExplicitFieldOrder` and the list of `fields`.
        return Objects.hash(useExplicitFieldOrder, fields);
    }

    @Override
    protected boolean doEquals(HighlightBuilder other) {
        // Functional Utility: Compares subclass-specific fields of two HighlightBuilder instances for equality.
        // It checks the equality of `useExplicitFieldOrder` and the lists of `fields`.
        return Objects.equals(useExplicitFieldOrder, other.useExplicitFieldOrder) && Objects.equals(fields, other.fields);
    }

    @Override
    public HighlightBuilder rewrite(QueryRewriteContext ctx) throws IOException {
        // Functional Utility: Rewrites the HighlightBuilder, particularly its highlight query and fields.
        // This is crucial for optimizing queries and ensuring they are in their final executable form.
        QueryBuilder highlightQuery = this.highlightQuery;
        // Conditional Logic: Rewrite highlightQuery if present.
        if (highlightQuery != null) {
            highlightQuery = this.highlightQuery.rewrite(ctx);
        }
        // Block Logic: Rewrite the list of fields.
        List<Field> fields = Rewriteable.rewrite(this.fields, ctx);
        // Conditional Logic: If no changes occurred during rewrite, return the current instance.
        if (highlightQuery == this.highlightQuery && fields == this.fields) {
            return this;
        }
        // Block Logic: Otherwise, return a new HighlightBuilder with the rewritten components.
        return new HighlightBuilder(this, highlightQuery, fields);
    }

    /**
     * @file HighlightBuilder.java
     * @brief Inner class representing a field to be highlighted in search results.
     * This class extends {@link AbstractHighlighterBuilder} to manage field-specific highlighting options,
     * such as fragment offset and matched fields.
     */
    public static final class Field extends AbstractHighlighterBuilder<Field> {
        static final NamedObjectParser<Field, Void> PARSER;
        static {
            // Block Logic: Initialize ObjectParser for Field, declaring how to parse fragmentOffset and matchedFields.
            ObjectParser<Field, Void> parser = new ObjectParser<>("highlight_field");
            parser.declareInt(Field::fragmentOffset, FRAGMENT_OFFSET_FIELD);
            parser.declareStringArray(fromList(String.class, Field::matchedFields), MATCHED_FIELDS_FIELD);
            BiFunction<XContentParser, Field, Field> decoratedParser = setupParser(parser);
            PARSER = (XContentParser p, Void c, String name) -> decoratedParser.apply(p, new Field(name));
        }

        private final String name;

        int fragmentOffset = -1;

        String[] matchedFields;

        public Field(String name) {
            // Functional Utility: Initializes a new Field instance with the given name.
            this.name = name;
        }

        Field(Field template, QueryBuilder builder) {
            // Functional Utility: Initializes a Field instance by copying settings from a template.
            // This constructor is used during the rewrite process to create a new field with a potentially rewritten highlight query.
            super(template, builder);
            name = template.name;
            fragmentOffset = template.fragmentOffset;
            matchedFields = template.matchedFields;
        }

        /**
         * Read from a stream.
         */
        public Field(StreamInput in) throws IOException {
            // Functional Utility: Reconstructs a Field instance from a StreamInput.
            // This constructor is used for deserialization, reading field-specific highlighting options from the stream.
            super(in);
            name = in.readString();
            fragmentOffset(in.readVInt());
            matchedFields(in.readOptionalStringArray());
            // Block Logic: Assertion to ensure copy constructor works correctly after deserialization.
            assert this.equals(new Field(this, highlightQuery)) : "copy constructor is broken";
        }

        @Override
        protected void doWriteTo(StreamOutput out) throws IOException {
            // Functional Utility: Writes the state of the Field to a StreamOutput.
            // This method serializes the field's name, fragment offset, and matched fields.
            out.writeString(name);
            out.writeVInt(fragmentOffset);
            out.writeOptionalStringArray(matchedFields);
        }

        public String name() {
            // Functional Utility: Returns the name of the field.
            return name;
        }

        public Field fragmentOffset(int fragmentOffset) {
            // Functional Utility: Sets the fragment offset for this field.
            this.fragmentOffset = fragmentOffset;
            return this;
        }

        /**
         * Set the matched fields to highlight against this field data.  Default to null, meaning just
         * the named field.  If you provide a list of fields here then don't forget to include name as
         * it is not automatically included.
         */
        public Field matchedFields(String... matchedFields) {
            // Functional Utility: Sets the array of fields whose matches should be highlighted within this field.
            this.matchedFields = matchedFields;
            return this;
        }

        @Override
        public void innerXContent(XContentBuilder builder) throws IOException {
            // Functional Utility: Writes the internal XContent representation of this Field.
            // This method outputs the field's name, common highlighting options, and field-specific options.
            builder.startObject(name);
            // write common options
            commonOptionsToXContent(builder);
            // write special field-highlighter options
            // Conditional Logic: Write fragment offset if it's not the default value.
            if (fragmentOffset != -1) {
                builder.field(FRAGMENT_OFFSET_FIELD.getPreferredName(), fragmentOffset);
            }
            // Conditional Logic: Write matched fields if present.
            if (matchedFields != null) {
                builder.array(MATCHED_FIELDS_FIELD.getPreferredName(), matchedFields);
            }
            builder.endObject();
        }

        @Override
        protected int doHashCode() {
            // Functional Utility: Computes a subclass-specific hash code for Field.
            // It combines the hash codes of `name`, `fragmentOffset`, and `matchedFields`.
            return Objects.hash(name, fragmentOffset, Arrays.hashCode(matchedFields));
        }

        @Override
        protected boolean doEquals(Field other) {
            // Functional Utility: Compares subclass-specific fields of two Field instances for equality.
            // It checks the equality of `name`, `fragmentOffset`, and `matchedFields`.
            return Objects.equals(name, other.name)
                && Objects.equals(fragmentOffset, other.fragmentOffset)
                && Arrays.equals(matchedFields, other.matchedFields);
        }

        @Override
        public Field rewrite(QueryRewriteContext ctx) throws IOException {
            // Functional Utility: Rewrites the Field instance, particularly its highlight query.
            // This ensures that any embedded query is optimized or transformed for execution.
            // Conditional Logic: Rewrite highlightQuery if present.
            if (highlightQuery != null) {
                QueryBuilder rewrite = highlightQuery.rewrite(ctx);
                // Conditional Logic: If the highlight query was rewritten, return a new Field instance.
                if (rewrite != highlightQuery) {
                    return new Field(this, rewrite);
                }
            }
            // Block Logic: If no changes, return the current instance.
            return this;
        }
    }

    /**
     * @file HighlightBuilder.java
     * @brief Enum representing the order in which highlight fragments should be returned.
     * Fragments can be ordered by their score or returned in their natural order.
     */
    public enum Order implements Writeable {
        NONE,
        SCORE;

        /**
         * Functional Utility: Reads an Order enum value from a StreamInput.
         * @param in The StreamInput to read from.
         * @return The deserialized Order enum value.
         * @throws IOException If an I/O error occurs.
         */
        public static Order readFromStream(StreamInput in) throws IOException {
            return in.readEnum(Order.class);
        }

        /**
         * Functional Utility: Writes the Order enum value to a StreamOutput.
         * @param out The StreamOutput to write to.
         * @throws IOException If an I/O error occurs.
         */
        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeEnum(this);
        }

        /**
         * Functional Utility: Converts a string to an Order enum value.
         * It specifically handles "score" as a valid option for SCORE.
         * @param order The string representation of the order.
         * @return The corresponding Order enum value.
         */
        public static Order fromString(String order) {
            if (order.toUpperCase(Locale.ROOT).equals(SCORE.name())) {
                return Order.SCORE;
            }
            return NONE;
        }

        /**
         * Functional Utility: Returns the lowercase string representation of the Order enum value.
         * @return The lowercase string representation.
         */
        @Override
        public String toString() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /**
     * @file HighlightBuilder.java
     * @brief Enum representing the type of boundary scanner to use for highlighting.
     * This defines how fragments are delimited (e.g., by characters, words, or sentences).
     */
    public enum BoundaryScannerType implements Writeable {
        CHARS,
        WORD,
        SENTENCE;

        /**
         * Functional Utility: Reads a BoundaryScannerType enum value from a StreamInput.
         * @param in The StreamInput to read from.
         * @return The deserialized BoundaryScannerType enum value.
         * @throws IOException If an I/O error occurs.
         */
        public static BoundaryScannerType readFromStream(StreamInput in) throws IOException {
            return in.readEnum(BoundaryScannerType.class);
        }

        /**
         * Functional Utility: Writes the BoundaryScannerType enum value to a StreamOutput.
         * @param out The StreamOutput to write to.
         * @throws IOException If an I/O error occurs.
         */
        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeEnum(this);
        }

        /**
         * Functional Utility: Converts a string to a BoundaryScannerType enum value.
         * @param boundaryScannerType The string representation of the boundary scanner type.
         * @return The corresponding BoundaryScannerType enum value.
         */
        public static BoundaryScannerType fromString(String boundaryScannerType) {
            return valueOf(boundaryScannerType.toUpperCase(Locale.ROOT));
        }

        /**
         * Functional Utility: Returns the lowercase string representation of the BoundaryScannerType enum value.
         * @return The lowercase string representation.
         */
        @Override
        public String toString() {
            return name().toLowerCase(Locale.ROOT);
        }
    }
}
}
