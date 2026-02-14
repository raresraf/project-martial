/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.common.xcontent;

import org.elasticsearch.ElasticsearchGenerationException;
import org.elasticsearch.ElasticsearchParseException;
import org.elasticsearch.TransportVersions;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.compress.Compressor;
import org.elasticsearch.common.compress.CompressorFactory;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.core.CheckedFunction;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.core.RestApiVersion;
import org.elasticsearch.core.Tuple;
import org.elasticsearch.plugins.internal.XContentParserDecorator;
import org.elasticsearch.xcontent.DeprecationHandler;
import org.elasticsearch.xcontent.NamedXContentRegistry;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.ToXContent.Params;
import org.elasticsearch.xcontent.XContent;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentParseException;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xcontent.XContentParserConfiguration;
import org.elasticsearch.xcontent.XContentType;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * @05314305-b286-4c5b-a80e-5c46defa6a97/arch/arm/crypto/Makefile XContentHelper.java
 * @brief Provides a collection of utility methods for handling and manipulating XContent (a flexible data format used in Elasticsearch).
 * This class facilitates parsing, content type detection, conversion between different XContent representations (e.g., bytes to maps,
 * bytes to JSON), and sophisticated merging operations for map structures, including handling of compressed data.
 * @description This class is central to how Elasticsearch processes and transforms data internally and externally,
 * ensuring consistency and flexibility across various data formats like JSON, YAML, and Smile.
 */
@SuppressWarnings("unchecked")
public class XContentHelper {

    /**
     * Creates an {@link XContentParser} based on the provided {@link BytesReference}.
     * This method attempts to auto-detect the content type, including handling compressed data.
     * @param registry A {@link NamedXContentRegistry} to use for named content parsing.
     * @param deprecation A {@link DeprecationHandler} for managing deprecated XContent features.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @return An {@link XContentParser} instance configured for the detected content type.
     * @throws IOException If an I/O error occurs during parser creation or content type detection.
     * @deprecated Use {@link #createParser(XContentParserConfiguration, BytesReference, XContentType)}
     * to explicitly specify the content type and avoid auto-detection, which can be less efficient and potentially ambiguous.
     */
    @Deprecated
    public static XContentParser createParser(NamedXContentRegistry registry, DeprecationHandler deprecation, BytesReference bytes)
        throws IOException {
        // Delegates to a more generalized parser creation method, supplying an empty configuration
        // augmented with the provided registry and deprecation handler.
        return createParser(XContentParserConfiguration.EMPTY.withRegistry(registry).withDeprecationHandler(deprecation), bytes);
    }

    /**
     * Creates an {@link XContentParser} based on the provided {@link BytesReference} and {@link XContentParserConfiguration}.
     * This method handles decompression if the bytes are compressed and attempts to auto-detect the content type.
     * @param config The {@link XContentParserConfiguration} to use for parser settings.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @return An {@link XContentParser} instance configured for the detected content type.
     * @throws IOException If an I/O error occurs during parser creation, decompression, or content type detection.
     * @deprecated Use {@link #createParser(XContentParserConfiguration, BytesReference, XContentType)}
     * to explicitly specify the content type and avoid auto-detection, which can be less efficient and potentially ambiguous.
     */
    @Deprecated
    public static XContentParser createParser(XContentParserConfiguration config, BytesReference bytes) throws IOException {
        // Attempts to find a compressor for the given bytes, indicating if the data is compressed.
        Compressor compressor = CompressorFactory.compressorForUnknownXContentType(bytes);
        if (compressor != null) {
            // If a compressor is found, decompress the input stream.
            InputStream compressedInput = compressor.threadLocalInputStream(bytes.streamInput());
            // Ensure the input stream supports marking for content type detection.
            if (compressedInput.markSupported() == false) {
                compressedInput = new BufferedInputStream(compressedInput);
            }
            // Detect the content type from the decompressed stream.
            final XContentType contentType = XContentFactory.xContentType(compressedInput);
            // Create and return an XContentParser for the detected content type and decompressed stream.
            return XContentFactory.xContent(contentType).createParser(config, compressedInput);
        } else {
            // If no compressor is found, treat the bytes as uncompressed and detect content type.
            return createParserNotCompressed(config, bytes, xContentType(bytes));
        }
    }

    /**
     * Creates an {@link XContentParser} from uncompressed {@link BytesReference} and a known {@link XContentType}.
     * This method is a more efficient alternative to {@link #createParser(XContentParserConfiguration, BytesReference)}
     * when the content type is already known and the bytes are confirmed to be uncompressed.
     * @param config The {@link XContentParserConfiguration} to use for parser settings.
     * @param bytes The uncompressed {@link BytesReference} containing the XContent data.
     * @param xContentType The explicit {@link XContentType} of the data.
     * @return An {@link XContentParser} instance configured for the specified content type.
     * @throws IOException If an I/O error occurs during parser creation.
     */
    public static XContentParser createParserNotCompressed(
        XContentParserConfiguration config,
        BytesReference bytes,
        XContentType xContentType
    ) throws IOException {
        // Retrieves the XContent instance associated with the specified content type.
        XContent xContent = xContentType.xContent();
        // Optimizes parser creation if the bytes reference has an accessible byte array.
        if (bytes.hasArray()) {
            return xContent.createParser(config, bytes.array(), bytes.arrayOffset(), bytes.length());
        }
        // Otherwise, creates a parser from the byte reference's input stream.
        return xContent.createParser(config, bytes.streamInput());
    }

    /**
     * Creates an {@link XContentParser} for the given {@link BytesReference} and explicit {@link XContentType}.
     * This method allows specifying the content type directly, bypassing auto-detection.
     * @param registry A {@link NamedXContentRegistry} to use for named content parsing.
     * @param deprecation A {@link DeprecationHandler} for managing deprecated XContent features.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param xContentType The explicit {@link XContentType} of the data.
     * @return An {@link XContentParser} instance configured for the specified content type.
     * @throws IOException If an I/O error occurs during parser creation.
     * @deprecated Use {@link #createParser(XContentParserConfiguration, BytesReference, XContentType)}
     * for a more direct configuration without redundant registry and deprecation handler parameters.
     */
    @Deprecated
    public static XContentParser createParser(
        NamedXContentRegistry registry,
        DeprecationHandler deprecation,
        BytesReference bytes,
        XContentType xContentType
    ) throws IOException {
        // Constructs a parser configuration and delegates to the more specific createParser method.
        return createParser(
            XContentParserConfiguration.EMPTY.withRegistry(registry).withDeprecationHandler(deprecation),
            bytes,
            xContentType
        );
    }

    /**
     * Creates an {@link XContentParser} for the given {@link BytesReference} and explicit {@link XContentType}.
     * This method is the preferred way to create a parser when the content type is known,
     * as it avoids auto-detection overhead and potential ambiguity. It handles compressed data.
     * @param config The {@link XContentParserConfiguration} to use for parser settings.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param xContentType The explicit {@link XContentType} of the data. Must not be null.
     * @return An {@link XContentParser} instance configured for the specified content type.
     * @throws IOException If an I/O error occurs during parser creation or decompression.
     */
    public static XContentParser createParser(XContentParserConfiguration config, BytesReference bytes, XContentType xContentType)
        throws IOException {
        Objects.requireNonNull(xContentType); // Ensure the content type is provided.
        // Checks if the bytes are compressed and obtains a compressor if so.
        Compressor compressor = CompressorFactory.compressor(bytes);
        if (compressor != null) {
            // If compressed, decompress the input stream and create a parser.
            return XContentFactory.xContent(xContentType).createParser(config, compressor.threadLocalInputStream(bytes.streamInput()));
        } else {
            // If not compressed, delegate to the uncompressed parser creation method.
            // TODO: The comment suggests a potential future optimization to avoid redundant checks.
            return createParserNotCompressed(config, bytes, xContentType);
        }
    }

    /**
     * Converts the given {@link BytesReference} into a Java {@link Map} representation.
     * The map can be optionally ordered.
     * This method attempts to auto-detect the {@link XContentType} and decompresses the input if necessary.
     * <p>
     * Warning: This method may result in a loss of precision for floating-point numbers
     * due to the conversion to {@code double}, which has limited precision (52 bits for mantissa).
     * This is particularly relevant for nanosecond precision dates stored as decimal numbers.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param ordered A boolean indicating whether the resulting map should preserve insertion order.
     * @return A {@link Tuple} containing the detected {@link XContentType} and the resulting {@link Map}.
     * @throws ElasticsearchParseException If parsing fails or the content is malformed.
     * @deprecated This method relies on auto-detection of content type.
     *             Use {@link #convertToMap(BytesReference, boolean, XContentType)} instead
     *             with an explicit {@link XContentType} for better performance and reliability.
     */
    @Deprecated
    public static Tuple<XContentType, Map<String, Object>> convertToMap(BytesReference bytes, boolean ordered)
        throws ElasticsearchParseException {
        // Delegates to a generalized parsing function, using a map extractor that respects order.
        return parseToType(ordered ? XContentParser::mapOrdered : XContentParser::map, bytes, null, XContentParserConfiguration.EMPTY);
    }

    /**
     * Converts the given {@link BytesReference} into a Java {@link Map} representation,
     * using an explicit {@link XContentType} and an optional {@link XContentParserDecorator}.
     * This method does not perform any field filtering.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param ordered A boolean indicating whether the resulting map should preserve insertion order.
     * @param xContentType The explicit {@link XContentType} of the data.
     * @param parserDecorator An optional {@link XContentParserDecorator} to apply additional processing to the parser.
     * @return A {@link Tuple} containing the detected {@link XContentType} and the resulting {@link Map}.
     * @throws ElasticsearchParseException If parsing fails or the content is malformed.
     */
    public static Tuple<XContentType, Map<String, Object>> convertToMap(
        BytesReference bytes,
        boolean ordered,
        XContentType xContentType,
        XContentParserDecorator parserDecorator
    ) {
        // Delegates to a generalized parsing function, ensuring order and applying the decorator.
        return parseToType(
            ordered ? XContentParser::mapOrdered : XContentParser::map,
            bytes,
            xContentType,
            XContentParserConfiguration.EMPTY,
            parserDecorator
        );
    }

    /**
     * Converts the given {@link BytesReference} into a Java {@link Map} representation,
     * using an explicit {@link XContentType}. The map can be optionally ordered.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param ordered A boolean indicating whether the resulting map should preserve insertion order.
     * @param xContentType The explicit {@link XContentType} of the data.
     * @return A {@link Tuple} containing the detected {@link XContentType} and the resulting {@link Map}.
     */
    public static Tuple<XContentType, Map<String, Object>> convertToMap(BytesReference bytes, boolean ordered, XContentType xContentType) {
        // Delegates to a generalized parsing function, respecting order.
        return parseToType(
            ordered ? XContentParser::mapOrdered : XContentParser::map,
            bytes,
            xContentType,
            XContentParserConfiguration.EMPTY
        );
    }

    /**
     * Converts the given {@link BytesReference} into a Java {@link Map} representation,
     * using an explicit {@link XContentType}. The map can be optionally ordered.
     * This method also supports including or excluding specific fields during parsing.
     * <p>
     * Warning: This method may result in a loss of precision for floating-point numbers
     * due to the conversion to {@code double}, which has limited precision (52 bits for mantissa).
     * This is particularly relevant for nanosecond precision dates stored as decimal numbers.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param ordered A boolean indicating whether the resulting map should preserve insertion order.
     * @param xContentType The explicit {@link XContentType} of the data. Must not be null.
     * @param include An optional {@link Set} of field names to include in the parsed map. If null, no fields are explicitly included.
     * @param exclude An optional {@link Set} of field names to exclude from the parsed map. If null, no fields are explicitly excluded.
     * @return A {@link Tuple} containing the detected {@link XContentType} and the resulting {@link Map}.
     * @throws ElasticsearchParseException If parsing fails or the content is malformed.
     */
    public static Tuple<XContentType, Map<String, Object>> convertToMap(
        BytesReference bytes,
        boolean ordered,
        XContentType xContentType,
        @Nullable Set<String> include,
        @Nullable Set<String> exclude
    ) throws ElasticsearchParseException {
        XContentParserConfiguration config = XContentParserConfiguration.EMPTY;
        // If include or exclude filters are provided, configure the parser accordingly.
        if (include != null || exclude != null) {
            config = config.withFiltering(null, include, exclude, false);
        }
        // Delegates to a generalized parsing function with the configured filtering.
        return parseToType(ordered ? XContentParser::mapOrdered : XContentParser::map, bytes, xContentType, config);
    }
    /**
     * Creates an {@link XContentParser} from the given {@link BytesReference} and extracts a typed object using the provided {@code extractor}.
     * This method handles content type auto-detection if {@code xContentType} is null and applies an optional {@link XContentParserConfiguration}.
     * @param <T> The type of the object to be extracted.
     * @param extractor A {@link CheckedFunction} that defines how to extract the object from an {@link XContentParser}.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param xContentType An optional explicit {@link XContentType} of the data. If null, the content type will be auto-detected.
     * @param config An optional {@link XContentParserConfiguration} for parser settings.
     * @return A {@link Tuple} containing the detected {@link XContentType} and the extracted object.
     * @throws ElasticsearchParseException If parsing or extraction fails.
     * @deprecated If {@code xContentType} is null, this method relies on auto-detection of content type.
     *             It is recommended to provide a non-null {@link XContentType} for clarity and efficiency.
     */
    @Deprecated
    public static <T> Tuple<XContentType, T> parseToType(
        CheckedFunction<XContentParser, T, IOException> extractor,
        BytesReference bytes,
        @Nullable XContentType xContentType,
        @Nullable XContentParserConfiguration config
    ) throws ElasticsearchParseException {
        // Delegates to an overloaded method, passing a no-op parser decorator.
        return parseToType(extractor, bytes, xContentType, config, XContentParserDecorator.NOOP);
    }

    /**
     * Creates an {@link XContentParser} from the given {@link BytesReference} and extracts a typed object using the provided {@code extractor}.
     * This method handles content type auto-detection if {@code xContentType} is null and applies an optional {@link XContentParserConfiguration}
     * and a {@link XContentParserDecorator}.
     * @param <T> The type of the object to be extracted.
     * @param extractor A {@link CheckedFunction} that defines how to extract the object from an {@link XContentParser}.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param xContentType An optional explicit {@link XContentType} of the data. If null, the content type will be auto-detected.
     * @param config An optional {@link XContentParserConfiguration} for parser settings.
     * @param parserDecorator A {@link XContentParserDecorator} to apply additional processing to the parser.
     * @return A {@link Tuple} containing the detected {@link XContentType} and the extracted object.
     * @throws ElasticsearchParseException If parsing or extraction fails.
     */
    public static <T> Tuple<XContentType, T> parseToType(
        CheckedFunction<XContentParser, T, IOException> extractor,
        BytesReference bytes,
        @Nullable XContentType xContentType,
        @Nullable XContentParserConfiguration config,
        XContentParserDecorator parserDecorator
    ) throws ElasticsearchParseException {
        // Initializes configuration, defaulting to EMPTY if not provided.
        config = config != null ? config : XContentParserConfiguration.EMPTY;
        try (
            // Creates a parser, either by auto-detecting or using the provided content type, and applies the decorator.
            XContentParser parser = parserDecorator.decorate(
                xContentType != null ? createParser(config, bytes, xContentType) : createParser(config, bytes)
            )
        ) {
            // Extracts the object using the provided extractor and wraps it with the content type in a Tuple.
            Tuple<XContentType, T> xContentTypeTTuple = new Tuple<>(parser.contentType(), extractor.apply(parser));
            return xContentTypeTTuple;
        } catch (IOException e) {
            // Catches IOException and re-throws it as ElasticsearchParseException for consistency.
            throw new ElasticsearchParseException("Failed to parse content to type", e);
        }
    }

    /**
     * Converts a string in a specified {@link XContent} format to a Java {@link Map}.
     * @param xContent The {@link XContent} instance representing the format of the input string.
     * @param string The input string containing the XContent data.
     * @param ordered A boolean indicating whether the resulting map should preserve insertion order.
     * @return A Java {@link Map} representing the parsed XContent.
     * @throws ElasticsearchParseException If there is any error during parsing (e.g., malformed content).
     */
    public static Map<String, Object> convertToMap(XContent xContent, String string, boolean ordered) throws ElasticsearchParseException {
        try (XContentParser parser = xContent.createParser(XContentParserConfiguration.EMPTY, string)) {
            // Parses the string into a map, preserving order if requested.
            return ordered ? parser.mapOrdered() : parser.map();
        } catch (IOException e) {
            // Catches IOException and re-throws it as ElasticsearchParseException.
            throw new ElasticsearchParseException("Failed to parse content to map", e);
        }
    }

    /**
     * Converts an {@link InputStream} containing XContent data into a Java {@link Map}.
     * This method does not perform any field filtering.
     * @param xContent The {@link XContent} instance representing the format of the input stream.
     * @param input The {@link InputStream} containing the XContent data.
     * @param ordered A boolean indicating whether the resulting map should preserve insertion order.
     * @return A Java {@link Map} representing the parsed XContent.
     * @throws ElasticsearchParseException If there is any error during parsing.
     */
    public static Map<String, Object> convertToMap(XContent xContent, InputStream input, boolean ordered)
        throws ElasticsearchParseException {
        // Delegates to an overloaded method, with no include/exclude filters.
        return convertToMap(xContent, input, ordered, null, null);
    }

    /**
     * Converts an {@link InputStream} containing XContent data into a Java {@link Map}.
     * This method allows for optional inclusion or exclusion of fields during parsing.
     * Unlike {@link #convertToMap(BytesReference, boolean)}, this method does not automatically
     * uncompress the input stream, so the input is expected to be uncompressed.
     * @param xContent The {@link XContent} instance representing the format of the input stream.
     * @param input The {@link InputStream} containing the XContent data.
     * @param ordered A boolean indicating whether the resulting map should preserve insertion order.
     * @param include An optional {@link Set} of field names to include in the parsed map.
     * @param exclude An optional {@link Set} of field names to exclude from the parsed map.
     * @return A Java {@link Map} representing the parsed XContent.
     * @throws ElasticsearchParseException If there is any error during parsing.
     */
    public static Map<String, Object> convertToMap(
        XContent xContent,
        InputStream input,
        boolean ordered,
        @Nullable Set<String> include,
        @Nullable Set<String> exclude
    ) throws ElasticsearchParseException {
        try (
            // Creates a parser with optional field filtering configured.
            XContentParser parser = xContent.createParser(
                XContentParserConfiguration.EMPTY.withFiltering(null, include, exclude, false),
                input
            )
        ) {
            // Parses the input stream into a map, preserving order if requested.
            return ordered ? parser.mapOrdered() : parser.map();
        } catch (IOException e) {
            // Catches IOException and re-throws it as ElasticsearchParseException.
            throw new ElasticsearchParseException("Failed to parse content to map", e);
        }
    }
    /**
     * Converts a byte array containing XContent data into a Java {@link Map}.
     * Unlike {@link #convertToMap(BytesReference, boolean)}, this method does not automatically
     * uncompress the input, so the byte array is expected to contain uncompressed data.
     * @param xContent The {@link XContent} instance representing the format of the byte array.
     * @param bytes The byte array containing the XContent data.
     * @param offset The starting offset within the byte array.
     * @param length The length of the XContent data within the byte array.
     * @param ordered A boolean indicating whether the resulting map should preserve insertion order.
     * @return A Java {@link Map} representing the parsed XContent.
     * @throws ElasticsearchParseException If there is any error during parsing.
     */
    public static Map<String, Object> convertToMap(XContent xContent, byte[] bytes, int offset, int length, boolean ordered)
        throws ElasticsearchParseException {
        // Delegates to an overloaded method with no include/exclude filters.
        return convertToMap(xContent, bytes, offset, length, ordered, null, null);
    }

    /**
     * Converts a byte array containing XContent data into a Java {@link Map}.
     * This method allows for optional inclusion or exclusion of fields during parsing.
     * Unlike {@link #convertToMap(BytesReference, boolean)}, this method does not automatically
     * uncompress the input, so the byte array is expected to contain uncompressed data.
     * @param xContent The {@link XContent} instance representing the format of the byte array.
     * @param bytes The byte array containing the XContent data.
     * @param offset The starting offset within the byte array.
     * @param length The length of the XContent data within the byte array.
     * @param ordered A boolean indicating whether the resulting map should preserve insertion order.
     * @param include An optional {@link Set} of field names to include in the parsed map.
     * @param exclude An optional {@link Set} of field names to exclude from the parsed map.
     * @return A Java {@link Map} representing the parsed XContent.
     * @throws ElasticsearchParseException If there is any error during parsing.
     */
    public static Map<String, Object> convertToMap(
        XContent xContent,
        byte[] bytes,
        int offset,
        int length,
        boolean ordered,
        @Nullable Set<String> include,
        @Nullable Set<String> exclude
    ) throws ElasticsearchParseException {
        try (
            // Creates a parser with optional field filtering configured.
            XContentParser parser = xContent.createParser(
                XContentParserConfiguration.EMPTY.withFiltering(null, include, exclude, false),
                bytes,
                offset,
                length
            )
        ) {
            // Parses the byte array into a map, preserving order if requested.
            return ordered ? parser.mapOrdered() : parser.map();
        } catch (IOException e) {
            // Catches IOException and re-throws it as ElasticsearchParseException.
            throw new ElasticsearchParseException("Failed to parse content to map", e);
        }
    }
    /**
     * Converts {@link BytesReference} containing XContent data into a JSON string.
     * This method implicitly uses auto-detection for the content type.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param reformatJson A boolean indicating whether the JSON should be reformatted (pretty-printed if applicable).
     * @return A JSON string representation of the XContent.
     * @throws IOException If an I/O error occurs during conversion.
     * @deprecated Use {@link #convertToJson(BytesReference, boolean, XContentType)}
     * to explicitly specify the content type and avoid auto-detection.
     */
    @Deprecated
    public static String convertToJson(BytesReference bytes, boolean reformatJson) throws IOException {
        // Delegates to an overloaded method without pretty-printing.
        return convertToJson(bytes, reformatJson, false);
    }

    /**
     * Converts {@link BytesReference} containing XContent data into a JSON string, with optional pretty-printing.
     * This method implicitly uses auto-detection for the content type.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param reformatJson A boolean indicating whether the JSON should be reformatted.
     * @param prettyPrint A boolean indicating whether the JSON output should be pretty-printed for readability.
     * @return A JSON string representation of the XContent.
     * @throws IOException If an I/O error occurs during conversion.
     * @deprecated Use {@link #convertToJson(BytesReference, boolean, boolean, XContentType)}
     * to explicitly specify the content type and avoid auto-detection.
     */
    @Deprecated
    public static String convertToJson(BytesReference bytes, boolean reformatJson, boolean prettyPrint) throws IOException {
        // Delegates to an overloaded method, first detecting the content type.
        return convertToJson(bytes, reformatJson, prettyPrint, xContentType(bytes));
    }

    /**
     * Converts {@link BytesReference} containing XContent data into a JSON string.
     * This method requires an explicit {@link XContentType}.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param reformatJson A boolean indicating whether the JSON should be reformatted.
     * @param xContentType The explicit {@link XContentType} of the data.
     * @return A JSON string representation of the XContent.
     * @throws IOException If an I/O error occurs during conversion.
     */
    public static String convertToJson(BytesReference bytes, boolean reformatJson, XContentType xContentType) throws IOException {
        // Delegates to an overloaded method without pretty-printing.
        return convertToJson(bytes, reformatJson, false, xContentType);
    }

    /**
     * Takes a JSON string, parses it, and then re-serializes it without any
     * unnecessary whitespace (e.g., for comparison in tests or storage optimization).
     *
     * @param json The input JSON string to strip whitespace from.
     * @return A reformatted JSON string with minimal whitespace.
     * @throws IOException If parsing or reformatting the JSON fails (e.g., due to malformed JSON).
     */
    public static String stripWhitespace(String json) throws IOException {
        // Converts the input string to BytesArray and then to JSON, forcing reformatting and JSON content type.
        return convertToJson(new BytesArray(json), true, XContentType.JSON);
    }

    /**
     * Converts {@link BytesReference} containing XContent data into a JSON string, with optional reformatting and pretty-printing.
     * This method requires an explicit {@link XContentType}. If the source is already JSON and no reformatting is requested,
     * it returns the UTF-8 string directly for efficiency.
     * @param bytes The {@link BytesReference} containing the XContent data.
     * @param reformatJson A boolean indicating whether the JSON should be reformatted (e.g., pretty-printed, stripped of whitespace).
     * @param prettyPrint A boolean indicating whether the JSON output should be pretty-printed for readability.
     * @param xContentType The explicit {@link XContentType} of the data. Must not be null.
     * @return A JSON string representation of the XContent.
     * @throws IOException If an I/O error occurs during conversion.
     */
    public static String convertToJson(BytesReference bytes, boolean reformatJson, boolean prettyPrint, XContentType xContentType)
        throws IOException {
        Objects.requireNonNull(xContentType);
        // Optimization: If the content is already canonical JSON and no reformatting is needed, return directly.
        if (xContentType.canonical() == XContentType.JSON && reformatJson == false) {
            return bytes.utf8ToString();
        }

        try (var parser = createParserNotCompressed(XContentParserConfiguration.EMPTY, bytes, xContentType)) {
            // Creates a parser from the bytes and then uses it to build a JSON string.
            return toJsonString(prettyPrint, parser);
        }
    }

    /**
     * Converts the current structure of an {@link XContentParser} into a JSON string.
     * This internal utility method handles the actual building of the JSON string from the parser's state.
     * @param prettyPrint A boolean indicating whether the JSON output should be pretty-printed.
     * @param parser The {@link XContentParser} whose current structure needs to be converted.
     * @return A JSON string representation of the parser's current structure.
     * @throws IOException If an I/O error occurs during JSON string generation.
     */
    private static String toJsonString(boolean prettyPrint, XContentParser parser) throws IOException {
        // Advance the parser to the next token, typically the start of the content.
        parser.nextToken();
        // Create a JSON XContentBuilder.
        XContentBuilder builder = XContentFactory.jsonBuilder();
        if (prettyPrint) {
            builder.prettyPrint(); // Enable pretty-printing if requested.
        }
        // Copy the current structure from the parser to the builder.
        builder.copyCurrentStructure(parser);
        // Convert the builder's content to a string.
        return Strings.toString(builder);
    }

    /**
     * Updates a source map with changes from another map.
     * This method performs a deep merge: if both source and changes contain a key that maps to another map,
     * the inner maps are recursively merged. For non-map values, the change value overrides the source value.
     *
     * @param source The original map to be updated.
     * @param changes The map containing the changes to apply.
     * @param checkUpdatesAreUnequal A boolean indicating whether to perform an {@code Objects.equals} check
     *                               for non-map value updates. Setting this to {@code true} can be
     *                               computationally expensive for large objects/strings but helps
     *                               track if a modification truly occurred.
     * @return {@code true} if the source map was modified as a result of the update, {@code false} otherwise.
     */
    public static boolean update(Map<String, Object> source, Map<String, Object> changes, boolean checkUpdatesAreUnequal) {
        boolean modified = false;
        // Iterate over each entry in the changes map.
        for (Map.Entry<String, Object> changesEntry : changes.entrySet()) {
            // Pre-condition: Check if the key from 'changes' exists in 'source'.
            if (source.containsKey(changesEntry.getKey()) == false) {
                // If the key does not exist in the source, it's a new entry; safely copy it over.
                source.put(changesEntry.getKey(), changesEntry.getValue());
                modified = true; // Mark as modified.
                continue; // Move to the next change entry.
            }

            // Invariant: The key exists in both 'source' and 'changes'.
            Object old = source.get(changesEntry.getKey());
            // Block Logic: Determines if a recursive merge is needed for nested maps.
            if (old instanceof Map && changesEntry.getValue() instanceof Map) {
                // If both old and new values are maps, recursively merge them.
                // The 'modified' status is updated based on the recursive call.
                modified |= update(
                    (Map<String, Object>) source.get(changesEntry.getKey()),
                    (Map<String, Object>) changesEntry.getValue(),
                    checkUpdatesAreUnequal && modified == false // Propagate checkUpdatesAreUnequal.
                );
                continue; // Move to the next change entry after recursive merge.
            }
            // Block Logic: Handles updates for non-map values.
            // Overwrites the existing value in 'source' with the new value from 'changes'.
            source.put(changesEntry.getKey(), changesEntry.getValue());
            if (modified) {
                // If already marked as modified, no need for further equality checks for this entry.
                continue;
            }
            if (checkUpdatesAreUnequal == false) {
                // If equality check is disabled, assume modification.
                modified = true;
                continue;
            }
            // Inline: Perform a deep equality check to determine if the value actually changed.
            modified = Objects.equals(old, changesEntry.getValue()) == false;
        }
        return modified; // Returns true if any modification occurred.
    }

    /**
     * Merges default values from the {@code defaults} map into the {@code content} map.
     * This is a specialized merge operation where values in {@code content} take precedence
     * over values in {@code defaults}. Only applies recursive merging for nested map structures.
     * @param content The primary map to be augmented with default values. Existing values in this map
     *                will not be overridden by defaults.
     * @param defaults The map containing default values.
     */
    public static void mergeDefaults(Map<String, Object> content, Map<String, Object> defaults) {
        // Delegates to the general merge method with 'first' taking precedence and no custom merge logic.
        merge(content, defaults, null);
    }
    /**
     * Merges the contents of the {@code second} map into the {@code first} map.
     * This is a general merge operation where values in {@code first} take precedence
     * over values in {@code second} unless a {@link CustomMerge} is provided for conflict resolution.
     * Recursive merging is applied to nested map structures.
     * @param first The base map, whose values take precedence in case of conflicts, unless handled by {@code customMerge}.
     * @param second The map whose contents are merged into the base map.
     * @param customMerge An optional {@link CustomMerge} instance for resolving conflicts when both maps
     *                    have concrete values (non-map/non-collection) for the same key. If null, {@code first}
     *                    map's values always take precedence.
     */
    public static void merge(Map<String, Object> first, Map<String, Object> second, @Nullable CustomMerge customMerge) {
        // Delegates to an overloaded method, indicating no parent key for the initial call.
        merge(null, first, second, customMerge);
    }
    /**
     * Merges the contents of the {@code second} map into the {@code first} map, providing context
     * about the parent key for recursive calls.
     * This is a general merge operation where values in {@code first} take precedence
     * over values in {@code second} unless a {@link CustomMerge} is provided for conflict resolution.
     * Recursive merging is applied to nested map structures.
     * @param parent The key of the common parent map, used for contextual information in recursive merges.
     * @param first The base map, whose values take precedence in case of conflicts, unless handled by {@code customMerge}.
     * @param second The map whose contents are merged into the base map.
     * @param customMerge An optional {@link CustomMerge} instance for resolving conflicts when both maps
     *                    have concrete values (non-map/non-collection) for the same key. If null, {@code first}
     *                    map's values always take precedence.
     * @throws IllegalStateException If a custom merge operation for maps does not yield a map.
     */
    public static void merge(
        @Nullable String parent,
        Map<String, Object> first,
        Map<String, Object> second,
        @Nullable CustomMerge customMerge
    ) {
        // Iterate over entries in the second map, which are potential additions or updates to the first map.
        for (Map.Entry<String, Object> toMergeEntry : second.entrySet()) {
            // Pre-condition: Check if the key from 'toMergeEntry' exists in 'first'.
            if (first.containsKey(toMergeEntry.getKey()) == false) {
                // Block Logic: Add new entries.
                // If the key does not exist in the 'first' map, directly copy the entry from 'second'.
                first.put(toMergeEntry.getKey(), toMergeEntry.getValue());
            } else {
                // Invariant: The key exists in both 'first' and 'second' maps, indicating a potential conflict or a deep merge.
                // Block Logic: Handle existing entries.
                Object baseValue = first.get(toMergeEntry.getKey());
                // Case 1: Both values are maps, requiring a recursive merge.
                if (baseValue instanceof Map && toMergeEntry.getValue() instanceof Map) {
                    Map<String, Object> mergedValue = null;
                    if (customMerge != null) {
                        // Attempt to use custom merge logic for the map.
                        Object tmp = customMerge.merge(parent, toMergeEntry.getKey(), baseValue, toMergeEntry.getValue());
                        // Validate that if a custom merge returns a value, it must be a map.
                        if (tmp != null && tmp instanceof Map == false) {
                            throw new IllegalStateException("merging of values for [" + toMergeEntry.getKey() + "] must yield a map");
                        }
                        mergedValue = (Map<String, Object>) tmp;
                    }
                    if (mergedValue != null) {
                        // If custom merge provided a result, use it.
                        first.put(toMergeEntry.getKey(), mergedValue);
                    } else {
                        // If no custom merge result, perform a standard recursive merge of maps.
                        merge(
                            toMergeEntry.getKey(), // Pass the current key as parent for the next level of recursion.
                            (Map<String, Object>) baseValue,
                            (Map<String, Object>) toMergeEntry.getValue(),
                            customMerge
                        );
                    }
                }
                // Case 2: Both values are lists, requiring a specific list merging strategy.
                else if (baseValue instanceof List && toMergeEntry.getValue() instanceof List) {
                    List<Object> listToMerge = (List<Object>) toMergeEntry.getValue();
                    List<Object> baseList = (List<Object>) baseValue;

                    // Block Logic: Special handling for lists of single-entry maps (e.g., [ {"key1": {}}, {"key2": {}} ]).
                    if (allListValuesAreMapsOfOne(listToMerge) && allListValuesAreMapsOfOne(baseList)) {
                        // Algorithm: Merge lists of single-entry maps based on their keys.
                        // Create a temporary map to hold processed entries from the base list, keyed by their single key.
                        Map<String, Map<String, Object>> processed = new LinkedHashMap<>();
                        for (Object o : baseList) {
                            Map<String, Object> map = (Map<String, Object>) o;
                            Map.Entry<String, Object> entry = map.entrySet().iterator().next();
                            processed.put(entry.getKey(), map);
                        }
                        // Iterate through the list to merge.
                        for (Object o : listToMerge) {
                            Map<String, Object> map = (Map<String, Object>) o;
                            Map.Entry<String, Object> entry = map.entrySet().iterator().next();
                            if (processed.containsKey(entry.getKey())) {
                                // If a key exists in both lists, recursively merge the corresponding maps.
                                merge(toMergeEntry.getKey(), processed.get(entry.getKey()), map, customMerge);
                            } else {
                                // If the key is new, add the map to the processed entries.
                                processed.put(entry.getKey(), map);
                            }
                        }
                        // Update the 'first' map with the new, merged list of maps.
                        first.put(toMergeEntry.getKey(), new ArrayList<>(processed.values()));
                    } else {
                        // Block Logic: General list merging for other list types.
                        // If not lists of single-entry maps, simply combine them, avoiding duplicates.
                        // Custom merge is not applicable here as it's for non-collection values.
                        List<Object> mergedList = new ArrayList<>(listToMerge); // Start with the second list's elements.

                        // Add elements from the base list if they are not already in the merged list.
                        for (Object o : baseList) {
                            if (mergedList.contains(o) == false) {
                                mergedList.add(o);
                            }
                        }
                        first.put(toMergeEntry.getKey(), mergedList); // Update the 'first' map with the combined list.
                    }
                }
                // Case 3: Values are concrete (not maps or lists), and a custom merge function is provided.
                else if (customMerge != null) {
                    // Attempt a custom merge for non-collection values.
                    Object mergedValue = customMerge.merge(parent, toMergeEntry.getKey(), baseValue, toMergeEntry.getValue());
                    if (mergedValue != null) {
                        // If custom merge provided a result, use it to update the 'first' map.
                        first.put(toMergeEntry.getKey(), mergedValue);
                    }
                    // Invariant: If customMerge returns null, the original value from 'first' is retained,
                    // as per the merge semantics (values from 'first' have precedence).
                }
            }
        }
    }

    /**
     * Determines if all elements within a given list are maps that contain exactly one entry.
     * This utility method is used to identify a specific structure in lists that requires a special merging strategy.
     * @param list The {@link List} of objects to check.
     * @return {@code true} if all elements are single-entry maps, {@code false} otherwise.
     */
    private static boolean allListValuesAreMapsOfOne(List<Object> list) {
        // Invariant: Iterate through each object in the list.
        for (Object o : list) {
            // Pre-condition 1: Check if the object is an instance of Map.
            if ((o instanceof Map) == false) {
                return false; // If not a map, the condition is not met.
            }
            // Pre-condition 2: Check if the map contains exactly one entry.
            // Inline: Cast to Map to access its size.
            if (((Map) o).size() != 1) {
                return false; // If the map size is not 1, the condition is not met.
            }
        }
        return true; // If all elements satisfy both conditions, return true.
    }
    /**
     * A functional interface enabling custom merge logic for map entries within the {@link XContentHelper#merge} methods.
     * Implementations of this interface can define how to resolve conflicts when merging values for the same key.
     */
    @FunctionalInterface
    public interface CustomMerge {
        /**
         * Computes a merged value for a given key based on its old and new values.
         *
         * @param parent The key of the parent map, providing context for the current merge operation.
         * @param key The key of the entry being merged.
         * @param oldValue The existing value associated with the key in the base map.
         * @param newValue The new value associated with the key from the map being merged in.
         * @return The custom merged value. If {@code null} is returned, the default merge behavior applies:
         *         if values are maps, they are merged recursively; otherwise, the {@code oldValue} is retained.
         * @throws RuntimeException Implementations are expected to throw a {@link RuntimeException} for illegal merge scenarios.
         */
        @Nullable
        Object merge(String parent, String key, Object oldValue, Object newValue);
    }
    /**
     * Writes a "raw" field (meaning its content is passed as bytes) to an {@link XContentBuilder}.
     * This method automatically handles decompression if the source bytes are compressed,
     * and optimizes writing by using {@link XContentBuilder#rawField(String, InputStream)}.
     * @param field The name of the field to write.
     * @param source The {@link BytesReference} containing the raw content.
     * @param builder The {@link XContentBuilder} to write the field to.
     * @param params Additional parameters for {@link ToXContent} (currently unused in this method's logic).
     * @throws IOException If an I/O error occurs during decompression or writing.
     * @deprecated Use {@link #writeRawField(String, BytesReference, XContentType, XContentBuilder, Params)}
     * to explicitly specify the content type and avoid auto-detection for better performance and reliability.
     */
    @Deprecated
    public static void writeRawField(String field, BytesReference source, XContentBuilder builder, ToXContent.Params params)
        throws IOException {
        // Attempts to find a compressor for the given source bytes.
        Compressor compressor = CompressorFactory.compressorForUnknownXContentType(source);
        if (compressor != null) {
            // If compressed, decompress the input stream and write the raw field.
            try (InputStream compressedStreamInput = compressor.threadLocalInputStream(source.streamInput())) {
                builder.rawField(field, compressedStreamInput);
            }
        } else {
            // If not compressed, directly write the raw field from the source's input stream.
            try (InputStream stream = source.streamInput()) {
                builder.rawField(field, stream);
            }
        }
    }
    /**
     * Writes a "raw" field (meaning its content is passed as bytes) to an {@link XContentBuilder}.
     * This method automatically handles decompression if the source bytes are compressed,
     * and optimizes writing by using {@link XContentBuilder#rawField(String, InputStream, XContentType)}.
     * This method requires an explicit {@link XContentType} for the raw field.
     * @param field The name of the field to write.
     * @param source The {@link BytesReference} containing the raw content.
     * @param xContentType The explicit {@link XContentType} of the raw content. Must not be null.
     * @param builder The {@link XContentBuilder} to write the field to.
     * @param params Additional parameters for {@link ToXContent} (currently unused in this method's logic).
     * @throws IOException If an I/O error occurs during decompression or writing.
     */
    public static void writeRawField(
        String field,
        BytesReference source,
        XContentType xContentType,
        XContentBuilder builder,
        ToXContent.Params params
    ) throws IOException {
        Objects.requireNonNull(xContentType); // Ensure content type is provided.
        // Attempts to find a compressor for the given source bytes.
        Compressor compressor = CompressorFactory.compressorForUnknownXContentType(source);
        if (compressor != null) {
            // If compressed, decompress the input stream and write the raw field with the specified content type.
            try (InputStream compressedStreamInput = compressor.threadLocalInputStream(source.streamInput())) {
                builder.rawField(field, compressedStreamInput, xContentType);
            }
        } else {
            // If not compressed, directly write the raw field from the source's input stream with the specified content type.
            try (InputStream stream = source.streamInput()) {
                builder.rawField(field, stream, xContentType);
            }
        }
    }
    /**
     * Converts a {@link ToXContent} object into its {@link XContent} byte representation.
     * The output format is specified by {@link XContentType}.
     * The method handles wrapping the content in an anonymous object if {@link ToXContent#isFragment()} returns true.
     * @param toXContent The {@link ToXContent} object to convert.
     * @param xContentType The desired {@link XContentType} for the output.
     * @param humanReadable A boolean indicating whether the output should be human-readable (pretty-printed).
     * @return A {@link BytesReference} containing the XContent byte representation.
     * @throws IOException If an I/O error occurs during XContent generation.
     */
    public static BytesReference toXContent(ToXContent toXContent, XContentType xContentType, boolean humanReadable) throws IOException {
        // Delegates to an overloaded method with empty parameters.
        return toXContent(toXContent, xContentType, ToXContent.EMPTY_PARAMS, humanReadable);
    }
    /**
     * Converts a {@link ChunkedToXContent} object into its {@link XContent} byte representation.
     * This method internally wraps the chunked content as a {@link ToXContent} object.
     * @param toXContent The {@link ChunkedToXContent} object to convert.
     * @param xContentType The desired {@link XContentType} for the output.
     * @param humanReadable A boolean indicating whether the output should be human-readable.
     * @return A {@link BytesReference} containing the XContent byte representation.
     * @throws IOException If an I/O error occurs during XContent generation.
     */
    public static BytesReference toXContent(ChunkedToXContent toXContent, XContentType xContentType, boolean humanReadable)
        throws IOException {
        // Wraps the ChunkedToXContent into a ToXContent and delegates to the corresponding method.
        return toXContent(ChunkedToXContent.wrapAsToXContent(toXContent), xContentType, humanReadable);
    }
    /**
     * Converts a {@link ToXContent} object into its {@link XContent} byte representation,
     * allowing for custom {@link ToXContent.Params}.
     * @param toXContent The {@link ToXContent} object to convert.
     * @param xContentType The desired {@link XContentType} for the output.
     * @param params Custom parameters to apply during XContent generation.
     * @param humanReadable A boolean indicating whether the output should be human-readable.
     * @return A {@link BytesReference} containing the XContent byte representation.
     * @throws IOException If an I/O error occurs during XContent generation.
     */
    public static BytesReference toXContent(ToXContent toXContent, XContentType xContentType, Params params, boolean humanReadable)
        throws IOException {
        // Delegates to an overloaded method, using the current REST API version.
        return toXContent(toXContent, xContentType, RestApiVersion.current(), params, humanReadable);
    }
    /**
     * Converts a {@link ToXContent} object into its {@link XContent} byte representation,
     * considering the target {@link RestApiVersion}.
     * @param toXContent The {@link ToXContent} object to convert.
     * @param xContentType The desired {@link XContentType} for the output.
     * @param restApiVersion The {@link RestApiVersion} against which the XContent should be generated.
     * @param params Custom parameters to apply during XContent generation.
     * @param humanReadable A boolean indicating whether the output should be human-readable.
     * @return A {@link BytesReference} containing the XContent byte representation.
     * @throws IOException If an I/O error occurs during XContent generation.
     */
    public static BytesReference toXContent(
        ToXContent toXContent,
        XContentType xContentType,
        RestApiVersion restApiVersion,
        Params params,
        boolean humanReadable
    ) throws IOException {
        // Block Logic: Construct the XContent using an XContentBuilder.
        try (XContentBuilder builder = XContentBuilder.builder(xContentType.xContent(), restApiVersion)) {
            builder.humanReadable(humanReadable); // Configure human-readable output.
            // If the ToXContent object represents a fragment, wrap it in a new JSON object.
            if (toXContent.isFragment()) {
                builder.startObject();
            }
            toXContent.toXContent(builder, params); // Write the actual content to the builder.
            if (toXContent.isFragment()) {
                builder.endObject();
            }
            return BytesReference.bytes(builder); // Return the generated XContent as BytesReference.
        }
    }
    /**
     * Converts a {@link ChunkedToXContent} object into its {@link XContent} byte representation,
     * allowing for custom {@link ToXContent.Params}.
     * This method internally wraps the chunked content as a {@link ToXContent} object.
     * @param toXContent The {@link ChunkedToXContent} object to convert.
     * @param xContentType The desired {@link XContentType} for the output.
     * @param params Custom parameters to apply during XContent generation.
     * @param humanReadable A boolean indicating whether the output should be human-readable.
     * @return A {@link BytesReference} containing the XContent byte representation.
     * @throws IOException If an I/O error occurs during XContent generation.
     */
    public static BytesReference toXContent(ChunkedToXContent toXContent, XContentType xContentType, Params params, boolean humanReadable)
        throws IOException {
        // Wraps the ChunkedToXContent into a ToXContent and delegates to the corresponding method.
        return toXContent(ChunkedToXContent.wrapAsToXContent(toXContent), xContentType, params, humanReadable);
    }
    /**
     * Attempts to guess the {@link XContentType} of provided bytes, which may be compressed.
     * This method involves decompressing the bytes if necessary to perform content type detection.
     * @param bytes The {@link BytesReference} containing the XContent data, potentially compressed.
     * @return The detected {@link XContentType}.
     * @throws UncheckedIOException If an {@link IOException} occurs during stream operations (e.g., decompression).
     * @deprecated Content type should ideally be known and explicitly provided, not guessed, except in specific
     *             scenarios where it's truly unknown (e.g., legacy APIs or specific client integrations).
     *             The REST layer should primarily rely on the "Content-Type" HTTP header.
     */
    @Deprecated
    public static XContentType xContentTypeMayCompressed(BytesReference bytes) {
        // Attempts to find a compressor for the given bytes, indicating if the data is compressed.
        Compressor compressor = CompressorFactory.compressorForUnknownXContentType(bytes);
        if (compressor != null) {
            // If compressed, decompress the input stream to allow content type detection.
            try {
                InputStream compressedStreamInput = compressor.threadLocalInputStream(bytes.streamInput());
                // Ensure the input stream supports marking for content type detection.
                if (compressedStreamInput.markSupported() == false) {
                    compressedStreamInput = new BufferedInputStream(compressedStreamInput);
                }
                return XContentFactory.xContentType(compressedStreamInput); // Detect content type from decompressed stream.
            } catch (IOException e) {
                // This scenario should ideally not occur as bytes are from memory.
                assert false : "Should not happen, we're just reading bytes from memory";
                throw new UncheckedIOException(e); // Wrap IOException in UncheckedIOException.
            }
        } else {
            // If not compressed, delegate to the method for uncompressed bytes.
            return XContentHelper.xContentType(bytes);
        }
    }

    /**
     * Guesses the {@link XContentType} of provided uncompressed bytes.
     * @param bytes The {@link BytesReference} containing the XContent data, expected to be uncompressed.
     * @return The detected {@link XContentType}.
     * @throws UncheckedIOException If an {@link IOException} occurs during stream operations.
     * @deprecated Content type should ideally be known and explicitly provided, not guessed.
     *             This method should be used sparingly in cases where explicit content type is genuinely unavailable.
     */
    @Deprecated
    public static XContentType xContentType(BytesReference bytes) {
        // Optimization: If the bytes reference has an accessible byte array, detect content type directly.
        if (bytes.hasArray()) {
            return XContentFactory.xContentType(bytes.array(), bytes.arrayOffset(), bytes.length());
        }
        // Otherwise, create an input stream from the bytes and detect content type.
        try {
            final InputStream inputStream = bytes.streamInput();
            assert inputStream.markSupported(); // Assumes input stream supports marking.
            return XContentFactory.xContentType(inputStream);
        } catch (IOException e) {
            // This scenario should ideally not occur as bytes are from memory.
            assert false : "Should not happen, we're just reading bytes from memory";
            throw new UncheckedIOException(e); // Wrap IOException in UncheckedIOException.
        }
    }
    /**
     * Extracts and returns the contents of the current object within an {@link XContentParser}
     * as an unparsed {@link BytesReference}. This is particularly useful for scenarios
     * like mapping definitions where the raw byte content is needed without full parsing
     * into Java objects, thereby avoiding unnecessary memory allocations and processing.
     * @param parser The {@link XContentParser} positioned at or before an object.
     * @return A {@link BytesReference} representing the raw bytes of the child object.
     * @throws IOException If an I/O error occurs, or if the parser's current token is not an object.
     * @throws XContentParseException If the expected start object token is not found.
     */
    public static BytesReference childBytes(XContentParser parser) throws IOException {
        // Pre-condition: Ensure the parser is at the start of an object.
        if (parser.currentToken() != XContentParser.Token.START_OBJECT) {
            // If not, try to advance the parser to the next token, expecting it to be START_OBJECT.
            if (parser.nextToken() != XContentParser.Token.START_OBJECT) {
                // If still not START_OBJECT, throw a parsing exception.
                throw new XContentParseException(
                    parser.getTokenLocation(),
                    "Expected [START_OBJECT] but got [" + parser.currentToken() + "]"
                );
            }
        }
        // Block Logic: Create a new XContentBuilder and copy the current object structure from the parser.
        XContentBuilder builder = XContentBuilder.builder(parser.contentType().xContent());
        builder.copyCurrentStructure(parser); // Copies the entire JSON object (or XContent equivalent) to the builder.
        return BytesReference.bytes(builder); // Returns the builder's content as raw bytes.
    }
    /**
     * Serializes {@link XContentType} values in a backward-compatible manner, particularly for
     * new VND_ (vendor-specific) values, when communicating with older nodes.
     * This method ensures that older nodes (pre-8.0.0) receive a canonical (recognized) content type ordinal,
     * while newer nodes receive the exact ordinal including vendor-specific types.
     * @param out The {@link StreamOutput} to write the content type to.
     * @param xContentType An instance of {@link XContentType} to serialize.
     * @throws IOException If an I/O error occurs during writing.
     * @deprecated This method is marked for removal in Elasticsearch v9, indicating that the BWC
     *             handling for XContentType ordinals will no longer be necessary.
     */
    public static void writeTo(StreamOutput out, XContentType xContentType) throws IOException {
        // Pre-condition: Check the transport version of the destination node.
        if (out.getTransportVersion().before(TransportVersions.V_8_0_0)) {
            // If the destination node is older than 8.0.0, send the canonical ordinal to ensure compatibility.
            // Older nodes might not recognize new VND_ XContentType instances directly.
            out.writeVInt(xContentType.canonical().ordinal());
        } else {
            // If the destination node is 8.0.0 or newer, send the exact ordinal, including VND_ types.
            out.writeVInt(xContentType.ordinal());
        }
    }
    /**
     * Convenience method that creates a {@link XContentParser} from a content map so that it can be passed to
     * existing REST based code for input parsing.
     *
     * @param config The {@link XContentParserConfiguration} for this mapper.
     * @param source The operator content as a map.
     * @return An {@link XContentParser} instance that can parse the map content.
     * @throws ElasticsearchGenerationException If an error occurs during the internal generation of XContent from the map.
     */
    public static XContentParser mapToXContentParser(XContentParserConfiguration config, Map<String, ?> source) {
        // Block Logic: Convert the map to JSON XContent and then create a parser from it.
        try (XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON)) {
            builder.map(source); // Write the map content to the builder.
            // Create and return a parser from the generated JSON bytes.
            return createParserNotCompressed(config, BytesReference.bytes(builder), builder.contentType());
        } catch (IOException e) {
            // Catches IOException during XContent generation and re-throws it as ElasticsearchGenerationException.
            throw new ElasticsearchGenerationException("Failed to generate [" + source + "]", e);
        }
    }}
