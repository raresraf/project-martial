/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.xcontent;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * @70adc879-bf7b-427b-89ae-eb337de7fff6/libs/x-content/src/main/java/org/elasticsearch/xcontent/Text.java
 * @brief Lazy-encoding container for text data with dual UTF-16 and UTF-8 representations.
 * 
 * Functional Intent: Minimizes encoding overhead by caching both String and ByteBuffer 
 * views. It materializes representations only when requested, ensuring that high-throughput 
 * I/O operations can utilize pre-encoded UTF-8 bytes without redundant processing.
 */
public final class Text implements XContentString, Comparable<Text>, ToXContentFragment {

    public static final Text[] EMPTY_ARRAY = new Text[0];

    /**
     * @brief Bulk conversion utility.
     * Logic: Maps an array of standard Strings to an array of Text objects to 
     * enable efficient downstream serialization.
     */
    public static Text[] convertFromStringArray(String[] strings) {
        if (strings.length == 0) {
            return EMPTY_ARRAY;
        }
        Text[] texts = new Text[strings.length];
        for (int i = 0; i < strings.length; i++) {
            texts[i] = new Text(strings[i]);
        }
        return texts;
    }

    private ByteBuffer bytes;
    private String text;
    private int hash;
    private int stringLength = -1;

    /**
     * @brief Constructs Text from raw UTF-8 bytes.
     * Logic: Defers length calculation until stringLength() or string() is invoked.
     */
    public Text(ByteBuffer bytes) {
        this.bytes = bytes;
    }

    /**
     * @brief Constructs Text from raw UTF-8 bytes with a pre-calculated character length.
     */
    public Text(ByteBuffer bytes, int stringLength) {
        this.bytes = bytes;
        this.stringLength = stringLength;
    }

    /**
     * @brief Constructs Text from a standard Java String.
     */
    public Text(String text) {
        this.text = text;
    }

    /**
     * @brief Predicate indicating if the UTF-8 view is currently cached.
     */
    public boolean hasBytes() {
        return bytes != null;
    }

    /**
     * Block Logic: Lazy UTF-8 encoding.
     * Logic: If bytes are missing, materializes them from the String representation 
     * using the standard UTF-8 encoder.
     */
    @Override
    public ByteBuffer bytes() {
        if (bytes == null) {
            bytes = StandardCharsets.UTF_8.encode(text);
        }
        return bytes;
    }

    /**
     * @brief Predicate indicating if the UTF-16 String view is currently cached.
     */
    public boolean hasString() {
        return text != null;
    }

    /**
     * Block Logic: Lazy UTF-16 decoding.
     * Logic: If the string is missing, materializes it from the byte representation 
     * using the standard UTF-8 decoder.
     */
    @Override
    public String string() {
        if (text == null) {
            text = StandardCharsets.UTF_8.decode(bytes).toString();
        }
        return text;
    }

    /**
     * Block Logic: Character length retrieval.
     * Logic: Returns the cached length if available, otherwise triggers 
     * string materialization to determine length.
     */
    @Override
    public int stringLength() {
        if (stringLength < 0) {
            stringLength = string().length();
        }
        return stringLength;
    }

    @Override
    public String toString() {
        return string();
    }

    @Override
    public int hashCode() {
        if (hash == 0) {
            hash = bytes().hashCode();
        }
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        return bytes().equals(((Text) obj).bytes());
    }

    @Override
    public int compareTo(Text text) {
        return bytes().compareTo(text.bytes());
    }

    /**
     * Block Logic: Optimized serialization dispatch.
     * Logic: 
     * 1. If a String already exists, uses the builder's standard string handler.
     * 2. Otherwise, performs a direct zero-copy write of the underlying 
     *    UTF-8 byte array to the output stream.
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        if (hasString()) {
            return builder.value(this.string());
        } else {
            // Functional Utility: Direct byte-level write avoiding intermediate string allocations.
            assert bytes.hasArray();
            return builder.utf8Value(bytes.array(), bytes.arrayOffset() + bytes.position(), bytes.remaining());
        }
    }
}
