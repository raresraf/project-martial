/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.xcontent;

import java.nio.ByteBuffer;

/**
 * @391d40c8-9021-4519-941e-3c1e7d277aa6/libs/x-content/src/main/java/org/elasticsearch/xcontent/XContentString.java
 * @brief Interface for optimized string representations during XContent serialization.
 * 
 * Functional Intent: Provides a dual-view abstraction (UTF-16 String and UTF-8 ByteBuffer) 
 * for string data. It allows serialization engines to avoid redundant UTF-8 encoding 
 * steps by providing direct access to pre-encoded byte buffers.
 */
public interface XContentString {
    /**
     * @brief Accessor for the high-level Java String representation.
     * @return Standard UTF-16 encoded string.
     */
    String string();

    /**
     * @brief Accessor for the low-level UTF-8 encoded byte representation.
     * Logic: Returns a read-only buffer suitable for direct I/O without extra allocations.
     * @return ByteBuffer containing the UTF-8 encoded sequence.
     */
    ByteBuffer bytes();

    /**
     * @brief Returns the character count (not byte count) of the sequence.
     */
    int stringLength();
}
