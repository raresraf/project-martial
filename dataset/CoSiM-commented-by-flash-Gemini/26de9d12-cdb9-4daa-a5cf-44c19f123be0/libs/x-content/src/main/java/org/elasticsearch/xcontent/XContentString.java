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
  * @brief Functional description of the XContentString interface.
  *        This is a placeholder for detailed semantic documentation.
  *        Further analysis will elaborate on its algorithm, complexity, and invariants.
  */
public interface XContentString {
    record UTF8Bytes(byte[] bytes, int offset, int length) implements Comparable<UTF8Bytes> {
        public UTF8Bytes(byte[] bytes) {
            this(bytes, 0, bytes.length);
        }

        @Override
        public int compareTo(UTF8Bytes o) {
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (this.bytes == o.bytes && this.offset == o.offset && this.length == o.length) {
                return 0;
            }

            return ByteBuffer.wrap(bytes, offset, length).compareTo(ByteBuffer.wrap(o.bytes, o.offset, o.length));
        }

        @Override
        public boolean equals(Object o) {
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (this == o) {
                return true;
            }
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (o == null || getClass() != o.getClass()) {
                return false;
            }

            return this.compareTo((UTF8Bytes) o) == 0;
        }

        @Override
        public int hashCode() {
            return ByteBuffer.wrap(bytes, offset, length).hashCode();
        }
    }

    /**
     * Returns a {@link String} view of the data.
     */
    String string();

    /**
     * Returns an encoded {@link UTF8Bytes} view of the data.
     */
    UTF8Bytes bytes();

    /**
     * Returns the number of characters in the represented string.
     */
    int stringLength();
}
