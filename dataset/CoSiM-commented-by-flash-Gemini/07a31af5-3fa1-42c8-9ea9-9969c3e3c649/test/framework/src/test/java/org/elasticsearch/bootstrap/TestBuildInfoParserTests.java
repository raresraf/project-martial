/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

/**
 * @file TestBuildInfoParserTests.java
 * @brief Unit tests for the TestBuildInfoParser class.
 *
 * This file contains comprehensive unit tests to ensure the correct functionality
 * of the {@link TestBuildInfoParser} class. It specifically focuses on validating
 * the parser's ability to accurately extract and interpret build-related
 * information from various XContent (e.g., JSON) inputs, thereby ensuring the
 * integrity and reliability of build data handling within the system.
 */

package org.elasticsearch.bootstrap;

import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentParserConfiguration;
import org.elasticsearch.xcontent.XContentType;

import java.io.IOException;

import static org.elasticsearch.test.LambdaMatchers.transformedItemsMatch;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.is;

/**
 * @brief Unit tests for {@link TestBuildInfoParser}.
 *
 * This class serves as a testing suite for the {@link TestBuildInfoParser},
 * ensuring that its methods for parsing build-related information from various
 * data formats (e.g., XContent like JSON) function as expected. The tests
 * validate the parser's ability to correctly extract and interpret build
 * component details and their associated locations.
 */
public class TestBuildInfoParserTests extends ESTestCase {
    /**
     * @brief Tests the basic parsing functionality of {@link TestBuildInfoParser}.
     * @throws IOException If an I/O error occurs during XContent parsing.
     *
     * Functional Utility: This test case provides a simple, well-formed XContent (JSON)
     * input string representing build information. It then verifies that the
     * {@link TestBuildInfoParser#fromXContent(org.elasticsearch.xcontent.XContentParser)}
     * method correctly extracts the component name ("lang-painless") and
     * accurately parses the list of {@link TestBuildInfoLocation} objects,
     * confirming that the module names and representative class paths are as expected.
     */
    public void testSimpleParsing() throws IOException {

        var input = """
            {
                "component": "lang-painless",
                "locations": [
                    {
                        "representative_class": "Location.class",
                        "module": "org.elasticsearch.painless"
                    },
                    {
                        "representative_class": "org/objectweb/asm/AnnotationVisitor.class",
                        "module": "org.objectweb.asm"
                    },
                    {
                        "representative_class": "org/antlr/v4/runtime/ANTLRErrorListener.class",
                        "module": "org.antlr.antlr4.runtime"
                    },
                    {
                        "representative_class": "org/objectweb/asm/commons/AdviceAdapter.class",
                        "module": "org.objectweb.asm.commons"
                    }
                ]
            }
            """;

        try (var parser = XContentFactory.xContent(XContentType.JSON).createParser(XContentParserConfiguration.EMPTY, input)) {
            var testInfo = TestBuildInfoParser.fromXContent(parser);
            assertThat(testInfo.component(), is("lang-painless"));
            assertThat(
                testInfo.locations(),
                transformedItemsMatch(
                    TestBuildInfoLocation::module,
                    contains("org.elasticsearch.painless", "org.objectweb.asm", "org.antlr.antlr4.runtime", "org.objectweb.asm.commons")
                )
            );

            assertThat(
                testInfo.locations(),
                transformedItemsMatch(
                    TestBuildInfoLocation::representativeClass,
                    contains(
                        "Location.class",
                        "org/objectweb/asm/AnnotationVisitor.class",
                        "org/antlr/v4/runtime/ANTLRErrorListener.class",
                        "org/objectweb/asm/commons/AdviceAdapter.class"
                    )
                )
            );
        }
    }
}
