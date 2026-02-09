/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
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
 * Unit tests for the {@link TestBuildInfoParser} class, ensuring that build
 * information for test components is parsed correctly from its JSON representation.
 */
public class TestBuildInfoParserTests extends ESTestCase {
    /**
     * Verifies that a standard JSON object representing a test component's build
     * information is parsed correctly.
     * <p>
     * This test checks the following:
     * <ul>
     *     <li>The top-level "component" name is parsed accurately.</li>
     *     <li>The list of "locations" is deserialized into a collection of
     *     {@link TestBuildInfoLocation} objects.</li>
     *     <li>The module names and representative class paths within each location
     *     are correctly extracted and match the expected values.</li>
     * </ul>
     *
     * @throws IOException if there is an error creating the JSON parser.
     */
    public void testSimpleParsing() throws IOException {

        // A JSON string representing the build metadata for a test component.
        var input = """
            {
                "component": "lang-painless",
                "locations": [
                    {
                        "representativeClass": "Location.class",
                        "module": "org.elasticsearch.painless"
                    },
                    {
                        "representativeClass": "org/objectweb/asm/AnnotationVisitor.class",
                        "module": "org.objectweb.asm"
                    },
                    {
                        "representativeClass": "org/antlr/v4/runtime/ANTLRErrorListener.class",
                        "module": "org.antlr.antlr4.runtime"
                    },
                    {
                        "representativeClass": "org/objectweb/asm/commons/AdviceAdapter.class",
                        "module": "org.objectweb.asm.commons"
                    }
                ]
            }
            """;

        // Block Logic: Parse the JSON input and deserialize it into a TestBuildInfo object.
        try (var parser = XContentFactory.xContent(XContentType.JSON).createParser(XContentParserConfiguration.EMPTY, input)) {
            var testInfo = TestBuildInfoParser.fromXContent(parser);

            // Assertion: Verify that the component name is correctly parsed.
            assertThat(testInfo.component(), is("lang-painless"));

            // Assertion: Verify that all module names are correctly extracted from the locations list.
            assertThat(
                testInfo.locations(),
                transformedItemsMatch(
                    TestBuildInfoLocation::module,
                    contains("org.elasticsearch.painless", "org.objectweb.asm", "org.antlr.antlr4.runtime", "org.objectweb.asm.commons")
                )
            );

            // Assertion: Verify that all representative class paths are correctly extracted.
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