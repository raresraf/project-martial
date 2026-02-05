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
 * Unit tests for the {@link TestBuildInfoParser} class, ensuring that component
 * build information is correctly parsed from its JSON representation.
 */
public class TestBuildInfoParserTests extends ESTestCase {
    /**
     * Tests the parsing of a standard JSON input representing build information for a component.
     * This test verifies that the component name and its associated module locations (including
     * representative classes and module names) are deserialized correctly.
     *
     * @throws IOException if an error occurs during JSON parsing.
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

        // Pre-condition: The input is a valid JSON object.
        try (var parser = XContentFactory.xContent(XContentType.JSON).createParser(XContentParserConfiguration.EMPTY, input)) {
            var testInfo = TestBuildInfoParser.fromXContent(parser);
            // Post-condition: Assert that the component name is parsed correctly.
            assertThat(testInfo.component(), is("lang-painless"));
            
            // Post-condition: Assert that all module names are extracted in the correct order.
            assertThat(
                testInfo.locations(),
                transformedItemsMatch(
                    TestBuildInfoLocation::module,
                    contains("org.elasticsearch.painless", "org.objectweb.asm", "org.antlr.antlr4.runtime", "org.objectweb.asm.commons")
                )
            );

            // Post-condition: Assert that all representative class names are extracted in the correct order.
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