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
 * @31a602a1-a8a2-4e1b-8c6b-ba63299d1680/test/framework/src/test/java/org/elasticsearch/bootstrap/TestBuildInfoParserTests.java
 * @brief Unit tests for the TestBuildInfoParser class.
 * 
 * Functional Intent: Validates the deserialization of build information from JSON 
 * formats used during the Elasticsearch bootstrap process. Ensures that component 
 * names and class-to-module mappings are correctly extracted and structured.
 */
public class TestBuildInfoParserTests extends ESTestCase {
    
    /**
     * Block Logic: Validates the core parsing logic for build metadata.
     * Logic: Defines a raw JSON input representing component locations, parses it 
     * using XContent, and asserts that the resulting TestBuildInfo object correctly 
     * reflects the input structure (component name and ordered location mappings).
     * 
     * @throws IOException If there is an error during XContent parsing.
     */
    public void testSimpleParsing() throws IOException {

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
