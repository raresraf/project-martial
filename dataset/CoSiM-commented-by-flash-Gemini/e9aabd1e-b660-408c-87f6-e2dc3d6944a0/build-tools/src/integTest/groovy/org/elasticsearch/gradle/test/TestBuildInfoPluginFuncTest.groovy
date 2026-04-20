/**
 * @e9aabd1e-b660-408c-87f6-e2dc3d6944a0/build-tools/src/integTest/groovy/org/elasticsearch/gradle/test/TestBuildInfoPluginFuncTest.groovy
 * @brief Functional verification suite for the `elasticsearch.test-build-info` Gradle plugin.
 * 
 * Functional Intent: Ensures that the build information plugin correctly metadata-tags 
 * Java modules during the build process. It validates the extraction of module names 
 * and representative classes from project sources and external dependencies (via 
 * module-info, manifest headers, or JAR name heuristics), producing a deterministic 
 * JSON descriptor used for downstream testing and isolation.
 * 
 * Domain: Build Engineering, Gradle Plugins, Java Module System (JPMS).
 */

package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper
import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

/**
 * @brief Integration tests for JPMS-aware build metadata generation.
 */
class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    
    /**
     * Block Logic: Validates metadata extraction for local project modules.
     * Logic: 
     * 1. Scaffolds a minimal Java module with a package and module-info.
     * 2. Executes the 'generateTestBuildInfo' task.
     * 3. Verifies that the produced JSON aligns with the project's structural identity.
     */
    def "basic functionality"() {
        given: "A simple Java project with a module-info and an example class"
        file("src/main/java/com/example/Example.java") << """
            package com.example;

            public class Example {
            }
        """

        file("src/main/java/module-info.java") << """
            module com.example {
                exports com.example;
            }
        """

        buildFile << """
        import org.elasticsearch.gradle.plugin.GenerateTestBuildInfoTask;

        plugins {
            id 'java'
            id 'elasticsearch.test-build-info'
        }

        repositories {
            mavenCentral()
        }

        tasks.withType(GenerateTestBuildInfoTask.class) {
            componentName = 'example-component'
            outputFile = new File('build/generated-build-info/plugin-test-build-info.json')
        }
        """

        when: "The generateTestBuildInfo Gradle task is executed"
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        then: "The task completes successfully and the build info JSON file is generated with expected content"
        task.outcome == TaskOutcome.SUCCESS

        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true

        // Functional Utility: Verification of local module metadata serialization.
        def location = Map.of(
            "module", "com.example",
            "representative_class", "com/example/Example.class"
        )
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(location)
        )
        new ObjectMapper().readValue(output, Map.class) == expectedOutput
    }

    /**
     * Block Logic: Validates metadata extraction across various external dependency types.
     * Logic: 
     * 1. Includes specific dependencies showcasing:
     *    - Explicit JPMS (asm).
     *    - Automatic-Module-Name (junit).
     *    - Inferred module naming (hamcrest).
     * 2. Asserts that the plugin correctly resolves and maps these distinct patterns 
     *    to their respective representative classes.
     */
    def "dependencies"() {
        buildFile << """
        import org.elasticsearch.gradle.plugin.GenerateTestBuildInfoTask;

        plugins {
            id 'java'
            id 'elasticsearch.test-build-info'
        }

        repositories {
            mavenCentral()
        }

        dependencies {
            // We pin to specific versions here because they are known to have the properties we want to test.
            implementation "org.ow2.asm:asm:9.7.1" 
            implementation "junit:junit:4.13" 
        }

        tasks.withType(GenerateTestBuildInfoTask.class) {
            componentName = 'example-component'
            outputFile = new File('build/generated-build-info/plugin-test-build-info.json')
        }
        """

        when: "The generateTestBuildInfo Gradle task is executed with dependencies"
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        then: "The task completes successfully and the build info JSON file is generated with expected dependency information"
        task.outcome == TaskOutcome.SUCCESS

        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true

        // Block Logic: Mapping of dependency-specific resolution rules.
        def locationFromModuleInfo = Map.of(
            "module", "org.objectweb.asm",
            "representative_class", 'org/objectweb/asm/AnnotationVisitor.class'
        )
        def locationFromManifest = Map.of(
            "module", "junit",
            "representative_class", 'junit/textui/TestRunner.class'
        )
        def locationFromJarFileName = Map.of(
            "module", "hamcrest.core",
            "representative_class", 'org/hamcrest/BaseDescription.class'
        )
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(locationFromModuleInfo, locationFromManifest, locationFromJarFileName)
        )

        // Final assertion ensures that the heterogeneous dependency set was correctly serialized.
        def value = new ObjectMapper().readValue(output, Map.class)
        value == expectedOutput
    }
}
