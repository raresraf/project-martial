/*
 * @file TestBuildInfoPluginFuncTest.groovy
 * @brief Functional tests for the Elasticsearch Gradle Build Info plugin.
 *
 * This file contains functional tests for a Gradle plugin that generates
 * build information. The tests verify that the plugin correctly identifies
 * module and class locations based on different build configurations and
 * dependency types (e.g., those with module-info, Automatic-Module-Name, or
 * inferred from JAR file names).
 *
 * The tests use Gradle TestKit to execute Gradle builds and assert
 * the generated JSON output for build information.
 */
package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper

import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

import java.nio.file.Path

/**
 * @8fe6a290-95ed-4a4d-8aad-c9638d2e0789/build-tools/src/integTest/groovy/org/elasticsearch/gradle/test/TestBuildInfoPluginFuncTest.groovy
 * @brief Functional tests for the Elasticsearch Gradle Build Info plugin.
 *
 * This class contains integration tests to verify the correct behavior of the
 * {@code elasticsearch.test-build-info} Gradle plugin, ensuring it accurately
 * generates build information in various scenarios, including basic project
 * structures and complex dependency configurations.
 */
class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    /**
     * @brief Tests the basic functionality of the build info plugin.
     *
     * This test case verifies that the {@code elasticsearch.test-build-info} plugin
     * correctly generates build information for a simple Java project with a {@code module-info.java}.
     * It sets up a basic project structure, applies the plugin, runs the
     * {@code generateTestBuildInfo} task, and asserts that the output JSON file
     * exists and contains the expected component and location information.
     */
    def "basic functionality"() {
        given: "A simple Java project with a module-info.java and a basic class"
        // Block Logic: Creates a Java source file to be included in the test project.
        file("src/main/java/com/example/Example.java") << """
            package com.example;

            public class Example {
            }
        """

        // Block Logic: Defines a module-info.java file for the test project,
        // explicitly declaring the module 'com.example'.
        file("src/main/java/module-info.java") << """
            module com.example {
                exports com.example;
            }
        """

        // Block Logic: Configures the build.gradle file to apply the Java plugin
        // and the elasticsearch.test-build-info plugin. It also sets up the
        // output file for the generated build information.
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

        when: "The `generateTestBuildInfo` task is executed"
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        then: "The task completes successfully and produces the expected JSON output"
        task.outcome == TaskOutcome.SUCCESS

        // Block Logic: Verifies that the output file exists and its content matches
        // the expected build information, including component name and module location.
        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true

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
     * @brief Verifies the plugin's ability to extract build information from project dependencies.
     *
     * This test case asserts that the {@code elasticsearch.test-build-info} plugin
     * correctly identifies and extracts module and representative class information
     * from various types of dependencies:
     * - Dependencies with an explicit {@code module-info.class} (e.g., ASM).
     * - Dependencies with an {@code Automatic-Module-Name} in their manifest (e.g., JUnit).
     * - Dependencies where module information is inferred from the JAR file name (e.g., Hamcrest).
     * The test explicitly pins dependency versions to ensure predictable properties for testing.
     */
    def "dependencies"() {
        given: "A Gradle project with various types of dependencies"
        // Block Logic: Configures the build.gradle file to apply necessary plugins
        // and declare various dependencies to test module information extraction.
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
            // Inline: Pinning to specific versions to ensure consistent testing
            // against known module information characteristics. These dependencies
            // are not actually executed but are analyzed for build info generation.
            implementation "org.ow2.asm:asm:9.7.1" // has module-info.class
            implementation "junit:junit:4.13" // has Automatic-Module-Name, and brings in hamcrest which does not
        }

        tasks.withType(GenerateTestBuildInfoTask.class) {
            componentName = 'example-component'
            outputFile = new File('build/generated-build-info/plugin-test-build-info.json')
        }
        """

        when: "The `generateTestBuildInfo` task is executed"
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        then: "The task completes successfully and produces the expected JSON output with dependency module information"
        task.outcome == TaskOutcome.SUCCESS

        // Block Logic: Verifies that the output file exists and its content contains
        // the expected module and representative class information for each dependency.
        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true

        // Block Logic: Defines the expected location maps for dependencies,
        // distinguishing how their module information is derived.
        def locationFromModuleInfo = Map.of(
            "module", "org.objectweb.asm",
            "representative_class", Path.of('org', 'objectweb', 'asm', 'AnnotationVisitor.class').toString()
        )
        def locationFromManifest = Map.of(
            "module", "junit",
            "representative_class", Path.of('junit', 'textui', 'TestRunner.class').toString()
        )
        def locationFromJarFileName = Map.of(
            "module", "hamcrest.core",
            "representative_class", Path.of('org', 'hamcrest', 'BaseDescription.class').toString()
        )
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(locationFromModuleInfo, locationFromManifest, locationFromJarFileName)
        )

        // Block Logic: Compares the actual generated JSON output with the predefined expected output.
        def value = new ObjectMapper().readValue(output, Map.class)
        expectedOutput.forEach((k,v) -> value.get(k) == v)
        value == expectedOutput
    }
}
