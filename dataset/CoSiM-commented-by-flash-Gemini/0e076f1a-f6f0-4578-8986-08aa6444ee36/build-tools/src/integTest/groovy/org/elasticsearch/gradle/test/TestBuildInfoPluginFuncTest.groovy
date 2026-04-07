// Package: org.elasticsearch.gradle.test
// @brief Integration tests for the Elasticsearch Build Info Gradle Plugin.
// This package contains Spock framework tests to verify the functional correctness
// of the `elasticsearch.test-build-info` Gradle plugin, ensuring it accurately
// generates build information for Java projects, including module and class locations.
// Domain: Gradle Plugin, Build System, Testing, Elasticsearch, Groovy/Spock.
package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper

import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

/**
 * @class TestBuildInfoPluginFuncTest
 * @brief Functional tests for the `elasticsearch.test-build-info` Gradle plugin.
 *
 * Functional Utility: Verifies that the Gradle plugin correctly extracts and
 *                     formats build information (component name, module locations)
 *                     for Java projects, handling both direct sources and dependencies.
 *                     It uses test-kit to run Gradle builds and assert on generated JSON output.
 * @augments AbstractGradleFuncTest
 */
class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    /**
     * @method basic functionality
     * @brief Tests the fundamental ability of the build info plugin to generate build metadata for a simple Java project.
     * Functional Utility: Verifies that the plugin correctly identifies the component name and
     *                     module location for a project with a basic `module-info.java` file.
     * Pre-condition: A Gradle project setup with `java` and `elasticsearch.test-build-info` plugins.
     * Post-condition: A JSON file containing accurate build information is generated and matches expected output.
     */
    def "basic functionality"() {
        given: "a simple Java project with a module-info.java and plugin configuration"
        // Block Logic: Defines a Java source file for the example component.
        file("src/main/java/com/example/Example.java") << """
            package com.example;

            public class Example {
            }
        """

        // Block Logic: Defines a module-info.java file to specify the Java module.
        file("src/main/java/module-info.java") << """
            module com.example {
                exports com.example;
            }
        """

        // Block Logic: Configures the Gradle build script to apply the Java and Elasticsearch build info plugins.
        //              It also customizes the `GenerateTestBuildInfoTask` with a component name and output file.
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

        // Block Logic: Defines the expected JSON structure for a single module location.
        def location = Map.of(
            "module", "com.example",
            "representative_class", "com/example/Example.class"
        )
        // Block Logic: Defines the complete expected JSON output, including the component name and module locations.
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(location)
        )

        // Block Logic: Specifies the path to the expected output JSON file.
        def output = file("build/generated-build-info/plugin-test-build-info.json")

        when: "the generateTestBuildInfo Gradle task is executed"
        // Block Logic: Runs the `generateTestBuildInfo` Gradle task and captures the result.
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")

        then: "the task succeeds and generates the expected build info JSON"
        // Assertion: Verifies that the Gradle task completed successfully.
        task.outcome == TaskOutcome.SUCCESS
        // Assertion: Verifies that the output JSON file was created.
        output.exists() == true
        // Assertion: Parses the generated JSON and asserts that its content matches the predefined expected output.
        new ObjectMapper().readValue(output, Map.class) == expectedOutput
    }

    /**
     * @method dependencies
     * @brief Tests the build info plugin's ability to correctly identify and report module locations from project dependencies.
     * Functional Utility: Verifies that the plugin can extract build information from JARs,
     *                     considering various sources like `module-info.class`, `Automatic-Module-Name`
     *                     in `MANIFEST.MF`, and even inferred names from JAR file names.
     * Pre-condition: A Gradle project setup with dependencies that exhibit different module naming conventions.
     * Post-condition: A JSON file is generated with accurate module locations for each analyzed dependency.
     */
    def "dependencies"() {
        given: "a Gradle project with various dependencies and plugin configuration"
        // Block Logic: Configures the Gradle build script to include multiple dependencies with different module metadata sources.
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
            // We're not actually running this code.
            implementation "org.ow2.asm:asm:9.7.1" // has module-info.class
            implementation "junit:junit:4.13" // has Automatic-Module-Name, and brings in hamcrest which does not
        }

        tasks.withType(GenerateTestBuildInfoTask.class) {
            componentName = 'example-component'
            outputFile = new File('build/generated-build-info/plugin-test-build-info.json')
        }
        """

        // Block Logic: Specifies the path to the expected output JSON file.
        def output = file("build/generated-build-info/plugin-test-build-info.json")

        // Block Logic: Defines the expected module location data for a dependency with `module-info.class`.
        def locationFromModuleInfo = Map.of(
            "module", "org.objectweb.asm",
            "representative_class", 'org/objectweb/asm/AnnotationVisitor.class'
        )
        // Block Logic: Defines the expected module location data for a dependency with `Automatic-Module-Name` in manifest.
        def locationFromManifest = Map.of(
            "module", "junit",
            "representative_class", 'junit/textui/TestRunner.class'
        )
        // Block Logic: Defines the expected module location data for a dependency where module name is inferred from JAR file.
        def locationFromJarFileName = Map.of(
            "module", "hamcrest.core",
            "representative_class", 'org/hamcrest/BaseDescription.class'
        )
        // Block Logic: Defines the complete expected JSON output, including the component name and all dependency locations.
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(locationFromModuleInfo, locationFromManifest, locationFromJarFileName)
        )

        when: "the generateTestBuildInfo Gradle task is executed"
        // Block Logic: Runs the `generateTestBuildInfo` Gradle task and captures the result.
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")

        then: "the task succeeds and generates the expected build info JSON for dependencies"
        // Assertion: Verifies that the Gradle task completed successfully.
        task.outcome == TaskOutcome.SUCCESS
        // Assertion: Verifies that the output JSON file was created.
        output.exists() == true
        // Assertion: Parses the generated JSON and asserts that its content matches the predefined expected output.
        new ObjectMapper().readValue(output, Map.class) == expectedOutput
    }}
