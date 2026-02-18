/**
 * @file TestBuildInfoPluginFuncTest.groovy
 * @brief Functional tests for the `elasticsearch.test-build-info` Gradle plugin.
 *
 * This file contains integration tests to verify the functionality of the
 * `elasticsearch.test-build-info` Gradle plugin. It ensures that the plugin
 * correctly generates a JSON file containing build information, including
 * component names, and details about Java module locations and their representative classes.
 * It covers scenarios for basic Java projects and projects with various types of dependencies
 * (e.g., those with `module-info.class`, `Automatic-Module-Name` in manifest, or inferred from JAR names).
 */
package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper

import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

import java.nio.file.Path

/**
 * @brief Functional test class for the `elasticsearch.test-build-info` Gradle plugin.
 * Extends `AbstractGradleFuncTest` to leverage common setup for Gradle functional tests.
 */
class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    /**
     * @brief Tests the basic functionality of the `elasticsearch.test-build-info` plugin.
     * Verifies that the plugin correctly generates build information for a simple Java project
     * with a `module-info.java` and a basic class.
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

        // Block Logic: Defines the expected structure and content for the 'locations' entry in the JSON output.
        def location = Map.of(
            "module", "com.example",
            "representative_class", "com/example/Example.class"
        )
        // Block Logic: Defines the overall expected JSON output structure.
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(location)
        )
        // Block Logic: Deserializes the generated JSON file and compares it with the expected output.
        new ObjectMapper().readValue(output, Map.class) == expectedOutput
    }

    /**
     * @brief Tests the plugin's ability to extract build information from various types of dependencies.
     * This includes dependencies that define modules via `module-info.class`, `Automatic-Module-Name`
     * in their JAR manifest, or by inferring from the JAR file name.
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
            // We're not actually running this code.
            implementation "org.ow2.asm:asm:9.7.1" // has module-info.class
            implementation "junit:junit:4.13" // has Automatic-Module-Name, and brings in hamcrest which does not
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

        // Block Logic: Defines the expected location entry for a dependency with `module-info.class`.
        def locationFromModuleInfo = Map.of(
            "module", "org.objectweb.asm",
            "representative_class", Path.of('org', 'objectweb', 'asm', 'AnnotationVisitor.class').toString()
        )
        // Block Logic: Defines the expected location entry for a dependency with `Automatic-Module-Name` in manifest.
        def locationFromManifest = Map.of(
            "module", "junit",
            "representative_class", Path.of('junit', 'textui', 'TestRunner.class').toString()
        )
        // Block Logic: Defines the expected location entry for a dependency where module name is inferred from JAR file name.
        def locationFromJarFileName = Map.of(
            "module", "hamcrest.core",
            "representative_class", Path.of('org', 'hamcrest', 'BaseDescription.class').toString()
        )
        // Block Logic: Defines the overall expected JSON output structure, including multiple dependency locations.
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(locationFromModuleInfo, locationFromManifest, locationFromJarFileName)
        )

        // Block Logic: Deserializes the generated JSON file and compares it with the expected output.
        def value = new ObjectMapper().readValue(output, Map.class)
        value == expectedOutput
    }
}
