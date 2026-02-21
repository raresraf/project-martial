package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper

import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

/**
 * Functional tests for the `elasticsearch.test-build-info` Gradle plugin.
 * These tests verify that the plugin correctly generates build information,
 * including module data from various sources within a project.
 */
class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    /**
     * Tests the basic functionality of the build info plugin, ensuring it can generate
     * build information for a simple Java project with a module-info.java file.
     */
    def "basic functionality"() {
        // Given a Java project setup
        given:
        // Create a dummy Java source file.
        file("src/main/java/com/example/Example.java") << """
            package com.example;

            public class Example {
            }
        """

        // Create a module-info.java file for the project.
        file("src/main/java/module-info.java") << """
            module com.example {
                exports com.example;
            }
        """

        // Configure the Gradle build file to apply the necessary plugins
        // and configure the GenerateTestBuildInfoTask.
        buildFile << """
        import org.elasticsearch.gradle.plugin.GenerateTestBuildInfoTask;

        plugins {
            id 'java'
            id 'elasticsearch.test-build-info'
        }

        repositories {
            mavenCentral()
        }

        // Configure the GenerateTestBuildInfoTask provided by the plugin.
        tasks.withType(GenerateTestBuildInfoTask.class) {
            componentName = 'example-component' // Set the component name for the build info.
            outputFile = new File('build/generated-build-info/plugin-test-build-info.json') // Define output file path.
        }
        """

        // When the 'generateTestBuildInfo' Gradle task is executed
        when:
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        // Then the task should succeed and produce the expected output.
        then:
        task.outcome == TaskOutcome.SUCCESS // Assert that the task completed successfully.

        // Verify the output file exists and its content.
        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true // Assert that the output JSON file was created.

        // Define the expected 'location' structure within the JSON.
        def location = Map.of(
            "module", "com.example", // Expected module name from module-info.java.
            "representative_class", "com/example/Example.class" // Expected representative class.
        )
        // Define the full expected output JSON structure.
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(location)
        )
        // Parse the generated JSON and compare it to the expected output.
        new ObjectMapper().readValue(output, Map.class) == expectedOutput
    }

    /**
     * Tests how the build info plugin processes different types of dependencies
     * and correctly extracts module information (or derives it) from them.
     */
    def "dependencies"() {
        // Given a Gradle build with specific dependencies
        given:
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
            // Dependency with a module-info.class (JPMS module).
            implementation "org.ow2.asm:asm:9.7.1"
            // Dependency with an Automatic-Module-Name in its manifest.
            // Also brings in 'hamcrest' which has no module info and should be derived from JAR name.
            implementation "junit:junit:4.13"
        }

        // Configure the GenerateTestBuildInfoTask.
        tasks.withType(GenerateTestBuildInfoTask.class) {
            componentName = 'example-component'
            outputFile = new File('build/generated-build-info/plugin-test-build-info.json')
        }
        """

        // When the 'generateTestBuildInfo' Gradle task is executed
        when:
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        // Then the task should succeed and produce build info reflecting dependency module details.
        then:
        task.outcome == TaskOutcome.SUCCESS // Assert task success.

        // Verify the output file exists and its content.
        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true // Assert output file creation.

        // Expected location data for a dependency with module-info.class.
        def locationFromModuleInfo = Map.of(
            "module", "org.objectweb.asm",
            "representative_class", 'org/objectweb/asm/AnnotationVisitor.class'
        )
        // Expected location data for a dependency with Automatic-Module-Name manifest entry.
        def locationFromManifest = Map.of(
            "module", "junit",
            "representative_class", 'junit/textui/TestRunner.class'
        )
        // Expected location data for a dependency where module name is derived from JAR file name.
        def locationFromJarFileName = Map.of(
            "module", "hamcrest.core",
            "representative_class", 'org/hamcrest/BaseDescription.class'
        )
        // Define the full expected output, including all three dependency types.
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(locationFromModuleInfo, locationFromManifest, locationFromJarFileName)
        )

        // Parse and compare the actual output with the expected output.
        def value = new ObjectMapper().readValue(output, Map.class)
        value == expectedOutput
    }
}
