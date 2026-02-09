/**
 * @file TestBuildInfoPluginFuncTest.groovy
 * @brief Functional tests for the `elasticsearch.test-build-info` Gradle plugin.
 *
 * This file contains functional tests written in Groovy using the Spock framework
 * to verify the behavior of the `generateTestBuildInfo` task provided by the
 * custom Gradle plugin.
 */
package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper

import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

import java.nio.file.Path

/**
 * A functional test class for the TestBuildInfoPlugin.
 * It uses a Gradle test harness (`AbstractGradleFuncTest`) to execute Gradle builds
 * in temporary project directories and assert the outcomes.
 */
class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    /**
     * Tests the basic functionality of the `generateTestBuildInfo` task.
     *
     * This test case sets up a simple Java project with a `module-info.java` file.
     * It then runs the `generateTestBuildInfo` task and verifies that the generated
     * JSON output correctly identifies the component name and the module location
     * based on the project's own source code.
     */
    def "basic functionality"() {
        given:
        // Setup a simple Java source file and a corresponding JPMS module descriptor.
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

        // Configure the Gradle build to apply the necessary plugins and configure the task.
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

        when:
        // Execute the Gradle task.
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        then:
        // Assert that the task executed successfully.
        task.outcome == TaskOutcome.SUCCESS

        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true

        // Assert that the content of the generated JSON file matches the expected structure and data.
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
     * Tests the plugin's ability to derive module information from project dependencies.
     *
     * This test case configures a project with several dependencies that provide module
     * information in different ways:
     * 1. A fully modular JAR with `module-info.class` (asm).
     * 2. A JAR with an `Automatic-Module-Name` entry in its manifest (junit).
     * 3. A traditional JAR with no module information, forcing a fallback to the JAR filename (hamcrest-core).
     *
     * It verifies that the `generateTestBuildInfo` task correctly identifies the module name
     * and a representative class for each of these cases.
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

        when:
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        then:
        task.outcome == TaskOutcome.SUCCESS

        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true

        // Define the expected module location data for each dependency type.
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
        
        // Assert that the generated JSON content matches the expected output.
        def value = new ObjectMapper().readValue(output, Map.class)
        value == expectedOutput
    }
}