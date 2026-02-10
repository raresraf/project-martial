/**
 * @file TestBuildInfoPluginFuncTest.groovy
 * @brief Functional test for the 'elasticsearch.test-build-info' Gradle plugin.
 * @details This test verifies the functionality of the GenerateTestBuildInfoTask, ensuring that it correctly
 *          generates a JSON file containing build information about the project and its dependencies.
 *          It uses the Gradle TestKit for running an embedded Gradle build.
 */
package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper

import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

import java.nio.file.Path

class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    /**
     * @brief Tests the basic functionality of the plugin with a simple local component.
     * @details This test case ensures that the plugin can correctly identify the module name and a representative
     *          class from a simple Java source set that includes a 'module-info.java' file.
     */
    def "basic functionality"() {
        given: "A simple Java project with a module-info.java"
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

        when: "The generateTestBuildInfo task is executed"
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        then: "The task succeeds and generates the correct JSON output"
        task.outcome == TaskOutcome.SUCCESS

        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true

        // Block Logic: Verifies that the generated JSON file contains the expected component name,
        // module name, and a path to a representative class file.
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
     * @brief Tests the plugin's ability to derive module information from project dependencies.
     * @details This test case verifies three different mechanisms for identifying module information from JARs:
     *          1. From 'module-info.class' (for full Java modules).
     *          2. From the 'Automatic-Module-Name' attribute in the JAR's manifest.
     *          3. By deriving the module name from the JAR file name as a fallback.
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

        when: "The generateTestBuildInfo task is executed"
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        then: "The task succeeds and generates a JSON file with module info from all dependencies"
        task.outcome == TaskOutcome.SUCCESS

        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true

        // Block Logic: Defines the expected output for each dependency, corresponding to the three
        // different module identification strategies.
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

        // Block Logic: Asserts that the generated JSON matches the expected structure and content.
        def value = new ObjectMapper().readValue(output, Map.class)
        expectedOutput.forEach((k,v) -> value.get(k) == v)
        value == expectedOutput
    }
}