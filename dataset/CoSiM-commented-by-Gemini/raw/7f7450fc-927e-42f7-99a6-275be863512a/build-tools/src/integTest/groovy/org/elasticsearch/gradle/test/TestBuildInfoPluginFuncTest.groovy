
package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper

import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

/**
 * @brief Functional tests for the `elasticsearch.test-build-info` Gradle plugin.
 *
 * This test class verifies the functionality of a Gradle plugin that generates
 * build information in JSON format. The plugin's purpose is to inspect the
 * compile classpath of a project, identify the Java module information for each
 * component (both the project's own source and its dependencies), and record
 * that information along with a representative class from that component.
 *
 * These tests use the Gradle TestKit (`GradleRunner`) to execute real Gradle
 * builds in temporary project directories and assert the outputs.
 */
class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    
    /**
     * @test "basic functionality"
     * @brief Verifies that the plugin correctly processes the project's own source code.
     *
     * This test case checks the simplest scenario where the project has its own
     * `module-info.java` file. It ensures the plugin generates a JSON file
     * containing the correct module name and a representative class for the
     * project's sources.
     */
    def "basic functionality"() {
        given: "A simple Java project with a module-info.java and the build info plugin applied"
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


        then: "The task succeeds and the generated JSON file contains the correct information"
        task.outcome == TaskOutcome.SUCCESS

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
     * @test "dependencies"
     * @brief Verifies that the plugin can correctly derive module information from various types of dependencies.
     *
     * This test case is crucial as it checks the robustness of the plugin's module
     * detection logic against real-world scenarios:
     * 1. A proper Java 9+ module (`module-info.class`).
     * 2. A legacy JAR with an `Automatic-Module-Name` in its manifest.
     * 3. A legacy JAR with no module information, where the name must be inferred from the filename.
     */
    def "dependencies"() {
        given: "A project with three different types of dependencies"
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


        then: "The task succeeds and the generated JSON file correctly identifies all three dependencies"
        task.outcome == TaskOutcome.SUCCESS

        def output = file("build/generated-build-info/plugin-test-build-info.json")
        output.exists() == true

        // 1. Module name derived from module-info.class
        def locationFromModuleInfo = Map.of(
            "module", "org.objectweb.asm",
            "representative_class", 'org/objectweb/asm/AnnotationVisitor.class'
        )
        // 2. Module name derived from MANIFEST.MF's Automatic-Module-Name
        def locationFromManifest = Map.of(
            "module", "junit",
            "representative_class", 'junit/textui/TestRunner.class'
        )
        // 3. Module name derived from the JAR filename
        def locationFromJarFileName = Map.of(
            "module", "hamcrest.core",
            "representative_class", 'org/hamcrest/BaseDescription.class'
        )
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(locationFromModuleInfo, locationFromManifest, locationFromJarFileName)
        )

        def value = new ObjectMapper().readValue(output, Map.class)
        // Using direct comparison to assert the structure and content of the JSON output.
        value == expectedOutput
    }
}
