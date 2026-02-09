package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper

import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

/**
 * Functional test for the {@code elasticsearch.test-build-info} plugin.
 * This test verifies that the {@code generateTestBuildInfo} task correctly
 * generates a JSON file containing build information about the project and its
 * dependencies. It uses the Gradle TestKit to run a real Gradle build and

 * asserts the output of the task.
 */
class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    /**
     * Tests the basic functionality of the plugin with a simple project that has no dependencies.
     * It checks if the generated JSON file contains the correct component name and location
     * information for the project's own module.
     */
    def "basic functionality"() {
        given:
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

        def location = Map.of(
            "module", "com.example",
            "representative_class", "com/example/Example.class"
        )
        def expectedOutput = Map.of(
            "component", "example-component",
            "locations", List.of(location)
        )

        def output = file("build/generated-build-info/plugin-test-build-info.json")

        when:
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")

        then:
        task.outcome == TaskOutcome.SUCCESS
        output.exists() == true
        new ObjectMapper().readValue(output, Map.class) == expectedOutput
    }

    /**
     * Tests how the plugin handles project dependencies. It verifies that the generated
     * JSON file correctly identifies the module information for dependencies that have:
     * 1. A {@code module-info.class} (JPMS module).
     * 2. An {@code Automatic-Module-Name} in their manifest.
     * 3. Neither of the above (in which case the module name is derived from the JAR file name).
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

        def output = file("build/generated-build-info/plugin-test-build-info.json")

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

        when:
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")

        then:
        task.outcome == TaskOutcome.SUCCESS
        output.exists() == true
        new ObjectMapper().readValue(output, Map.class) == expectedOutput
    }
}
