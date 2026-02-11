/**
 * This file contains functional tests for the `elasticsearch.test-build-info` Gradle plugin.
 *
 * These tests use the Gradle TestKit (`GradleRunner`) to execute real Gradle builds
 * in temporary projects and verify the output of the plugin.
 */
package org.elasticsearch.gradle.test

import com.fasterxml.jackson.databind.ObjectMapper

import org.elasticsearch.gradle.fixtures.AbstractGradleFuncTest
import org.gradle.testkit.runner.TaskOutcome

class TestBuildInfoPluginFuncTest extends AbstractGradleFuncTest {
    /**
     * Tests the basic functionality of the plugin with a simple local Java module.
     *
     * This test verifies that the `generateTestBuildInfo` task correctly identifies
     * the module name and a representative class from a project's own source code
     * when a `module-info.java` file is present.
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

        when:
        def result = gradleRunner('generateTestBuildInfo').build()
        def task = result.task(":generateTestBuildInfo")


        then:
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
     * Tests the plugin's ability to resolve module information from external dependencies
     * with different Java Platform Module System (JPMS) characteristics.
     *
     * This test ensures the plugin can correctly identify module names from:
     * 1. A JAR with an explicit `module-info.class` (org.ow2.asm:asm).
     * 2. A JAR with an `Automatic-Module-Name` entry in its manifest (junit:junit).
     * 3. A traditional JAR with no module information, where the name must be
     *    inferred from the JAR filename (hamcrest-core).
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

        def value = new ObjectMapper().readValue(output, Map.class)
        value == expectedOutput
    }
}
