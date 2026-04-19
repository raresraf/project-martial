/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.gradle.plugin;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.SerializationFeature;

import org.gradle.api.DefaultTask;
import org.gradle.api.file.FileCollection;
import org.gradle.api.file.RegularFileProperty;
import org.gradle.api.provider.Property;
import org.gradle.api.tasks.CacheableTask;
import org.gradle.api.tasks.Classpath;
import org.gradle.api.tasks.Input;
import org.gradle.api.tasks.Optional;
import org.gradle.api.tasks.OutputFile;
import org.gradle.api.tasks.TaskAction;
import org.jetbrains.annotations.NotNull;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ModuleVisitor;
import org.objectweb.asm.Opcodes;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.CodeSource;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;

import static java.nio.file.FileVisitResult.CONTINUE;
import static java.nio.file.FileVisitResult.TERMINATE;

/**
 * @brief A Gradle task that generates a JSON file mapping code locations to their Java module names.
 *
 * This task inspects the classpath of a given source set (typically the test classpath).
 * For each JAR or directory on the classpath, it determines the corresponding Java module name
 * using a series of strategies. This information is then written to a JSON file, creating a
 * "semantic fingerprint" of the build's components, which is used to support Java module-aware
 * features like security policy lookups in a non-modular test environment.
 */
@CacheableTask
public abstract class GenerateTestBuildInfoTask extends DefaultTask {

    public static final String DESCRIPTION = "generates plugin test dependencies file";

    public static final String META_INF_VERSIONS_PREFIX = "META-INF/versions/";
    public static final String JAR_DESCRIPTOR_SUFFIX = ".jar";

    public GenerateTestBuildInfoTask() {
        setDescription(DESCRIPTION);
    }

    /**
     * @return The module name of the project's own sources, if explicitly defined.
     */
    @Input
    @Optional
    public abstract Property<String> getModuleName();

    /**
     * @return An arbitrary name for the component being built, used for grouping in the output file.
     */
    @Input
    public abstract Property<String> getComponentName();

    /**
     * @return The collection of files (JARs and directories) to be scanned.
     *         This is annotated with `@Classpath` to ensure Gradle provides a properly resolved classpath.
     */
    @Classpath
    public abstract Property<FileCollection> getCodeLocations();

    /**
     * @return The file where the final JSON output will be written.
     */
    @OutputFile
    public abstract RegularFileProperty getOutputFile();

    /**
     * @brief The main action of the Gradle task.
     *
     * This method is executed by Gradle. It orchestrates the process of scanning the classpath,
     * building the location list, and writing the final JSON output file.
     */
    @TaskAction
    public void generatePropertiesFile() throws IOException {
        Path outputFile = getOutputFile().get().getAsFile().toPath();
        Files.createDirectories(outputFile.getParent());

        try (var writer = Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8)) {
            ObjectMapper mapper = new ObjectMapper().configure(SerializationFeature.INDENT_OUTPUT, true)
                .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
            // Serialize the collected build info into the output file.
            mapper.writeValue(writer, new OutputFileContents(getComponentName().get(), buildLocationList()));
        }
    }

    /**
     * @brief A data class representing the top-level structure of the output JSON file.
     */
    record OutputFileContents(String component, List<Location> locations) {}

    /**
     * @brief A data class representing a single code location (a JAR or directory) and its module info.
     *
     * This is an analog of {@link CodeSource#getLocation()}, providing the necessary metadata
     * to simulate a modular environment.
     *
     * @param module              The determined Java module name.
     * @param representativeClass An example class file within this location, used for identification.
     */
    record Location(String module, String representativeClass) {}

    /**
     * @brief Builds the list of {@link Location}s by iterating through all code locations.
     * @return A list of populated {@link Location} objects.
     */
    private List<Location> buildLocationList() throws IOException {
        List<Location> locations = new ArrayList<>();
        for (File file : getCodeLocations().get().getFiles()) {
            if (file.exists()) {
                if (file.getName().endsWith(JAR_DESCRIPTOR_SUFFIX)) {
                    extractLocationsFromJar(file, locations);
                } else if (file.isDirectory()) {
                    extractLocationsFromDirectory(file, locations);
                } else {
                    throw new IllegalArgumentException("unrecognized classpath entry: " + file);
                }
            }
        }
        return List.copyOf(locations);
    }

    /**
     * @brief Extracts module and class information from a JAR file.
     */
    private void extractLocationsFromJar(File file, List<Location> locations) throws IOException {
        try (JarFile jarFile = new JarFile(file)) {
            var className = extractClassNameFromJar(jarFile);

            if (className.isPresent()) {
                // Determine module name using a series of fallback strategies.
                String moduleName = extractModuleNameFromJar(file, jarFile);
                locations.add(new Location(moduleName, className.get()));
            }
        }
    }

    /**
     * @brief Finds the first suitable "representative" class file within a JAR.
     */
    private java.util.Optional<String> extractClassNameFromJar(JarFile jarFile) {
        return jarFile.stream()
            .filter(
                je -> je.getName().startsWith("META-INF") == false
                    && je.getName().equals("module-info.class") == false
                    && je.getName().contains("$") == false // Avoid anonymous/inner classes
                    && je.getName().endsWith(".class")
            )
            .findFirst()
            .map(ZipEntry::getName);
    }

    /**
     * @brief Determines the module name from a JAR file using a multi-step strategy.
     *
     * This method mirrors the logic of {@link java.lang.module.ModuleFinder#of} by trying
     * different strategies in order of precedence.
     *
     * @return The determined module name.
     */
    private String extractModuleNameFromJar(File file, JarFile jarFile) throws IOException {
        String moduleName = null;

        // Strategy 1: Check for multi-release JARs and find the latest module-info.class.
        if (jarFile.isMultiRelease()) {
            StringBuilder dir = versionDirectoryIfExists(jarFile);
            if (dir != null) {
                dir.append("/module-info.class");
                moduleName = getModuleNameFromModuleInfoFile(dir.toString(), jarFile);
            }
        }

        // Strategy 2: Look for a standard module-info.class in the root.
        if (moduleName == null) {
            moduleName = getModuleNameFromModuleInfoFile("module-info.class", jarFile);
        }

        // Strategy 3: Look for an "Automatic-Module-Name" entry in the manifest.
        if (moduleName == null) {
            moduleName = getAutomaticModuleNameFromManifest(jarFile);
        }

        // Strategy 4: Derive the module name from the JAR file's name as a last resort.
        if (moduleName == null) {
            moduleName = deriveModuleNameFromJarFileName(file);
        }

        return moduleName;
    }

    // ... (other private helper methods) ...

    /**
     * @brief Extracts the module name from a `module-info.class` file using an ASM ClassVisitor.
     * @param inputStream The input stream of the `module-info.class` file.
     * @return The module name as a string.
     */
    private String extractModuleNameFromModuleInfo(InputStream inputStream) throws IOException {
        String[] moduleName = new String[1];
        ClassReader cr = new ClassReader(inputStream);
        cr.accept(new ClassVisitor(Opcodes.ASM9) {
            @Override
            public ModuleVisitor visitModule(String name, int access, String version) {
                moduleName[0] = name;
                return super.visitModule(name, access, version);
            }
        }, Opcodes.ASM9);
        return moduleName[0];
    }
}
