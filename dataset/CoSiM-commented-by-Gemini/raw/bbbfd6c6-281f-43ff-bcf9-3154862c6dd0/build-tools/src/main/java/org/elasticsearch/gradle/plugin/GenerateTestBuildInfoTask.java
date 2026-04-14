/**
 * @file GenerateTestBuildInfoTask.java
 * @brief Defines a Gradle task for generating a test build information file for Elasticsearch plugins.
 *
 * @details This file contains the `GenerateTestBuildInfoTask`, a custom Gradle task used within the
 * Elasticsearch build system. The primary purpose of this task is to scan the test classpath
 * (composed of JAR files and directories) and generate a JSON file. This JSON file maps each
 * code location to its corresponding Java Platform Module System (JPMS) module name and includes
 * a representative class name.
 *
 * Production Systems (Build-time):
 * This task is crucial for enabling module-aware features, like security entitlements, to function
 * correctly in a classpath-based unit testing environment. Since unit tests often run without the
 * full JPMS module path, the application cannot normally determine the module of a given class at
 * runtime. This task pre-computes this mapping at build time. The resulting JSON file is then
 * used by the test runtime to simulate module lookups, allowing for consistent behavior between
 * testing and production environments.
 *
 * Algorithm:
 * The task follows a multi-step process to determine the module name for each classpath entry,
 * mirroring the logic of `java.lang.module.ModuleFinder`:
 * 1.  For multi-release JARs, it checks for `module-info.class` in version-specific directories.
 * 2.  It then checks for a `module-info.class` at the root of the JAR.
 * 3.  If not found, it inspects the JAR's manifest for an `Automatic-Module-Name` entry.
 * 4.  As a final fallback, it derives a module name from the JAR's filename.
 * The ASM library is used to parse the bytecode of `module-info.class` to extract the module name.
 */

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
 * This task generates a file with a class-to-module mapping
 * used to imitate modular behavior during unit tests so
 * entitlements can look up correct policies. This is a cacheable
 * Gradle task, meaning its output can be reused if inputs have not changed.
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
     * An optional property for the module name, typically used for directory-based classpath entries.
     */
    @Input
    @Optional
    public abstract Property<String> getModuleName();

    /**
     * The name of the component, used for entitlement lookups.
     */
    @Input
    public abstract Property<String> getComponentName();

    /**
     * The collection of classpath entries (JARs and directories) to be analyzed.
     */
    @Classpath
    public abstract Property<FileCollection> getCodeLocations();

    /**
     * The output JSON file where the build information will be written.
     */
    @OutputFile
    public abstract RegularFileProperty getOutputFile();

    /**
     * The main action of the Gradle task. It orchestrates the generation of the
     * build info file.
     *
     * @throws IOException if an I/O error occurs writing the file.
     */
    @TaskAction
    public void generatePropertiesFile() throws IOException {
        Path outputFile = getOutputFile().get().getAsFile().toPath();
        Files.createDirectories(outputFile.getParent());

        try (var writer = Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8)) {
            ObjectMapper mapper = new ObjectMapper().configure(SerializationFeature.INDENT_OUTPUT, true)
                .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
            // Build the list of locations and serialize it to JSON.
            mapper.writeValue(writer, new OutputFileContents(getComponentName().get(), buildLocationList()));
        }
    }

    /**
     * Defines the structure of the root JSON object in the output file.
     * @param component the entitlements <em>component</em> name of the artifact we're describing.
     * @param locations a {@link Location} for each code directory/jar in this artifact.
     */
    record OutputFileContents(String component, List<Location> locations) {}

    /**
     * Our analog of a single {@link CodeSource#getLocation()}.
     * All classes in any single <em>location</em> (a directory or jar)
     * are considered to be part of the same Java module for entitlements purposes.
     * Since tests run without Java modules, and entitlements are all predicated on modules,
     * this info lets us determine what the module <em>would have been</em>
     * so we can look up the appropriate entitlements.
     *
     * @param module              the name of the Java module corresponding to this {@code Location}.
     * @param representativeClass an example of any <code>.class</code> file within this {@code Location}
     *                            whose name will be unique within its {@link ClassLoader} at run time.
     */
    record Location(String module, String representativeClass) {}

    /**
     * Build the list of {@link Location}s for all {@link #getCodeLocations() code locations}.
     * It dispatches to different methods based on whether the classpath entry is a JAR or a directory.
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
     * Extracts module and class information from a JAR file.
     */
    private void extractLocationsFromJar(File file, List<Location> locations) throws IOException {
        try (JarFile jarFile = new JarFile(file)) {
            var className = extractClassNameFromJar(jarFile);

            if (className.isPresent()) {
                String moduleName = extractModuleNameFromJar(file, jarFile);
                locations.add(new Location(moduleName, className.get()));
            }
        }
    }

    /**
     * Scans a JAR file to find the first suitable representative class name.
     * It avoids `module-info.class`, files in `META-INF`, and anonymous inner classes.
     */
    private java.util.Optional<String> extractClassNameFromJar(JarFile jarFile) {
        return jarFile.stream()
            .filter(
                je -> je.getName().startsWith("META-INF") == false
                    && je.getName().equals("module-info.class") == false
                    && je.getName().contains("$") == false
                    && je.getName().endsWith(".class")
            )
            .findFirst()
            .map(ZipEntry::getName);
    }

    /**
     * Determines the module name for a JAR file using a succession of techniques,
     * mimicking {@link java.lang.module.ModuleFinder#of}.
     */
    private String extractModuleNameFromJar(File file, JarFile jarFile) throws IOException {
        String moduleName = null;

        // 1. Check for module-info in versioned, multi-release directories.
        if (jarFile.isMultiRelease()) {
            StringBuilder dir = versionDirectoryIfExists(jarFile);
            if (dir != null) {
                dir.append("/module-info.class");
                moduleName = getModuleNameFromModuleInfoFile(dir.toString(), jarFile);
            }
        }

        // 2. Check for module-info at the root.
        if (moduleName == null) {
            moduleName = getModuleNameFromModuleInfoFile("module-info.class", jarFile);
        }

        // 3. Check for Automatic-Module-Name in the manifest.
        if (moduleName == null) {
            moduleName = getAutomaticModuleNameFromManifest(jarFile);
        }

        // 4. Derive the module name from the JAR filename as a last resort.
        if (moduleName == null) {
            moduleName = deriveModuleNameFromJarFileName(file);
        }

        return moduleName;
    }

    /**
     * For a multi-release JAR, finds the latest version-specific directory
     * (e.g., `META-INF/versions/11`) that is compatible with the current JVM version.
     *
     * @return a {@link StringBuilder} with the path (e.g., {@code META-INF/versions/<version>}) or null if none is found.
     */
    private static StringBuilder versionDirectoryIfExists(JarFile jarFile) {
        List<Integer> versions = jarFile.stream()
            .filter(je -> je.getName().startsWith(META_INF_VERSIONS_PREFIX) && je.getName().endsWith("/module-info.class"))
            .map(
                je -> Integer.parseInt(
                    je.getName().substring(META_INF_VERSIONS_PREFIX.length(), je.getName().length() - META_INF_VERSIONS_PREFIX.length())
                )
            )
            .toList();
        versions = new ArrayList<>(versions);
        versions.sort(Integer::compareTo);
        versions = versions.reversed();
        int major = Runtime.version().feature();
        StringBuilder path = new StringBuilder(META_INF_VERSIONS_PREFIX);
        for (int version : versions) {
            // Pre-condition: Check if the directory's Java version is usable by the current JVM.
            if (version <= major) {
                return path.append(version);
            }
        }
        return null;
    }

    /**
     * Reads a `module-info.class` file from within a JAR and extracts the module name.
     * @return the module name, or null if the file does not exist.
     */
    private String getModuleNameFromModuleInfoFile(String moduleInfoFileName, JarFile jarFile) throws IOException {
        JarEntry moduleEntry = jarFile.getJarEntry(moduleInfoFileName);
        if (moduleEntry != null) {
            try (InputStream inputStream = jarFile.getInputStream(moduleEntry)) {
                return extractModuleNameFromModuleInfo(inputStream);
            }
        }
        return null;
    }

    /**
     * Reads the `MANIFEST.MF` file from a JAR and returns the `Automatic-Module-Name` value if present.
     * @return the module name, or null if not found.
     */
    private static String getAutomaticModuleNameFromManifest(JarFile jarFile) throws IOException {
        JarEntry manifestEntry = jarFile.getJarEntry("META-INF/MANIFEST.MF");
        if (manifestEntry != null) {
            try (InputStream inputStream = jarFile.getInputStream(manifestEntry)) {
                Manifest manifest = new Manifest(inputStream);
                String amn = manifest.getMainAttributes().getValue("Automatic-Module-Name");
                if (amn != null) {
                    return amn;
                }
            }
        }
        return null;
    }

    /**
     * Derives a module name from a JAR filename, as a fallback.
     * This logic strips version numbers and replaces non-alphanumeric characters with dots.
     */
    private static @NotNull String deriveModuleNameFromJarFileName(File jarFile) {
        String jn = jarFile.getName().substring(0, jarFile.getName().length() - JAR_DESCRIPTOR_SUFFIX.length());
        Matcher matcher = Pattern.compile("-(\d+(\.|$))").matcher(jn);
        if (matcher.find()) {
            jn = jn.substring(0, matcher.start());
        }
        jn = jn.replaceAll("[^A-Za-z0-9]", ".");
        return jn;
    }

    /**
     * Extracts module and class information from a directory classpath entry.
     */
    private void extractLocationsFromDirectory(File dir, List<Location> locations) throws IOException {
        String className = extractClassNameFromDirectory(dir);
        String moduleName = extractModuleNameFromDirectory(dir);

        if (className != null && moduleName != null) {
            locations.add(new Location(moduleName, className));
        }
    }

    /**
     * Walks a directory to find the first suitable representative class file.
     */
    private String extractClassNameFromDirectory(File dir) throws IOException {
        var visitor = new SimpleFileVisitor<Path>() {
            String result = null;

            @Override
            public @NotNull FileVisitResult visitFile(@NotNull Path candidate, @NotNull BasicFileAttributes attrs) {
                String name = candidate.getFileName().toString();
                // Find a class that is not module-info and not an anonymous inner class.
                if (name.endsWith(".class") && (name.equals("module-info.class") || name.contains("$")) == false) {
                    result = candidate.toAbsolutePath().toString().substring(dir.getAbsolutePath().length() + 1);
                    return TERMINATE; // Stop after finding the first one.
                } else {
                    return CONTINUE;
                }
            }
        };
        Files.walkFileTree(dir.toPath(), visitor);
        return visitor.result;
    }

    /**
     * Walks a directory to find `module-info.class` and extract the module name.
     * If not found, it falls back to the module name provided to the task, if any.
     */
    private String extractModuleNameFromDirectory(File dir) throws IOException {
        var visitor = new SimpleFileVisitor<Path>() {
            private String result = getModuleName().getOrNull();

            @Override
            public @NotNull FileVisitResult visitFile(@NotNull Path candidate, @NotNull BasicFileAttributes attrs) throws IOException {
                String name = candidate.getFileName().toString();
                if (name.equals("module-info.class")) {
                    try (InputStream inputStream = new FileInputStream(candidate.toFile())) {
                        result = extractModuleNameFromModuleInfo(inputStream);
                        return TERMINATE; // Stop after finding module-info.
                    }
                } else {
                    return CONTINUE;
                }
            }
        };
        Files.walkFileTree(dir.toPath(), visitor);
        return visitor.result;
    }

    /**
     * Uses the ASM bytecode manipulation library to read a `module-info.class`
     * file's bytecode and extract the declared module name.
     * @param inputStream An input stream to the `module-info.class` data.
     * @return The declared module name.
     * @throws IOException if the stream cannot be read.
     */
    private String extractModuleNameFromModuleInfo(InputStream inputStream) throws IOException {
        String[] moduleName = new String[1];
        ClassReader cr = new ClassReader(inputStream);
        // Use a ClassVisitor to hook into the bytecode parsing process.
        cr.accept(new ClassVisitor(Opcodes.ASM9) {
            @Override
            public ModuleVisitor visitModule(String name, int access, String version) {
                // The 'name' parameter of visitModule is the declared module name.
                moduleName[0] = name;
                return super.visitModule(name, access, version);
            }
        }, Opcodes.ASM9);
        return moduleName[0];
    }
}
