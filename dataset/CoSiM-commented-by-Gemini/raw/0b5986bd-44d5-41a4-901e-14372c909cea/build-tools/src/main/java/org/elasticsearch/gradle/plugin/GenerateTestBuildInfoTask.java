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
import java.util.Arrays;
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
 * This Gradle task generates a JSON file containing a mapping of code locations (JARs or directories)
 * to their corresponding Java module names. This information is used during unit tests to simulate
 * Java module behavior, which is necessary for features like entitlements to look up correct policies
 * when running in a non-modular test environment.
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
     * An optional property for the module name, used as a fallback for directories.
     */
    @Input
    @Optional
    public abstract Property<String> getModuleName();

    /**
     * The entitlements component name of the artifact being described.
     */
    @Input
    public abstract Property<String> getComponentName();

    /**
     * The collection of code locations (JARs and directories) to be analyzed.
     */
    @Classpath
    public abstract Property<FileCollection> getCodeLocations();

    /**
     * The output file where the JSON mapping will be written.
     */
    @OutputFile
    public abstract RegularFileProperty getOutputFile();

    /**
     * The main action of the task. It orchestrates the analysis of code locations
     * and writes the resulting mapping to a JSON file.
     * @throws IOException if there is an error writing the file.
     */
    @TaskAction
    public void generatePropertiesFile() throws IOException {
        Path outputFile = getOutputFile().get().getAsFile().toPath();
        Files.createDirectories(outputFile.getParent());

        try (var writer = Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8)) {
            ObjectMapper mapper = new ObjectMapper().configure(SerializationFeature.INDENT_OUTPUT, true)
                .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
            mapper.writeValue(writer, new OutputFileContents(getComponentName().get(), buildLocationList()));
        }
    }

    /**
     * Represents the top-level structure of the JSON output file.
     * @param component The entitlements component name.
     * @param locations A list of code locations associated with this component.
     */
    record OutputFileContents(String component, List<Location> locations) {}

    /**
     * Represents a single code location (a JAR or directory) and its associated Java module.
     * This information allows the test framework to determine what the module would have been
     * in a modular environment, which is crucial for looking up entitlements.
     *
     * @param module              The name of the Java module.
     * @param representativeClass An example class file within this location, used for runtime lookups.
     */
    record Location(String module, String representativeClass) {}

    /**
     * Builds the list of {@link Location}s for all code locations on the classpath.
     * @return A list of Location objects.
     * @throws IOException if an I/O error occurs.
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
     * Extracts the module name and a representative class from a JAR file.
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
     * Scans a JAR file to find the first non-anonymous, non-module-info class.
     * @return An Optional containing the class name, or empty if none is found.
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
     * Determines the module name from a JAR file using a series of fallbacks,
     * mimicking the logic of {@link java.lang.module.ModuleFinder#of}.
     * 1. Check for `module-info.class` in a multi-release version directory.
     * 2. Check for `module-info.class` at the root.
     * 3. Check for `Automatic-Module-Name` in the manifest.
     * 4. Derive the module name from the JAR file name.
     * @return The determined module name.
     */
    private String extractModuleNameFromJar(File file, JarFile jarFile) throws IOException {
        String moduleName = null;

        if (jarFile.isMultiRelease()) {
            StringBuilder dir = versionDirectoryIfExists(jarFile);
            if (dir != null) {
                dir.append("/module-info.class");
                moduleName = getModuleNameFromModuleInfoFile(dir.toString(), jarFile);
            }
        }

        if (moduleName == null) {
            moduleName = getModuleNameFromModuleInfoFile("module-info.class", jarFile);
        }

        if (moduleName == null) {
            moduleName = getAutomaticModuleNameFromManifest(jarFile);
        }

        if (moduleName == null) {
            moduleName = deriveModuleNameFromJarFileName(file);
        }

        return moduleName;
    }

    /**
     * For a multi-release JAR, finds the path to the highest-versioned directory
     * (less than or equal to the current JVM version) that contains a `module-info.class`.
     * @return A StringBuilder with the path (e.g., "META-INF/versions/11"), or null if none is found.
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
            if (version <= major) {
                return path.append(version);
            }
        }
        return null;
    }

    /**
     * Reads a `module-info.class` file from a JAR and extracts the module name.
     * @return The module name, or null if the file doesn't exist.
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
     * Reads the `Automatic-Module-Name` attribute from the JAR's manifest file.
     * @return The automatic module name, or null if not present.
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
     * Derives a module name from the JAR file's name, according to the rules
     * documented in {@link java.lang.module.ModuleFinder#of}.
     */
    private static @NotNull String deriveModuleNameFromJarFileName(File jarFile) {
        String jn = jarFile.getName().substring(0, jarFile.getName().length() - JAR_DESCRIPTOR_SUFFIX.length());
        Matcher matcher = Pattern.compile("-(\\d+(\\.|$))").matcher(jn);
        if (matcher.find()) {
            jn = jn.substring(0, matcher.start());
        }
        jn = jn.replaceAll("[^A-Za-z0-9]", ".");
        return jn;
    }

    /**
     * Extracts the module name and a representative class from a directory on the classpath.
     */
    private void extractLocationsFromDirectory(File dir, List<Location> locations) throws IOException {
        String className = extractClassNameFromDirectory(dir);
        String moduleName = extractModuleNameFromDirectory(dir);

        if (className != null && moduleName != null) {
            locations.add(new Location(moduleName, className));
        }
    }

    /**
     * Finds the first suitable representative class file in a directory.
     * @return The relative path of the class file, or null if not found.
     */
    private String extractClassNameFromDirectory(File dir) throws IOException {
        var visitor = new SimpleFileVisitor<Path>() {
            String result = null;

            @Override
            public @NotNull FileVisitResult visitFile(@NotNull Path candidate, @NotNull BasicFileAttributes attrs) {
                String name = candidate.getFileName().toString(); // Just the part after the last dir separator
                if (name.endsWith(".class") && (name.equals("module-info.class") || name.contains("$")) == false) {
                    result = candidate.toAbsolutePath().toString().substring(dir.getAbsolutePath().length() + 1);
                    return TERMINATE;
                } else {
                    return CONTINUE;
                }
            }
        };
        Files.walkFileTree(dir.toPath(), visitor);
        return visitor.result;
    }

    /**
     * Finds the module name from a directory, either from a `module-info.class` file
     * or by falling back to the module name provided to the task.
     */
    private String extractModuleNameFromDirectory(File dir) throws IOException {
        List<File> files = new ArrayList<>(List.of(dir));
        while (files.isEmpty() == false) {
            File find = files.removeFirst();
            if (find.exists()) {
                if (find.getName().equals("module-info.class")) {
                    try (InputStream inputStream = new FileInputStream(find)) {
                        return extractModuleNameFromModuleInfo(inputStream);
                    }
                } else if (find.isDirectory()) {
                    files.addAll(Arrays.asList(find.listFiles()));
                }
            }
        }
        return getModuleName().getOrNull();
    }

    /**
     * A helper method that uses ASM to parse a `module-info.class` byte stream
     * and extract the module name.
     * @return The module name.
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
