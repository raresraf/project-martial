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
 * This Gradle task generates a JSON file that maps classes to their Java modules.
 * This mapping is used during unit tests to simulate the behavior of the Java module system,
 * which is necessary for the entitlements system to look up correct policies when tests are run
 * without full modularity.
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
     * The name of the Java module, if explicitly provided.
     */
    @Input
    @Optional
    public abstract Property<String> getModuleName();

    /**
     * The name of the component to which this build info belongs.
     */
    @Input
    public abstract Property<String> getComponentName();

    /**
     * The collection of code locations (JARs and directories) to be scanned.
     */
    @Classpath
    public abstract Property<FileCollection> getCodeLocations();

    /**
     * The output JSON file that will contain the build information.
     */
    @OutputFile
    public abstract RegularFileProperty getOutputFile();

    /**
     * The main action of the task. It generates the JSON file with the class-to-module mapping.
     * @throws IOException if there is an error writing the output file.
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
     * Represents the structure of the output JSON file.
     * @param component The entitlements component name of the artifact.
     * @param locations A list of code locations within the artifact.
     */
    record OutputFileContents(String component, List<Location> locations) {}

    /**
     * Represents a single code location (a directory or a JAR file) on the classpath.
     * All classes within a location are considered part of the same Java module for entitlements purposes.
     * @param module The name of the Java module for this location.
     * @param representativeClass A representative class from this location.
     */
    record Location(String module, String representativeClass) {}

    /**
     * Builds the list of {@link Location} objects for all code locations.
     * @return A list of {@link Location} objects.
     * @throws IOException if there is an error reading from the code locations.
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
     * @param file The JAR file.
     * @param locations The list to which the new {@link Location} will be added.
     * @throws IOException if there is an error reading the JAR file.
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
     * Finds a representative class name from a JAR file.
     * @param jarFile The JAR file to search.
     * @return An optional containing the class name, or empty if no suitable class is found.
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
     * Extracts the module name from a JAR file using various strategies, mimicking the behavior of the Java module system.
     * @param file The JAR file.
     * @param jarFile The {@link JarFile} object.
     * @return The module name.
     * @throws IOException if there is an error reading the JAR file.
     */
    private String extractModuleNameFromJar(File file, JarFile jarFile) throws IOException {
        String moduleName = null;

        // For multi-release JARs, check for module-info.class in versioned directories.
        if (jarFile.isMultiRelease()) {
            StringBuilder dir = versionDirectoryIfExists(jarFile);
            if (dir != null) {
                dir.append("/module-info.class");
                moduleName = getModuleNameFromModuleInfoFile(dir.toString(), jarFile);
            }
        }

        if (moduleName == null) {
            // Check for module-info.class in the root of the JAR.
            moduleName = getModuleNameFromModuleInfoFile("module-info.class", jarFile);
        }

        if (moduleName == null) {
            // Check for Automatic-Module-Name in the manifest.
            moduleName = getAutomaticModuleNameFromManifest(jarFile);
        }

        if (moduleName == null) {
            // Derive the module name from the JAR file name as a last resort.
            moduleName = deriveModuleNameFromJarFileName(file);
        }

        return moduleName;
    }

    /**
     * For multi-release JARs, finds the latest versioned directory that contains a module-info.class
     * and is compatible with the current JVM version.
     * @param jarFile The JAR file.
     * @return A {@link StringBuilder} with the path to the versioned directory, or null if none is found.
     */
    private static StringBuilder versionDirectoryIfExists(JarFile jarFile) {
        Comparator<Integer> numericOrder = Integer::compareTo;
        List<Integer> versions = jarFile.stream()
            .filter(je -> je.getName().startsWith(META_INF_VERSIONS_PREFIX) && je.getName().endsWith("/module-info.class"))
            .map(
                je -> Integer.parseInt(
                    je.getName().substring(META_INF_VERSIONS_PREFIX.length(), je.getName().indexOf('/', META_INF_VERSIONS_PREFIX.length()))
                )
            )
            .sorted(numericOrder.reversed())
            .toList();
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
     * Extracts the module name from a module-info.class file within a JAR.
     * @param moduleInfoFileName The name of the module-info.class file.
     * @param jarFile The JAR file.
     * @return The module name, or null if not found.
     * @throws IOException if there is an error reading the JAR file.
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
     * Extracts the Automatic-Module-Name from the JAR's manifest.
     * @param jarFile The JAR file.
     * @return The module name, or null if not found.
     * @throws IOException if there is an error reading the manifest.
     */
    private static String getAutomaticModuleNameFromManifest(JarFile jarFile) throws IOException {
        JarEntry manifestEntry = jarFile.getJarEntry("META-INF/MANIFEST.MF");
        if (manifestEntry != null) {
            try (InputStream inputStream = jarFile.getInputStream(manifestEntry)) {
                Manifest manifest = new Manifest(inputStream);
                return manifest.getMainAttributes().getValue("Automatic-Module-Name");
            }
        }
        return null;
    }

    /**
     * Derives a module name from the JAR file name, following the rules of the Java module system.
     * @param jarFile The JAR file.
     * @return The derived module name.
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
     * Extracts module and class information from a directory.
     * @param dir The directory.
     * @param locations The list to which the new {@link Location} will be added.
     * @throws IOException if there is an error reading the directory.
     */
    private void extractLocationsFromDirectory(File dir, List<Location> locations) throws IOException {
        String className = extractClassNameFromDirectory(dir);
        String moduleName = extractModuleNameFromDirectory(dir);

        if (className != null && moduleName != null) {
            locations.add(new Location(moduleName, className));
        }
    }

    /**
     * Finds a representative class name from a directory.
     * @param dir The directory to search.
     * @return The class name, or null if no suitable class is found.
     * @throws IOException if there is an error traversing the directory.
     */
    private String extractClassNameFromDirectory(File dir) throws IOException {
        var visitor = new SimpleFileVisitor<Path>() {
            String result = null;

            @Override
            public @NotNull FileVisitResult visitFile(@NotNull Path candidate, @NotNull BasicFileAttributes attrs) {
                String name = candidate.getFileName().toString();
                if (name.endsWith(".class") && (name.equals("module-info.class") || name.contains("$")) == false) {
                    result = dir.toPath().relativize(candidate).toString();
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
     * Extracts the module name from a directory, looking for module-info.class or using a preset name.
     * @param dir The directory.
     * @return The module name, or null if not found.
     * @throws IOException if there is an error traversing the directory.
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
                        return TERMINATE;
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
     * Extracts the module name from a module-info.class file using ASM.
     * @param inputStream The input stream of the module-info.class file.
     * @return The module name.
     * @throws IOException if there is an error reading the input stream.
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
        }, ClassReader.SKIP_CODE | ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
        return moduleName[0];
    }
}
