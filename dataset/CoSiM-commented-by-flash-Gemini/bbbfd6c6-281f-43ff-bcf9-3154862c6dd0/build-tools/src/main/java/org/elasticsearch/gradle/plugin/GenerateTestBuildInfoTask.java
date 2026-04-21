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
 * @file GenerateTestBuildInfoTask.java
 * @brief Gradle task for generating class-to-module mapping metadata for unit testing.
 * 
 * Functional Intent: Facilitates the simulation of Java Platform Module System (JPMS) 
 * behavior in non-modular test environments. It scans classpath entries (JARs or 
 * directories), identifies their declared or automatic module names, and selects 
 * a representative class for each. This mapping is used by security managers and 
 * entitlement systems to look up correct policies based on the effective module 
 * context at runtime.
 * 
 * Domain: Production Systems, Build Engineering, Java Modules (JPMS), Security.
 */
@CacheableTask
public abstract class GenerateTestBuildInfoTask extends DefaultTask {

    public static final String DESCRIPTION = "generates plugin test dependencies file";

    public static final String META_INF_VERSIONS_PREFIX = "META-INF/versions/";
    public static final String JAR_DESCRIPTOR_SUFFIX = ".jar";

    public GenerateTestBuildInfoTask() {
        setDescription(DESCRIPTION);
    }

    @Input
    @Optional
    public abstract Property<String> getModuleName();

    @Input
    public abstract Property<String> getComponentName();

    @Classpath
    public abstract Property<FileCollection> getCodeLocations();

    @OutputFile
    public abstract RegularFileProperty getOutputFile();

    /**
     * @brief Entry point for the Gradle task action.
     * 
     * Logic: Orchestrates the location-discovery process and serializes results 
     * to a JSON file using Snake Case naming conventions.
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
     * @record OutputFileContents
     * @brief Root container for the generated build info JSON.
     */
    record OutputFileContents(String component, List<Location> locations) {}

    /**
     * @record Location
     * @brief Captures the module identity for a specific classpath resource.
     */
    record Location(String module, String representativeClass) {}

    /**
     * buildLocationList - Iterates through provided code locations and resolves modules.
     * 
     * Algorithm: Type-driven extraction strategy.
     * Logic: Differentiates between JAR files and flat directories to apply 
     * specialized discovery logic for each entry type.
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
     * extractLocationsFromJar - Processes a JAR file to find its module name and a sample class.
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
     * extractClassNameFromJar - Identifies the first non-infrastructure class in the JAR.
     * 
     * Logic: Filters out META-INF metadata, module descriptors, and anonymous classes 
     * to find a unique, named representative class.
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
     * extractModuleNameFromJar - Multi-tiered module identification strategy.
     * 
     * Algorithm: Tiered lookup corresponding to JDK ModuleFinder specification.
     * 1. Check Multi-Release versioned module-info.
     * 2. Check root module-info.class.
     * 3. Check Automatic-Module-Name in Manifest.
     * 4. Fall back to filename-derived name.
     */
    private String extractModuleNameFromJar(File file, JarFile jarFile) throws IOException {
        String moduleName = null;

        // Block Logic: Multi-release JAR handling.
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
            // Logic: Ultimate fallback using sanitized filename.
            moduleName = deriveModuleNameFromJarFileName(file);
        }

        return moduleName;
    }

    /**
     * versionDirectoryIfExists - Finds the best-matching versioned module info in a MR-JAR.
     */
    private static StringBuilder versionDirectoryIfExists(JarFile jarFile) {
        List<Integer> versions = jarFile.stream()
            .filter(je -> je.getName().startsWith(META_INF_VERSIONS_PREFIX) && je.getName().endsWith("/module-info.class"))
            .map(
                je -> Integer.parseInt(
                    je.getName().substring(META_INF_VERSIONS_PREFIX.length(), je.getName().length() - "/module-info.class".length() - 1)
                )
            )
            .toList();
        versions = new ArrayList<>(versions);
        versions.sort(Integer::compareTo);
        versions = versions.reversed();
        int major = Runtime.version().feature();
        StringBuilder path = new StringBuilder(META_INF_VERSIONS_PREFIX);
        /**
         * Block Logic: Version selection.
         * Invariant: Selects the highest version descriptor that is compatible with the current JVM.
         */
        for (int version : versions) {
            if (version <= major) {
                return path.append(version);
            }
        }
        return null;
    }

    private String getModuleNameFromModuleInfoFile(String moduleInfoFileName, JarFile jarFile) throws IOException {
        JarEntry moduleEntry = jarFile.getJarEntry(moduleInfoFileName);
        if (moduleEntry != null) {
            try (InputStream inputStream = jarFile.getInputStream(moduleEntry)) {
                return extractModuleNameFromModuleInfo(inputStream);
            }
        }
        return null;
    }

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
     * deriveModuleNameFromJarFileName - Heuristic-based module naming from files.
     * 
     * Logic: Strips version numbers and sanitizes special characters into dots 
     * to approximate a valid Java package name.
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
     * extractLocationsFromDirectory - Processes a filesystem directory to find module context.
     */
    private void extractLocationsFromDirectory(File dir, List<Location> locations) throws IOException {
        String className = extractClassNameFromDirectory(dir);
        String moduleName = extractModuleNameFromDirectory(dir);

        if (className != null && moduleName != null) {
            locations.add(new Location(moduleName, className));
        }
    }

    /**
     * extractClassNameFromDirectory - Recursive search for a valid .class file.
     */
    private String extractClassNameFromDirectory(File dir) throws IOException {
        var visitor = new SimpleFileVisitor<Path>() {
            String result = null;

            @Override
            public @NotNull FileVisitResult visitFile(@NotNull Path candidate, @NotNull BasicFileAttributes attrs) {
                String name = candidate.getFileName().toString(); // Just the part after the last dir separator
                if (name.endsWith(".class") && (name.equals("module-info.class") || name.contains("$")) == false) {
                    // Logic: Computes the relative path (class name equivalent) for the selected representative.
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

    private String extractModuleNameFromDirectory(File dir) throws IOException {
        var visitor = new SimpleFileVisitor<Path>() {
            private String result = getModuleName().getOrNull();

            @Override
            public @NotNull FileVisitResult visitFile(@NotNull Path candidate, @NotNull BasicFileAttributes attrs) throws IOException {
                String name = candidate.getFileName().toString(); // Just the part after the last dir separator
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
     * extractModuleNameFromModuleInfo - Expert byte-code parsing via ASM.
     * 
     * Functional Intent: Directly inspects the Java class file format to extract 
     * the module name from the module descriptor attributes.
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
