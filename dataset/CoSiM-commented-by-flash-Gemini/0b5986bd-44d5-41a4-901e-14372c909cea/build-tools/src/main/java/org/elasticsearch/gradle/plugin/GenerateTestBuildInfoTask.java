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
 * @file GenerateTestBuildInfoTask.java
 * @brief Gradle task for generating test build information, specifically class-to-module mappings.
 *
 * This task is crucial for testing environments where Java's modular behavior needs to be
 * accurately imitated for entitlements lookups. It analyzes the project's classpath,
 * identifies classes and their corresponding module names (even for non-modular JARs
 * or directories), and outputs this mapping into a structured JSON file. This allows
 * unit tests to correctly resolve policies that depend on module information,
 * bypassing the complexities of actual Java modules during testing.
 */
/**
 * @class GenerateTestBuildInfoTask
 * @brief A Gradle task that generates a JSON file containing class-to-module mapping information.
 *
 * Functional Utility: This task's primary role is to bridge the gap between
 * modular entitlements and non-modular test execution environments. It ensures
 * that security entitlements, which are often module-aware, can still be correctly
 * applied and validated during unit tests by providing a simulated module context.
 * It's {@link CacheableTask} to optimize build times by only executing when inputs change.
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
     * @property getModuleName
     * @brief Represents the explicit module name to use for this artifact.
     * Functional Utility: Allows overriding the automatically derived module name.
     * @return A Gradle Property that might contain the module name as a String.
     */
    @Input
    @Optional
    public abstract Property<String> getModuleName();

    /**
     * @property getComponentName
     * @brief Represents the entitlements component name of the artifact.
     * Functional Utility: Used in the generated output to identify the component
     * for which the class-to-module mapping is being created.
     * @return A Gradle Property containing the component name as a String.
     */
    @Input
    public abstract Property<String> getComponentName();

    /**
     * @property getCodeLocations
     * @brief Represents the collection of code locations (JARs or directories) for this artifact.
     * Functional Utility: Defines the input classpath that this task will analyze
     * to extract class and module information.
     * @return A Gradle Property containing a FileCollection of code locations.
     */
    @Classpath
    public abstract Property<FileCollection> getCodeLocations();

    /**
     * @property getOutputFile
     * @brief Represents the output file where the generated JSON mapping will be written.
     * Functional Utility: Specifies the destination for the serialized class-to-module
     * information.
     * @return A Gradle RegularFileProperty pointing to the output file.
     */
    @OutputFile
    public abstract RegularFileProperty getOutputFile();

    @TaskAction
    public void generatePropertiesFile() throws IOException {
        Path outputFile = getOutputFile().get().getAsFile().toPath();
        Files.createDirectories(outputFile.getParent());

        /**
         * Block Logic: Writes the generated class-to-module mapping to the output file in JSON format.
         * Functional Utility: Uses Jackson ObjectMapper for structured serialization, ensuring
         * the output is readable and consistently formatted (pretty-printed with snake_case).
         * Pre-condition: `outputFile` has been created and its parent directories exist.
         * Invariant: The output file will contain a JSON representation of `OutputFileContents`.
         */
        try (var writer = Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8)) {
            ObjectMapper mapper = new ObjectMapper().configure(SerializationFeature.INDENT_OUTPUT, true)
                .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
            mapper.writeValue(writer, new OutputFileContents(getComponentName().get(), buildLocationList()));
        }
    }

    /**
     * @record OutputFileContents
     * @brief Represents the structure of the JSON output file generated by this task.
     * @param component The entitlements component name of the artifact being described.
     * @param locations A list of {@link Location} objects, each describing a code location
     *                  (directory or JAR) within this artifact.
     * Functional Utility: Provides a clear, type-safe schema for the serialized output,
     * ensuring that the generated build information is well-defined and consumable.
     */
    record OutputFileContents(String component, List<Location> locations) {}

    /**
     * @record Location
     * @brief Represents a single code source location (a directory or JAR) and its derived module information.
     * Functional Utility: Acts as an analog to {@link CodeSource#getLocation()}, providing
     * crucial details for entitlements lookups in a non-modular test environment.
     * All classes within a single {@code Location} are considered part of the same Java module
     * for entitlements.
     *
     * @param module The name of the Java module corresponding to this {@code Location}.
     * @param representativeClass An example of any non-anonymous <code>.class</code> file within
     *                            this {@code Location} whose name is expected to be unique within
     *                            its {@link ClassLoader} at runtime.
     */
    record Location(String module, String representativeClass) {}

    /**
     * @method buildLocationList
     * @brief Builds the list of {@link Location} objects for all code locations on the classpath.
     * Functional Utility: Iterates through each file on the configured classpath,
     * delegating to specific extraction methods based on whether the entry is a JAR or a directory.
     * @return A List of {@link Location} objects, each representing a processed code location.
     * @throws IOException If an I/O error occurs during file processing.
     * @throws IllegalArgumentException If an unrecognized classpath entry type is encountered.
     * Pre-condition: `getCodeLocations()` must return a valid FileCollection.
     */
    private List<Location> buildLocationList() throws IOException {
        List<Location> locations = new ArrayList<>();
        /**
         * Block Logic: Iterates over each file in the resolved code locations.
         * Invariant: Each file is processed based on its type (JAR or directory) to extract module information.
         */
        for (File file : getCodeLocations().get().getFiles()) {
            if (file.exists()) {
                if (file.getName().endsWith(JAR_DESCRIPTOR_SUFFIX)) {
                    // Functional Utility: Delegates to method for extracting information from JAR files.
                    extractLocationsFromJar(file, locations);
                } else if (file.isDirectory()) {
                    // Functional Utility: Delegates to method for extracting information from directory.
                    extractLocationsFromDirectory(file, locations);
                } else {
                    // Functional Utility: Throws an exception for unsupported classpath entry types.
                    throw new IllegalArgumentException("unrecognized classpath entry: " + file);
                }
            }
        }
        // Functional Utility: Returns an immutable copy of the collected locations.
        return List.copyOf(locations);
    }

    /**
     * @method extractLocationsFromJar
     * @brief Extracts class and module information from a given JAR file.
     * Functional Utility: Opens the JAR, finds a representative class name, and determines
     * the module name using various heuristics, then adds this as a {@link Location} to the list.
     * @param file The JAR file to analyze.
     * @param locations The list to which the extracted {@link Location} will be added.
     * @throws IOException If an I/O error occurs while reading the JAR file.
     * Pre-condition: `file` must be an existing JAR file.
     * Invariant: If a representative class is found, a new `Location` entry is added to `locations`.
     */
    private void extractLocationsFromJar(File file, List<Location> locations) throws IOException {
        try (JarFile jarFile = new JarFile(file)) {
            // Functional Utility: Attempts to find a suitable representative class within the JAR.
            var className = extractClassNameFromJar(jarFile);

            // Block Logic: If a class name is successfully extracted, determine the module name and add the location.
            if (className.isPresent()) {
                String moduleName = extractModuleNameFromJar(file, jarFile);
                locations.add(new Location(moduleName, className.get()));
            }
        }
    }

    /**
     * @method extractClassNameFromJar
     * @brief Locates the first suitable representative class name within a JAR file.
     * Functional Utility: Scans the JAR entries to find a `.class` file that is not
     * a `module-info.class`, not in `META-INF`, and not an anonymous class. This
     * class name serves as a unique identifier for the location within its
     * {@link ClassLoader} at runtime.
     * @param jarFile The {@link JarFile} to search.
     * @return An {@link java.util.Optional} containing the class name as a String if found,
     *         otherwise an empty Optional.
     * Pre-condition: `jarFile` is a valid, open JAR file.
     * Invariant: Returned class name, if present, is considered unique for the JAR's context.
     */
    private java.util.Optional<String> extractClassNameFromJar(JarFile jarFile) {
        return jarFile.stream()
            /**
             * Block Logic: Filters JAR entries to find a suitable class file.
             * It excludes:
             * - Files under "META-INF" (metadata, not unique classes).
             * - "module-info.class" (system-defined, not unique to a specific module).
             * - Anonymous classes (identified by '$' in their name), which are dynamic and not stable representatives.
             * - Non-class files.
             * Functional Utility: Ensures that the selected class is a stable and unique identifier for the JAR's contents.
             */
            .filter(
                je -> je.getName().startsWith("META-INF") == false
                    && je.getName().equals("module-info.class") == false
                    && je.getName().contains("$") == false
                    && je.getName().endsWith(".class")
            )
            // Functional Utility: Finds the first class that matches the criteria.
            .findFirst()
            // Functional Utility: Maps the ZipEntry to its name (which is the class path).
            .map(ZipEntry::getName);
    }

    /**
     * @method extractModuleNameFromJar
     * @brief Determines the module name for a given JAR file using a series of fallback mechanisms.
     * Functional Utility: Emulates the JDK's module name resolution process, attempting to find
     * an explicit module name from `module-info.class` (considering multi-release JARs),
     * then from the `Automatic-Module-Name` manifest attribute, and finally deriving it
     * from the JAR's file name.
     * @param file The {@link File} object representing the JAR.
     * @param jarFile The {@link JarFile} object for the JAR.
     * @return The determined module name as a String, or null if it cannot be determined.
     * @throws IOException If an I/O error occurs while reading the JAR file or manifest.
     * Pre-condition: `file` and `jarFile` represent an existing JAR.
     * Invariant: The module name is determined based on JDK's specified heuristics.
     */
    private String extractModuleNameFromJar(File file, JarFile jarFile) throws IOException {
        String moduleName = null;

        /**
         * Block Logic: Attempts to extract the module name from a multi-release JAR's `module-info.class`.
         * Functional Utility: Checks if the JAR is multi-release and if a versioned `module-info.class`
         * exists that matches the current JVM's major version.
         */
        if (jarFile.isMultiRelease()) {
            StringBuilder dir = versionDirectoryIfExists(jarFile);
            if (dir != null) {
                dir.append("/module-info.class");
                moduleName = getModuleNameFromModuleInfoFile(dir.toString(), jarFile);
            }
        }

        /**
         * Block Logic: If not found in multi-release `module-info.class`, checks the root `module-info.class`.
         */
        if (moduleName == null) {
            moduleName = getModuleNameFromModuleInfoFile("module-info.class", jarFile);
        }

        /**
         * Block Logic: If no `module-info.class` is found, checks for `Automatic-Module-Name` in `MANIFEST.MF`.
         */
        if (moduleName == null) {
            moduleName = getAutomaticModuleNameFromManifest(jarFile);
        }

        /**
         * Block Logic: As a last resort, if no explicit module name is found, derives it from the JAR file name.
         */
        if (moduleName == null) {
            moduleName = deriveModuleNameFromJarFileName(file);
        }

        return moduleName;
    }

    /**
     * @method versionDirectoryIfExists
     * @brief Determines the path to the appropriate versioned `META-INF/versions/` directory in a multi-release JAR.
     * Functional Utility: Scans a multi-release JAR for `module-info.class` files within versioned directories.
     * It identifies the highest version directory that is less than or equal to the current JVM's major version.
     * This follows the JDK's rule for resolving module-info in multi-release JARs.
     * @param jarFile The {@link JarFile} to inspect.
     * @return A {@link StringBuilder} containing the path "META-INF/versions/<version number>" if a suitable
     *         versioned directory is found, otherwise null.
     * Pre-condition: `jarFile` is a multi-release JAR.
     * Invariant: The returned path, if non-null, points to the relevant `META-INF/versions/` directory for the current JVM.
     */
    private static StringBuilder versionDirectoryIfExists(JarFile jarFile) {
        // Block Logic: Filter JAR entries to find module-info.class files within versioned META-INF directories.
        List<Integer> versions = jarFile.stream()
            .filter(je -> je.getName().startsWith(META_INF_VERSIONS_PREFIX) && je.getName().endsWith("/module-info.class"))
            // Functional Utility: Extract the version number from the path.
            .map(
                je -> Integer.parseInt(
                    je.getName().substring(META_INF_VERSIONS_PREFIX.length(), je.getName().length() - META_INF_VERSIONS_PREFIX.length())
                )
            )
            .toList();
        versions = new ArrayList<>(versions);
        // Functional Utility: Sort versions to process from lowest to highest.
        versions.sort(Integer::compareTo);
        // Functional Utility: Reverse the sorted list to iterate from highest to lowest version.
        versions = versions.reversed();
        // Functional Utility: Get the current JVM's major version.
        int major = Runtime.version().feature();
        // Initialize path for potential versioned directory.
        StringBuilder path = new StringBuilder(META_INF_VERSIONS_PREFIX);
        /**
         * Block Logic: Iterate through sorted versions to find the highest version less than or equal to the current JVM major version.
         * Invariant: The loop finds the most compatible versioned directory according to JVM rules.
         */
        for (int version : versions) {
            if (version <= major) {
                return path.append(version);
            }
        }
        return null;
    }

    /**
     * @method getModuleNameFromModuleInfoFile
     * @brief Extracts the module name from a `module-info.class` file within a JAR.
     * Functional Utility: Locates the specified `module-info.class` entry in the JAR,
     * reads its content, and uses ASM to parse and extract the declared module name.
     * @param moduleInfoFileName The name of the `module-info.class` file (e.g., "module-info.class" or "META-INF/versions/X/module-info.class").
     * @param jarFile The {@link JarFile} containing the module-info.
     * @return The declared module name as a String, or null if the entry does not exist or parsing fails.
     * @throws IOException If an I/O error occurs while reading the `module-info.class` file.
     * Pre-condition: `jarFile` is a valid, open JAR file.
     */
    private String getModuleNameFromModuleInfoFile(String moduleInfoFileName, JarFile jarFile) throws IOException {
        // Functional Utility: Retrieve the JAR entry for the module-info file.
        JarEntry moduleEntry = jarFile.getJarEntry(moduleInfoFileName);
        // Block Logic: If the module-info entry exists, process its input stream to extract the module name.
        if (moduleEntry != null) {
            try (InputStream inputStream = jarFile.getInputStream(moduleEntry)) {
                return extractModuleNameFromModuleInfo(inputStream);
            }
        }
        return null;
    }

    /**
     * @method getAutomaticModuleNameFromManifest
     * @brief Extracts the `Automatic-Module-Name` from a JAR's `MANIFEST.MF` file.
     * Functional Utility: Reads the JAR's manifest and retrieves the value of the
     * `Automatic-Module-Name` attribute, which explicitly defines an automatic
     * module name for non-modular JARs.
     * @param jarFile The {@link JarFile} to inspect.
     * @return The automatic module name as a String, or null if the manifest is
     *         non-existent or the attribute is not present.
     * @throws IOException If an I/O error occurs while reading the manifest.
     * Pre-condition: `jarFile` is a valid, open JAR file.
     */
    private static String getAutomaticModuleNameFromManifest(JarFile jarFile) throws IOException {
        // Functional Utility: Retrieve the JAR entry for the manifest file.
        JarEntry manifestEntry = jarFile.getJarEntry("META-INF/MANIFEST.MF");
        // Block Logic: If the manifest entry exists, read its contents and extract the automatic module name.
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
     * @method deriveModuleNameFromJarFileName
     * @brief Derives a module name from the JAR file's name, following JDK conventions.
     * Functional Utility: This is a fallback mechanism to determine a module name
     * when explicit module-info or manifest attributes are absent. It processes
     * the JAR file name by removing version suffixes and replacing invalid characters,
     * mimicking {@link java.lang.module.ModuleFinder#of}'s behavior.
     * @param jarFile The {@link File} object representing the JAR.
     * @return A non-null String representing the derived module name.
     * Pre-condition: `jarFile` is a valid JAR file.
     * Invariant: The returned name adheres to typical module naming conventions,
     * derived algorithmically from the file name.
     */
    private static @NotNull String deriveModuleNameFromJarFileName(File jarFile) {
        // Functional Utility: Extracts the base name of the JAR file by removing the ".jar" suffix.
        String jn = jarFile.getName().substring(0, jarFile.getName().length() - JAR_DESCRIPTOR_SUFFIX.length());
        // Functional Utility: Uses a regex to find and remove version strings (e.g., "-1.0", "-2.0-SNAPSHOT").
        Matcher matcher = Pattern.compile("-(\\d+(\\.|$))").matcher(jn);
        /**
         * Block Logic: If a version pattern is found, truncate the file name to exclude the version.
         * Invariant: `jn` is updated to contain only the artifact name without versioning if a match is found.
         */
        if (matcher.find()) {
            jn = jn.substring(0, matcher.start());
        }
        // Functional Utility: Replaces non-alphanumeric characters with dots, normalizing the name for module compatibility.
        jn = jn.replaceAll("[^A-Za-z0-9]", ".");
        return jn;
    }

    /**
     * @method extractLocationsFromDirectory
     * @brief Extracts class and module information from a given directory.
     * Functional Utility: Scans the directory to find a representative class name
     * and determines the module name, then adds this as a {@link Location} to the list.
     * @param dir The directory to analyze.
     * @param locations The list to which the extracted {@link Location} will be added.
     * @throws IOException If an I/O error occurs during directory traversal or file reading.
     * Pre-condition: `dir` must be an existing directory.
     * Invariant: If both a representative class and a module name are found, a new `Location`
     *            entry is added to `locations`.
     */
    private void extractLocationsFromDirectory(File dir, List<Location> locations) throws IOException {
        // Functional Utility: Find a representative class name within the directory.
        String className = extractClassNameFromDirectory(dir);
        // Functional Utility: Determine the module name associated with the directory.
        String moduleName = extractModuleNameFromDirectory(dir);

        // Block Logic: If both a class name and module name are successfully extracted, add them as a new Location.
        if (className != null && moduleName != null) {
            locations.add(new Location(moduleName, className));
        }
    }

    /**
     * @method extractClassNameFromDirectory
     * @brief Finds the first suitable representative class name within a directory.
     * Functional Utility: Traverses the directory structure to locate a `.class` file
     * that is not `module-info.class` and not an anonymous class. This class name
     * serves as a unique identifier for the location within its {@link ClassLoader} at runtime.
     * @param dir The directory to search.
     * @return The path of the representative class relative to the `dir` as a String,
     *         or null if no suitable class is found.
     * @throws IOException If an I/O error occurs during file system traversal.
     * Pre-condition: `dir` is an existing directory.
     * Invariant: The returned class path, if present, is a stable and unique identifier for the directory's contents.
     */
    private String extractClassNameFromDirectory(File dir) throws IOException {
        var visitor = new SimpleFileVisitor<Path>() {
            String result = null;

            @Override
            public @NotNull FileVisitResult visitFile(@NotNull Path candidate, @NotNull BasicFileAttributes attrs) {
                String name = candidate.getFileName().toString(); // Just the part after the last dir separator
                /**
                 * Block Logic: Filters for class files that are not module-info.class or anonymous classes.
                 * Functional Utility: Ensures the selected class is a stable representative.
                 * Invariant: `result` is set upon finding the first valid class file, and traversal terminates.
                 */
                if (name.endsWith(".class") && (name.equals("module-info.class") || name.contains("$")) == false) {
                    // Functional Utility: Stores the relative path of the class file.
                    result = candidate.toAbsolutePath().toString().substring(dir.getAbsolutePath().length() + 1);
                    return TERMINATE; // Functional Utility: Terminates traversal after finding the first suitable class.
                } else {
                    return CONTINUE; // Functional Utility: Continues traversal if the current file is not suitable.
                }
            }
        };
        // Functional Utility: Initiates file system traversal to find the class.
        Files.walkFileTree(dir.toPath(), visitor);
        return visitor.result;
    }

    /**
     * @method extractModuleNameFromDirectory
     * @brief Determines the module name for a given directory.
     * Functional Utility: Prioritizes extracting the module name from a `module-info.class`
     * file found within the directory structure. If no `module-info.class` is present,
     * it falls back to a predefined module name (if available via `getModuleName()`).
     * @param dir The directory to search.
     * @return The determined module name as a String, or null if it cannot be found.
     * @throws IOException If an I/O error occurs during directory traversal or file reading.
     * Pre-condition: `dir` is an existing directory.
     * Invariant: The method attempts to derive a module name following Java module system heuristics.
     */
    private String extractModuleNameFromDirectory(File dir) throws IOException {
        List<File> files = new ArrayList<>(List.of(dir));
        /**
         * Block Logic: Traverses the directory tree to find a `module-info.class` file.
         * Pre-condition: `files` contains directories and files to be explored.
         * Invariant: The loop continues until all files in the current `files` list are processed or `module-info.class` is found.
         */
        while (files.isEmpty() == false) {
            File find = files.removeFirst();
            if (find.exists()) {
                /**
                 * Block Logic: If `module-info.class` is found, extract and return its module name.
                 * Functional Utility: Uses ASM to parse the bytecode of `module-info.class`.
                 */
                if (find.getName().equals("module-info.class")) {
                    try (InputStream inputStream = new FileInputStream(find)) {
                        return extractModuleNameFromModuleInfo(inputStream);
                    }
                } else if (find.isDirectory()) {
                    // Functional Utility: Adds all files and subdirectories to the list for further traversal.
                    files.addAll(Arrays.asList(find.listFiles()));
                }
            }
        }
        // Functional Utility: Fallback to the module name provided by the Gradle task configuration.
        return getModuleName().getOrNull();
    }

    /**
     * @method extractModuleNameFromModuleInfo
     * @brief Extracts the module name from the bytecode of a `module-info.class` file.
     * Functional Utility: Uses the ASM library to parse the `module-info.class` bytecode
     * and retrieve the module's declared name. This is the most authoritative way to get
     * the module name for an explicit module.
     * @param inputStream An {@link InputStream} providing the bytecode of `module-info.class`.
     * @return The declared module name as a String.
     * @throws IOException If an I/O error occurs while reading the input stream.
     * Pre-condition: `inputStream` contains valid bytecode for a `module-info.class` file.
     */
    private String extractModuleNameFromModuleInfo(InputStream inputStream) throws IOException {
        String[] moduleName = new String[1];
        // Functional Utility: Create a ClassReader to parse the bytecode.
        ClassReader cr = new ClassReader(inputStream);
        /**
         * Block Logic: Use an anonymous ClassVisitor to capture the module name during bytecode parsing.
         * Functional Utility: The `visitModule` method of the `ModuleVisitor` is invoked when the module declaration is encountered.
         * Invariant: `moduleName[0]` will hold the module name after `accept` completes.
         */
        cr.accept(new ClassVisitor(Opcodes.ASM9) {
            @Override
            public ModuleVisitor visitModule(String name, int access, String version) {
                // Functional Utility: Store the module name found in the bytecode.
                moduleName[0] = name;
                return super.visitModule(name, access, version);
            }
        }, Opcodes.ASM9);
        return moduleName[0];
    }
}
