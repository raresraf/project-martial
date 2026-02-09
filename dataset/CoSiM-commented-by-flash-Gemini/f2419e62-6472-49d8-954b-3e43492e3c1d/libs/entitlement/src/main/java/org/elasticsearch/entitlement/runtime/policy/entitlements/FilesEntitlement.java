/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.runtime.policy.entitlements;

import org.elasticsearch.core.Strings;
import org.elasticsearch.entitlement.runtime.policy.ExternalEntitlement;
import org.elasticsearch.entitlement.runtime.policy.FileUtils;
import org.elasticsearch.entitlement.runtime.policy.PathLookup;
import org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir;
import org.elasticsearch.entitlement.runtime.policy.Platform;
import org.elasticsearch.entitlement.runtime.policy.PolicyValidationException;

import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.stream.Stream;

/**
 * @brief Represents an entitlement governing file system access, specifying paths and allowed modes (read/read-write).
 *
 * This record encapsulates a list of {@link FileData} entries, each detailing a specific
 * file or directory path along with its permitted access mode and other attributes.
 * It serves as a declarative way to define what files and directories a component
 * or module is allowed to interact with.
 *
 * Functional Utility: Provides a granular mechanism for controlling file system operations,
 *                     essential for enforcing security policies and containing the impact
 *                     of potentially compromised components within an Elasticsearch instance.
 * Architecture: Supports various ways to specify file paths: absolute, relative to
 *               predefined base directories (e.g., config, data, plugins), or dynamically
 *               resolved from settings. It also includes support for platform-specific
 *               entitlements and marking paths as exclusive.
 *
 * @param filesData A {@link List} of {@link FileData} entries, each specifying a file path and its access mode.
 */
public record FilesEntitlement(List<FileData> filesData) implements Entitlement {

    public static final String SEPARATOR = FileSystems.getDefault().getSeparator();

    /**
     * @brief An empty {@link FilesEntitlement} instance.
     * Functional Utility: Represents a file entitlement that grants no access,
     *                     useful as a default or for cases where no file entitlements are defined.
     */
    public static final FilesEntitlement EMPTY = new FilesEntitlement(List.of());

    /**
     * @brief Defines the access mode for a file entitlement.
     * Functional Utility: Specifies the type of operations permitted on a file or directory.
     */
    public enum Mode {
        /**
         * @brief Grants read-only access.
         */
        READ,
        /**
         * @brief Grants both read and write access.
         */
        READ_WRITE
    }

    public sealed interface FileData {

        /**
         * @brief Resolves the specified path(s) based on the provided {@link PathLookup}.
         * @param pathLookup The {@link PathLookup} instance used to resolve paths, especially for base directories.
         * @return A {@link Stream} of resolved {@link Path} objects.
         * Functional Utility: Translates abstract path specifications (e.g., relative to a base directory)
         *                     into concrete, absolute file system paths.
         */
        Stream<Path> resolvePaths(PathLookup pathLookup);

        /**
         * @brief Returns the access {@link Mode} for this file data entry.
         * @return The access {@link Mode} (READ or READ_WRITE).
         * Functional Utility: Indicates the level of permission granted for the specified path.
         */
        Mode mode();

        /**
         * @brief Returns whether this file data entry grants exclusive access.
         * @return `true` if access is exclusive, `false` otherwise.
         * Functional Utility: Specifies if the path is exclusively controlled by this entitlement,
         *                     preventing other entitlements from affecting it.
         */
        boolean exclusive();

        /**
         * @brief Returns a new {@link FileData} instance with the specified exclusive flag.
         * @param exclusive The new exclusive flag.
         * @return A new {@link FileData} instance with the updated exclusive flag.
         * Functional Utility: Allows modification of the exclusive property without mutating the original object.
         */
        FileData withExclusive(boolean exclusive);

        /**
         * @brief Returns the {@link Platform} this file data entry is specific to, or `null` if it's platform-agnostic.
         * @return The target {@link Platform} or `null`.
         * Functional Utility: Enables platform-specific entitlements, ensuring rules are applied only
         *                     on relevant operating systems.
         */
        Platform platform();

        /**
         * @brief Returns a new {@link FileData} instance with the specified platform.
         * @param platform The new {@link Platform} to associate with this file data.
         * @return A new {@link FileData} instance with the updated platform.
         * Functional Utility: Allows modification of the platform property without mutating the original object.
         */
        FileData withPlatform(Platform platform);

        /**
         * @brief Provides a human-readable description of this file data entry.
         * @return A {@link String} describing the entitlement.
         * Functional Utility: Useful for logging and debugging purposes to understand the details of a file entitlement.
         */
        String description();

        /**
         * @brief Creates a {@link FileData} for an absolute path.
         * @param path The absolute {@link Path}.
         * @param mode The access {@link Mode}.
         * @return A new {@link FileData} instance for an absolute path.
         */
        static FileData ofPath(Path path, Mode mode) {
            return new AbsolutePathFileData(path, mode, null, false);
        }

        /**
         * @brief Creates a {@link FileData} for a base directory path.
         * @param baseDir The {@link BaseDir} enum value representing a known Elasticsearch base directory.
         * @param mode The access {@link Mode}.
         * @return A new {@link FileData} instance for a base directory.
         */
        static FileData ofBaseDirPath(BaseDir baseDir, Mode mode) {
            return new RelativePathFileData(Path.of(""), baseDir, mode, null, false);
        }

        /**
         * @brief Creates a {@link FileData} for a path relative to a base directory.
         * @param relativePath The path relative to the `baseDir`.
         * @param baseDir The {@link BaseDir} enum value.
         * @param mode The access {@link Mode}.
         * @return A new {@link FileData} instance for a relative path.
         */
        static FileData ofRelativePath(Path relativePath, BaseDir baseDir, Mode mode) {
            return new RelativePathFileData(relativePath, baseDir, mode, null, false);
        }

        /**
         * @brief Creates a {@link FileData} for a path resolved from a setting key relative to a base directory.
         * @param setting The setting key whose value is a path.
         * @param baseDir The {@link BaseDir} enum value to resolve the setting path against.
         * @param mode The access {@link Mode}.
         * @return A new {@link FileData} instance for a path derived from a setting.
         */
        static FileData ofPathSetting(String setting, BaseDir baseDir, Mode mode) {
            return new PathSettingFileData(setting, baseDir, mode, null, false);
        }
    }

    private record AbsolutePathFileData(Path path, Mode mode, Platform platform, boolean exclusive) implements FileData {

        /**
         * @brief Constructs an {@link AbsolutePathFileData} entry.
         * @param path The absolute {@link Path} for this entitlement.
         * @param mode The access {@link Mode} (READ or READ_WRITE).
         * @param platform The {@link Platform} this entitlement applies to, or `null` for all platforms.
         * @param exclusive `true` if this entitlement is exclusive, `false` otherwise.
         */
        @Override
        public AbsolutePathFileData withExclusive(boolean exclusive) {
            return new AbsolutePathFileData(path, mode, platform, exclusive);
        }

        @Override
        public FileData withPlatform(Platform platform) {
            if (platform == platform()) {
                return this;
            }
            return new AbsolutePathFileData(path, mode, platform, exclusive);
        }

        @Override
        public Stream<Path> resolvePaths(PathLookup pathLookup) {
            return Stream.of(path);
        }

        @Override
        public String description() {
            return Strings.format("[%s] %s%s", mode, path.toAbsolutePath().normalize(), exclusive ? " (exclusive)" : "");
        }
    }

    private record RelativePathFileData(Path relativePath, BaseDir baseDir, Mode mode, Platform platform, boolean exclusive)
        implements
            FileData {

        /**
         * @brief Constructs a {@link RelativePathFileData} entry.
         * @param relativePath The path relative to the `baseDir`.
         * @param baseDir The {@link BaseDir} enum value representing a known Elasticsearch base directory.
         * @param mode The access {@link Mode} (READ or READ_WRITE).
         * @param platform The {@link Platform} this entitlement applies to, or `null` for all platforms.
         * @param exclusive `true` if this entitlement is exclusive, `false` otherwise.
         */
        @Override
        public RelativePathFileData withExclusive(boolean exclusive) {
            return new RelativePathFileData(relativePath, baseDir, mode, platform, exclusive);
        }

        @Override
        public FileData withPlatform(Platform platform) {
            if (platform == platform()) {
                return this;
            }
            return new RelativePathFileData(relativePath, baseDir, mode, platform, exclusive);
        }

        @Override
        public Stream<Path> resolvePaths(PathLookup pathLookup) {
            return pathLookup.getBaseDirPaths(baseDir).map(path -> path.resolve(relativePath));
        }

        @Override
        public String description() {
            return Strings.format("[%s] <%s>%s%s%s", mode, baseDir, SEPARATOR, relativePath, exclusive ? " (exclusive)" : "");
        }
    }

    private record PathSettingFileData(String setting, BaseDir baseDir, Mode mode, Platform platform, boolean exclusive)
        implements
            FileData {

        /**
         * @brief Constructs a {@link PathSettingFileData} entry.
         * @param setting The name of the setting that holds the path value.
         * @param baseDir The {@link BaseDir} enum value to resolve the setting path against.
         * @param mode The access {@link Mode} (READ or READ_WRITE).
         * @param platform The {@link Platform} this entitlement applies to, or `null` for all platforms.
         * @param exclusive `true` if this entitlement is exclusive, `false` otherwise.
         */
        @Override
        public PathSettingFileData withExclusive(boolean exclusive) {
            return new PathSettingFileData(setting, baseDir, mode, platform, exclusive);
        }

        @Override
        public FileData withPlatform(Platform platform) {
            if (platform == platform()) {
                return this;
            }
            return new PathSettingFileData(setting, baseDir, mode, platform, exclusive);
        }

        @Override
        public Stream<Path> resolvePaths(PathLookup pathLookup) {
            return pathLookup.resolveSettingPaths(baseDir, setting);
        }

        @Override
        public String description() {
            return Strings.format("[%s] <%s>%s<%s>%s", mode, baseDir, SEPARATOR, setting, exclusive ? " (exclusive)" : "");
        }
    }

    private static Mode parseMode(String mode) {
        /**
         * Block Logic: Parses a string representation of an access mode into its corresponding {@link Mode} enum value.
         * Functional Utility: Provides a robust way to convert string-based policy configurations into strongly typed enums,
         *                     ensuring only valid access modes are processed.
         * Pre-condition: `mode` is a string value.
         * Invariant: An {@link PolicyValidationException} is thrown if `mode` does not match "read" or "read_write".
         */
        if (mode.equals("read")) {
            return Mode.READ;
        } else if (mode.equals("read_write")) {
            return Mode.READ_WRITE;
        } else {
            throw new PolicyValidationException("invalid mode: " + mode + ", valid values: [read, read_write]");
        }
    }

    private static Platform parsePlatform(String platform) {
        /**
         * Block Logic: Parses a string representation of a platform into its corresponding {@link Platform} enum value.
         * Functional Utility: Provides a robust way to convert string-based policy configurations into strongly typed enums,
         *                     ensuring only valid platforms are processed.
         * Pre-condition: `platform` is a string value.
         * Invariant: An {@link PolicyValidationException} is thrown if `platform` does not match "linux", "macos", or "windows".
         */
        if (platform.equals("linux")) {
            return Platform.LINUX;
        } else if (platform.equals("macos")) {
            return Platform.MACOS;
        } else if (platform.equals("windows")) {
            return Platform.WINDOWS;
        } else {
            throw new PolicyValidationException("invalid platform: " + platform + ", valid values: [linux, macos, windows]");
        }
    }

    private static BaseDir parseBaseDir(String baseDir) {
        /**
         * Block Logic: Maps a string representation of a base directory to its corresponding {@link BaseDir} enum value.
         * Functional Utility: Provides a flexible mechanism to specify common Elasticsearch directories
         *                     in entitlement policies, converting them into structured enum types.
         * Pre-condition: `baseDir` is a string value.
         * Invariant: An {@link PolicyValidationException} is thrown if `baseDir` does not match any known base directories.
         */
        return switch (baseDir) {
            case "config" -> BaseDir.CONFIG;
            case "data" -> BaseDir.DATA;
            case "home" -> BaseDir.USER_HOME;
            // it would be nice to limit this to just ES modules, but we don't have a way to plumb that through to here
            // however, we still don't document in the error case below that shared_repo is valid
            case "shared_repo" -> BaseDir.SHARED_REPO;
            default -> throw new PolicyValidationException(
                "invalid relative directory: " + baseDir + ", valid values: [config, data, home]"
            );
        };
    }

    @ExternalEntitlement(parameterNames = { "paths" }, esModulesOnly = false)
    @SuppressWarnings("unchecked")
    public static FilesEntitlement build(List<Object> paths) {
        /**
         * Block Logic: Validates that the list of paths provided for the entitlement is not null or empty.
         * Functional Utility: Enforces a basic structural requirement for file entitlements, ensuring
         *                     that a policy explicitly specifies at least one path.
         * Pre-condition: `paths` is the raw list of objects from the entitlement definition.
         * Invariant: An {@link PolicyValidationException} is thrown if the path list is invalid.
         */
        if (paths == null || paths.isEmpty()) {
            throw new PolicyValidationException("must specify at least one path");
        }
        /**
         * Functional Utility: A lambda function to safely extract and validate string values from a map.
         *                     It removes the key from the map and throws a validation exception if the value is not a string.
         */
        BiFunction<Map<String, Object>, String, String> checkString = (values, key) -> {
            Object value = values.remove(key);
            if (value == null) {
                return null;
            } else if (value instanceof String str) {
                return str;
            }
            throw new PolicyValidationException(
                "expected ["
                    + key
                    + "] to be type ["
                    + String.class.getSimpleName()
                    + "] but found type ["
                    + value.getClass().getSimpleName()
                    + "]"
            );
        };
        /**
         * Functional Utility: A lambda function to safely extract and validate boolean values from a map.
         *                     It removes the key from the map and throws a validation exception if the value is not a boolean.
         */
        BiFunction<Map<String, Object>, String, Boolean> checkBoolean = (values, key) -> {
            Object value = values.remove(key);
            if (value == null) {
                return null;
            } else if (value instanceof Boolean bool) {
                return bool;
            }
            throw new PolicyValidationException(
                "expected ["
                    + key
                    + "] to be type ["
                    + boolean.class.getSimpleName()
                    + "] but found type ["
                    + value.getClass().getSimpleName()
                    + "]"
            );
        };
        List<FileData> filesData = new ArrayList<>();
        /**
         * Block Logic: Iterates through each file entry provided in the `paths` list.
         * Functional Utility: Parses and validates the attributes of each file entitlement definition,
         *                     constructing a {@link FileData} object for each.
         * Pre-condition: Each object in `paths` is expected to be a `Map<String, Object>` representing file data.
         * Invariant: Each file entry is processed, and its details are extracted and validated.
         */
        for (Object object : paths) {
            Map<String, Object> file = new HashMap<>((Map<String, Object>) object);
            String pathAsString = checkString.apply(file, "path");
            String relativePathAsString = checkString.apply(file, "relative_path");
            String relativeTo = checkString.apply(file, "relative_to");
            String pathSetting = checkString.apply(file, "path_setting");
            String settingBaseDirAsString = checkString.apply(file, "basedir_if_relative");
            String modeAsString = checkString.apply(file, "mode");
            String platformAsString = checkString.apply(file, "platform");
            Boolean exclusiveBoolean = checkBoolean.apply(file, "exclusive");
            boolean exclusive = exclusiveBoolean != null && exclusiveBoolean;

            /**
             * Block Logic: Checks for any unparsed or unknown keys in a file entitlement entry.
             * Functional Utility: Ensures that only recognized attributes are present in the entitlement definition,
             *                     preventing misconfigurations due to typos or unsupported properties.
             * Invariant: An {@link PolicyValidationException} is thrown if unexpected keys are found.
             */
            if (file.isEmpty() == false) {
                throw new PolicyValidationException("unknown key(s) [" + file + "] in a listed file for files entitlement");
            }
            int foundKeys = (pathAsString != null ? 1 : 0) + (relativePathAsString != null ? 1 : 0) + (pathSetting != null ? 1 : 0);
            /**
             * Block Logic: Ensures that exactly one path-defining key ("path", "relative_path", or "path_setting") is present.
             * Functional Utility: Enforces that each file entitlement entry unambiguously specifies its target path.
             * Invariant: An {@link PolicyValidationException} is thrown if the path definition is ambiguous or missing.
             */
            if (foundKeys != 1) {
                throw new PolicyValidationException(
                    "a files entitlement entry must contain one of " + "[path, relative_path, path_setting]"
                );
            }

            /**
             * Block Logic: Validates the presence of a 'mode' attribute.
             * Functional Utility: Ensures that every file entitlement explicitly defines its access mode.
             * Invariant: An {@link PolicyValidationException} is thrown if 'mode' is not specified.
             */
            if (modeAsString == null) {
                throw new PolicyValidationException("files entitlement must contain 'mode' for every listed file");
            }
            Mode mode = parseMode(modeAsString);
            Platform platform = null;
            if (platformAsString != null) {
                platform = parsePlatform(platformAsString);
            }

            /**
             * Block Logic: Validates that 'relative_to' is only used in conjunction with 'relative_path'.
             * Functional Utility: Enforces logical consistency in path specification, preventing invalid combinations.
             * Invariant: An {@link PolicyValidationException} is thrown if 'relative_to' is used incorrectly.
             */
            if (relativeTo != null && relativePathAsString == null) {
                throw new PolicyValidationException("'relative_to' may only be used with 'relative_path'");
            }

            /**
             * Block Logic: Validates that 'basedir_if_relative' is only used with 'path_setting'.
             * Functional Utility: Ensures logical consistency for path definitions tied to settings.
             * Invariant: An {@link PolicyValidationException} is thrown if 'basedir_if_relative' is used incorrectly.
             */
            if (settingBaseDirAsString != null && pathSetting == null) {
                throw new PolicyValidationException("'basedir_if_relative' may only be used with 'path_setting'");
            }

            final FileData fileData;
            /**
             * Block Logic: Determines the type of {@link FileData} to create based on the path specification.
             * Functional Utility: Creates the appropriate concrete {@link FileData} instance (`RelativePathFileData`,
             *                     `AbsolutePathFileData`, or `PathSettingFileData`) based on the parsed attributes.
             * Pre-condition: Exactly one of `relativePathAsString`, `pathAsString`, or `pathSetting` is non-null.
             * Invariant: The correct `FileData` subclass is instantiated and configured.
             */
            if (relativePathAsString != null) {
                /**
                 * Block Logic: Validates that 'relative_to' is provided when 'relative_path' is used.
                 * Functional Utility: Ensures that relative paths have a defined base for resolution.
                 * Invariant: Throws {@link PolicyValidationException} if `relativeTo` is missing.
                 */
                if (relativeTo == null) {
                    throw new PolicyValidationException("files entitlement with a 'relative_path' must specify 'relative_to'");
                }
                BaseDir baseDir = parseBaseDir(relativeTo);
                Path relativePath = Path.of(relativePathAsString);
                /**
                 * Block Logic: Validates that a 'relative_path' is indeed relative.
                 * Functional Utility: Prevents absolute paths from being mistakenly treated as relative,
                 *                     maintaining consistency in path interpretation.
                 * Invariant: Throws {@link PolicyValidationException} if `relativePathAsString` is absolute.
                 */
                if (FileUtils.isAbsolutePath(relativePathAsString)) {
                    throw new PolicyValidationException("'relative_path' [" + relativePathAsString + "] must be relative");
                }
                fileData = FileData.ofRelativePath(relativePath, baseDir, mode);
            } else if (pathAsString != null) {
                Path path = Path.of(pathAsString);
                /**
                 * Block Logic: Validates that a 'path' (absolute path) is indeed absolute.
                 * Functional Utility: Ensures correct interpretation of paths explicitly defined as absolute.
                 * Invariant: Throws {@link PolicyValidationException} if `pathAsString` is not absolute.
                 */
                if (FileUtils.isAbsolutePath(pathAsString) == false) {
                    throw new PolicyValidationException("'path' [" + pathAsString + "] must be absolute");
                }
                fileData = FileData.ofPath(path, mode);
            } else if (pathSetting != null) {
                /**
                 * Block Logic: Validates that 'basedir_if_relative' is provided when 'path_setting' is used.
                 * Functional Utility: Ensures that paths resolved from settings have a defined base for interpretation.
                 * Invariant: Throws {@link PolicyValidationException} if `settingBaseDirAsString` is missing.
                 */
                if (settingBaseDirAsString == null) {
                    throw new PolicyValidationException("files entitlement with a 'path_setting' must specify 'basedir_if_relative'");
                }
                BaseDir baseDir = parseBaseDir(settingBaseDirAsString);
                fileData = FileData.ofPathSetting(pathSetting, baseDir, mode);
            } else {
                throw new AssertionError("File entry validation error");
            }
            filesData.add(fileData.withPlatform(platform).withExclusive(exclusive));
        }
        return new FilesEntitlement(filesData);
    }
}
