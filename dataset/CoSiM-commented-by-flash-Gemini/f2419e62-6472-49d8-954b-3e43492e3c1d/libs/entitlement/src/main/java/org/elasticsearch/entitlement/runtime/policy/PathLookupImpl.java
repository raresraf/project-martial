/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.runtime.policy;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;
import java.util.stream.Stream;

import static java.util.Objects.requireNonNull;

/**
 * @brief A standard implementation of the {@link PathLookup} interface for resolving known Elasticsearch paths.
 *
 * This record provides a concrete mechanism to look up various base directories
 * (e.g., home, config, data, plugins, modules, logs, temp) and resolve paths based on settings.
 * It serves as the primary source of truth for file system locations that are
 * relevant for entitlement checks and other bootstrap operations.
 *
 * Functional Utility: Centralizes and standardizes access to critical file system paths,
 *                     ensuring consistency and correctness across different components
 *                     that rely on these locations for configuration, data storage, or plugin management.
 * Architectural Role: Acts as a configurable dependency, allowing the Elasticsearch environment
 *                     to be defined and queried for path information, which is then consumed
 *                     by security and other core modules.
 * @param homeDir The Elasticsearch home directory.
 * @param configDir The configuration directory.
 * @param dataDirs An array of data directories.
 * @param sharedRepoDirs An array of shared repository directories.
 * @param libDir The library directory.
 * @param modulesDir The modules directory.
 * @param pluginsDir The plugins directory.
 * @param logsDir The logs directory.
 * @param tempDir The temporary directory.
 * @param pidFile The path to the PID file.
 * @param settingResolver A function to resolve setting names into a stream of path strings.
 */
public record PathLookupImpl(
    Path homeDir,
    Path configDir,
    Path[] dataDirs,
    Path[] sharedRepoDirs,
    Path libDir,
    Path modulesDir,
    Path pluginsDir,
    Path logsDir,
    Path tempDir,
    Path pidFile,
    Function<String, Stream<String>> settingResolver
) implements PathLookup {

    public PathLookupImpl {
        /**
         * Block Logic: Ensures that the `homeDir` parameter is not null.
         * Functional Utility: Guarantees the presence of a valid Elasticsearch home directory,
         *                     a fundamental requirement for the system to function correctly.
         * Invariant: A {@link NullPointerException} is thrown if `homeDir` is null.
         */
        requireNonNull(homeDir);
        /**
         * Block Logic: Ensures that the `dataDirs` array parameter is not null and contains at least one data directory.
         * Functional Utility: Validates that essential data storage locations are properly configured,
         *                     preventing issues related to data persistence and access.
         * Invariant: A {@link NullPointerException} is thrown if `dataDirs` is null.
         *            An {@link IllegalArgumentException} is thrown if `dataDirs` is empty.
         */
        requireNonNull(dataDirs);
        if (dataDirs.length == 0) {
            throw new IllegalArgumentException("must provide at least one data directory");
        }
        /**
         * Block Logic: Ensures that the `sharedRepoDirs` array parameter is not null.
         * Functional Utility: Guarantees that shared repository paths, even if empty, are properly initialized.
         * Invariant: A {@link NullPointerException} is thrown if `sharedRepoDirs` is null.
         */
        requireNonNull(sharedRepoDirs);
        /**
         * Block Logic: Ensures that the `configDir` parameter is not null.
         * Functional Utility: Guarantees the presence of a valid configuration directory.
         * Invariant: A {@link NullPointerException} is thrown if `configDir` is null.
         */
        requireNonNull(configDir);
        /**
         * Block Logic: Ensures that the `libDir` parameter is not null.
         * Functional Utility: Guarantees the presence of a valid library directory.
         * Invariant: A {@link NullPointerException} is thrown if `libDir` is null.
         */
        requireNonNull(libDir);
        /**
         * Block Logic: Ensures that the `modulesDir` parameter is not null.
         * Functional Utility: Guarantees the presence of a valid modules directory.
         * Invariant: A {@link NullPointerException} is thrown if `modulesDir` is null.
         */
        requireNonNull(modulesDir);
        /**
         * Block Logic: Ensures that the `pluginsDir` parameter is not null.
         * Functional Utility: Guarantees the presence of a valid plugins directory.
         * Invariant: A {@link NullPointerException} is thrown if `pluginsDir` is null.
         */
        requireNonNull(pluginsDir);
        /**
         * Block Logic: Ensures that the `logsDir` parameter is not null.
         * Functional Utility: Guarantees the presence of a valid logs directory.
         * Invariant: A {@link NullPointerException} is thrown if `logsDir` is null.
         */
        requireNonNull(logsDir);
        /**
         * Block Logic: Ensures that the `tempDir` parameter is not null.
         * Functional Utility: Guarantees the presence of a valid temporary directory.
         * Invariant: A {@link NullPointerException} is thrown if `tempDir` is null.
         */
        requireNonNull(tempDir);
        /**
         * Block Logic: Ensures that the `settingResolver` parameter is not null.
         * Functional Utility: Guarantees the availability of a mechanism to resolve paths from settings.
         * Invariant: A {@link NullPointerException} is thrown if `settingResolver` is null.
         */
        requireNonNull(settingResolver);
    }

    @Override
    public Stream<Path> getBaseDirPaths(BaseDir baseDir) {
        /**
         * Block Logic: Maps a {@link BaseDir} enum value to its corresponding {@link Stream} of {@link Path}s.
         * Functional Utility: Provides a centralized and type-safe mechanism to retrieve all physical paths
         *                     associated with a logical base directory, supporting single and multiple path configurations.
         * Pre-condition: `baseDir` is a valid {@link BaseDir} enum value.
         * Invariant: Returns a {@link Stream} of {@link Path} objects; never returns null.
         */
        return switch (baseDir) {
            case USER_HOME -> Stream.of(homeDir);
            case DATA -> Arrays.stream(dataDirs);
            case SHARED_REPO -> Arrays.stream(sharedRepoDirs);
            case CONFIG -> Stream.of(configDir);
            case LIB -> Stream.of(libDir);
            case MODULES -> Stream.of(modulesDir);
            case PLUGINS -> Stream.of(pluginsDir);
            case LOGS -> Stream.of(logsDir);
            case TEMP -> Stream.of(tempDir);
        };
    }

    @Override
    public Stream<Path> resolveSettingPaths(BaseDir baseDir, String settingName) {
        /**
         * Block Logic: Resolves a given `settingName` into a list of relative paths, filtering out HTTPS URLs and ensuring uniqueness.
         * Functional Utility: Extracts relevant file paths defined in settings, ensuring they are valid local paths
         *                     and preparing them for resolution against base directories.
         * Pre-condition: `settingResolver` is a functional interface capable of providing string streams for settings.
         * Invariant: `relativePaths` contains distinct, non-HTTPS {@link Path} objects derived from the setting.
         */
        List<Path> relativePaths = settingResolver.apply(settingName)
            .filter(s -> s.toLowerCase(Locale.ROOT).startsWith("https://") == false)
            .distinct()
            .map(Path::of)
            .toList();
        /**
         * Block Logic: Combines the base directory paths with the resolved relative paths from settings.
         * Functional Utility: Generates a flat stream of absolute {@link Path}s by resolving each relative path
         *                     against every applicable base directory path.
         * Pre-condition: `getBaseDirPaths(baseDir)` returns a stream of base paths, and `relativePaths` contains valid relative paths.
         * Invariant: The returned stream contains all possible concrete paths resulting from the combination.
         */
        return getBaseDirPaths(baseDir).flatMap(path -> relativePaths.stream().map(path::resolve));
    }
}
