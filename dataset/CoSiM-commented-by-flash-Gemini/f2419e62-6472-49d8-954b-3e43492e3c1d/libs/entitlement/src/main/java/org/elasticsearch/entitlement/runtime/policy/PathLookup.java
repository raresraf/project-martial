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
import java.util.stream.Stream;

/**
 * @brief Defines a contract for resolving various file system paths relevant to Elasticsearch entitlements.
 *
 * This interface provides methods to abstract away the specifics of path resolution,
 * allowing entitlement policies to refer to logical locations (e.g., config directory,
 * plugin directory) rather than hardcoded absolute paths. This is critical for
 * maintaining portability and flexibility across different environments.
 *
 * Functional Utility: Serves as a central registry for known and dynamically resolved
 *                     file system locations, which are then used by {@link FileAccessTree}
 *                     and other entitlement components to enforce access policies.
 * Architectural Role: Decouples the definition of entitlement rules from the underlying
 *                     file system structure, promoting a more declarative and maintainable
 *                     security framework.
 */
public interface PathLookup {
    enum BaseDir {
        /**
         * @brief Represents the user's home directory.
         */
        USER_HOME,
        /**
         * @brief Represents the Elasticsearch configuration directory.
         */
        CONFIG,
        /**
         * @brief Represents the Elasticsearch data directory.
         */
        DATA,
        /**
         * @brief Represents a shared repository directory.
         */
        SHARED_REPO,
        /**
         * @brief Represents the Elasticsearch library directory.
         */
        LIB,
        /**
         * @brief Represents the Elasticsearch modules directory.
         */
        MODULES,
        /**
         * @brief Represents the Elasticsearch plugins directory.
         */
        PLUGINS,
        /**
         * @brief Represents the Elasticsearch logs directory.
         */
        LOGS,
        /**
         * @brief Represents the Elasticsearch temporary directory.
         */
        TEMP
    }

    /**
     * @brief Returns the {@link Path} to the PID file.
     * @return The {@link Path} to the PID file, or `null` if not configured.
     * Functional Utility: Provides access to the location of the process ID file,
     *                     which might be subject to entitlement checks.
     */
    Path pidFile();

    /**
     * @brief Returns a stream of all concrete {@link Path}s associated with a given {@link BaseDir}.
     * @param baseDir The {@link BaseDir} enum value representing the type of base directory to look up.
     * @return A {@link Stream} of {@link Path} objects.
     * Functional Utility: Provides all physical locations corresponding to a logical base directory,
     *                     accommodating scenarios where a base directory might map to multiple paths (e.g., data paths).
     */
    Stream<Path> getBaseDirPaths(BaseDir baseDir);

    /**
     * @brief Resolves paths dynamically from a named setting, relative to a given base directory.
     * @param baseDir The {@link BaseDir} enum value used as the base for resolving the setting's path.
     * @param settingName The name of the setting whose value(s) represent file paths.
     * @return A {@link Stream} of all concrete {@link Path}s obtained by resolving the setting's values
     *         under all paths of the given `baseDir`.
     * Functional Utility: Allows entitlement policies to refer to paths that are configurable via Elasticsearch settings,
     *                     making the policies more dynamic and adaptable to deployment-specific configurations.
     */
    Stream<Path> resolveSettingPaths(BaseDir baseDir, String settingName);
}
