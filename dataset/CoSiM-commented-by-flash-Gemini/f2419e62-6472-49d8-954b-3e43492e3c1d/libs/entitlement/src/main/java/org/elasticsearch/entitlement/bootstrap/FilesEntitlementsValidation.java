/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.bootstrap;

import org.elasticsearch.core.Strings;
import org.elasticsearch.entitlement.runtime.policy.FileAccessTree;
import org.elasticsearch.entitlement.runtime.policy.PathLookup;
import org.elasticsearch.entitlement.runtime.policy.Policy;
import org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.CONFIG;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.LIB;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.MODULES;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.PLUGINS;
import static org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement.Mode.READ;
import static org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement.Mode.READ_WRITE;

/**
 * @brief Provides static utility methods for validating file access entitlements defined in policies.
 *
 * This class ensures that plugins and modules adhere to predefined restrictions regarding
 * file system access. Specifically, it validates that no policy grants read or write access
 * to directories that are inherently forbidden (e.g., core Elasticsearch directories like
 * plugins, modules, lib, or config directories).
 *
 * Functional Utility: Enforces a security boundary by preventing unintended or malicious
 *                     file system interactions from plugins and modules, contributing to
 *                     the overall stability and security of the Elasticsearch system.
 * Architecture: Integrates with {@link FileAccessTree} to perform efficient path checks
 *               against a set of forbidden directories.
 */
class FilesEntitlementsValidation {

    static void validate(Map<String, Policy> pluginPolicies, PathLookup pathLookup) {
        Set<Path> readAccessForbidden = new HashSet<>();
        /**
         * Block Logic: Populates the `readAccessForbidden` set with absolute and normalized paths
         *              of core Elasticsearch directories (plugins, modules, lib).
         * Functional Utility: Defines the critical system locations that plugins are inherently not
         *                     allowed to read from without explicit, highly scrutinized permission.
         * Pre-condition: `pathLookup` is an initialized instance providing access to base directory paths.
         * Invariant: The set `readAccessForbidden` contains canonical paths of directories considered off-limits for read access.
         */
        pathLookup.getBaseDirPaths(PLUGINS).forEach(p -> readAccessForbidden.add(p.toAbsolutePath().normalize()));
        pathLookup.getBaseDirPaths(MODULES).forEach(p -> readAccessForbidden.add(p.toAbsolutePath().normalize()));
        pathLookup.getBaseDirPaths(LIB).forEach(p -> readAccessForbidden.add(p.toAbsolutePath().normalize()));
        Set<Path> writeAccessForbidden = new HashSet<>();
        /**
         * Block Logic: Populates the `writeAccessForbidden` set with absolute and normalized paths
         *              of the Elasticsearch configuration directory.
         * Functional Utility: Defines the critical system location that plugins are inherently not
         *                     allowed to write to, to protect system configuration integrity.
         * Pre-condition: `pathLookup` is an initialized instance providing access to base directory paths.
         * Invariant: The set `writeAccessForbidden` contains canonical paths of directories considered off-limits for write access.
         */
        pathLookup.getBaseDirPaths(CONFIG).forEach(p -> writeAccessForbidden.add(p.toAbsolutePath().normalize()));
        /**
         * Block Logic: Iterates through each {@link Policy} defined for installed plugins.
         * Functional Utility: Ensures that each plugin's declared file access entitlements are
         *                     validated against the system's forbidden path lists.
         * Pre-condition: `pluginPolicies` maps plugin names to their loaded policies.
         * Invariant: Every plugin policy is examined for compliance with file access rules.
         */
        for (var pluginPolicy : pluginPolicies.entrySet()) {
            /**
             * Block Logic: Iterates through each {@link Policy.PolicyScope} within a plugin's policy.
             * Functional Utility: Allows for the validation of entitlements granularly defined
             *                     within different scopes (e.g., modules) of a single plugin.
             * Pre-condition: `scope` represents a defined access policy scope for a module or component.
             * Invariant: All defined policy scopes for the current plugin are checked.
             */
            for (var scope : pluginPolicy.getValue().scopes()) {
                /**
                 * Block Logic: Filters for and extracts the {@link FilesEntitlement} from the current policy scope.
                 * Functional Utility: Isolates the specific entitlement object responsible for file system access rules.
                 * Pre-condition: `scope.entitlements()` may contain various types of entitlements.
                 * Invariant: If a {@link FilesEntitlement} is present, it is prepared for validation.
                 */
                var filesEntitlement = scope.entitlements()
                    .stream()
                    .filter(x -> x instanceof FilesEntitlement)
                    .map(x -> ((FilesEntitlement) x))
                    .findFirst();
                /**
                 * Block Logic: Checks if a {@link FilesEntitlement} was found for the current scope.
                 * Functional Utility: If a files entitlement is present, it is used to construct a
                 *                     {@link FileAccessTree} and perform read/write access validations.
                 * Pre-condition: `filesEntitlement` is an Optional containing the relevant entitlement.
                 * Invariant: If files entitlements are present, they are validated against forbidden paths;
                 *            otherwise, this step is skipped for the current scope.
                 */
                if (filesEntitlement.isPresent()) {
                    var fileAccessTree = FileAccessTree.withoutExclusivePaths(filesEntitlement.get(), pathLookup, List.of());
                    validateReadFilesEntitlements(pluginPolicy.getKey(), scope.moduleName(), fileAccessTree, readAccessForbidden);
                    validateWriteFilesEntitlements(pluginPolicy.getKey(), scope.moduleName(), fileAccessTree, writeAccessForbidden);
                }
            }
        }
    }

    private static IllegalArgumentException buildValidationException(
        String componentName,
        String moduleName,
        Path forbiddenPath,
        FilesEntitlement.Mode mode
    ) {
        /**
         * @brief Constructs an {@link IllegalArgumentException} for a file entitlement validation failure.
         * @param componentName The name of the component associated with the invalid policy.
         * @param moduleName The name of the module within the component.
         * @param forbiddenPath The specific path that violated the entitlement rule.
         * @param mode The access {@link FilesEntitlement.Mode} (READ or READ_WRITE) that was forbidden.
         * @return An {@link IllegalArgumentException} detailing the validation error.
         * Functional Utility: Standardizes the error message format for entitlement violations,
         *                     making it easier to understand and debug policy configuration issues.
         */
        return new IllegalArgumentException(
            Strings.format(
                "policy for module [%s] in [%s] has an invalid file entitlement. Any path under [%s] is forbidden for mode [%s].",
                moduleName,
                componentName,
                forbiddenPath,
                mode
            )
        );
    }

    private static void validateReadFilesEntitlements(
        String componentName,
        String moduleName,
        FileAccessTree fileAccessTree,
        Set<Path> readForbiddenPaths
    ) {
        /**
         * Block Logic: Iterates through each path that is explicitly marked as forbidden for read access.
         * Functional Utility: Systematically checks if any of the plugin's or module's declared read entitlements
         *                     unintentionally grant access to restricted system locations.
         * Pre-condition: `readForbiddenPaths` contains canonical paths that are not allowed to be read.
         * Invariant: Each forbidden path is checked against the plugin's effective read access tree.
         */
        for (Path forbiddenPath : readForbiddenPaths) {
            /**
             * Block Logic: Determines if the plugin's {@link FileAccessTree} grants read access to a `forbiddenPath`.
             * Functional Utility: If access is granted to a forbidden path, it indicates a policy violation,
             *                     and a validation exception is thrown.
             * Pre-condition: `fileAccessTree` is the compiled access policy for the current plugin/module.
             * Invariant: An {@link IllegalArgumentException} is thrown immediately upon detecting a violation,
             *            preventing the loading of an insecure policy.
             */
            if (fileAccessTree.canRead(forbiddenPath)) {
                throw buildValidationException(componentName, moduleName, forbiddenPath, READ);
            }
        }
    }

    private static void validateWriteFilesEntitlements(
        String componentName,
        String moduleName,
        FileAccessTree fileAccessTree,
        Set<Path> writeForbiddenPaths
    ) {
        /**
         * Block Logic: Iterates through each path that is explicitly marked as forbidden for write access.
         * Functional Utility: Systematically checks if any of the plugin's or module's declared write entitlements
         *                     unintentionally grant access to restricted system locations (e.g., config directories).
         * Pre-condition: `writeForbiddenPaths` contains canonical paths that are not allowed to be written to.
         * Invariant: Each forbidden path is checked against the plugin's effective write access tree.
         */
        for (Path forbiddenPath : writeForbiddenPaths) {
            /**
             * Block Logic: Determines if the plugin's {@link FileAccessTree} grants write access to a `forbiddenPath`.
             * Functional Utility: If write access is granted to a forbidden path, it indicates a policy violation,
             *                     and a validation exception is thrown.
             * Pre-condition: `fileAccessTree` is the compiled access policy for the current plugin/module.
             * Invariant: An {@link IllegalArgumentException} is thrown immediately upon detecting a violation,
             *            preventing the loading of an insecure policy.
             */
            if (fileAccessTree.canWrite(forbiddenPath)) {
                throw buildValidationException(componentName, moduleName, forbiddenPath, READ_WRITE);
            }
        }
    }
}
