/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.runtime.policy;

import org.elasticsearch.core.Strings;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement.Mode;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiConsumer;

import static java.util.Comparator.comparing;
import static org.elasticsearch.core.PathUtils.getDefaultFileSystem;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.CONFIG;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.TEMP;
import static org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement.Mode.READ_WRITE;

/**
 * @brief Manages file access entitlements by organizing and checking paths based on a hierarchical filesystem structure.
 *
 * This class facilitates looking up file entitlements for specific component-module combinations. It handles grants at both
 * directory and individual file levels. The core mechanism relies on normalizing paths to absolute strings with consistent
 * separators, allowing for predictable behavior across platforms.
 *
 * Internally, it optimizes permission checks by maintaining sorted arrays of paths (read, write, exclusive) and performing
 * binary searches. This approach is not a tree data structure but leverages the tree-like nature of the filesystem where
 * parent directories are prefixes of child paths. Special considerations are made for path comparison and pruning redundant
 * entries to ensure efficient and accurate lookups, especially regarding sibling paths and platform-specific path separators.
 *
 * Algorithm: Sorted array binary search with path normalization and pruning.
 * Time Complexity: O(log N) for permission checks after initial setup.
 * Space Complexity: O(N) where N is the number of distinct paths.
 */
public final class FileAccessTree {

    /**
     * @brief An intermediary record to encapsulate details of a file entitlement that grants exclusive access.
     * @param componentName The name of the component claiming exclusive access.
     * @param moduleName The name of the module within the component claiming exclusive access.
     * @param filesEntitlement The {@link FilesEntitlement} instance defining the exclusive access.
     */
    record ExclusiveFileEntitlement(String componentName, String moduleName, FilesEntitlement filesEntitlement) {}

    /**
     * @brief An intermediary record representing a path that has been exclusively claimed by a specific component and its modules.
     *        This is used for global validation of exclusive paths and building module-specific exclusive path lists.
     * @param componentName The name of the component that exclusively claims this path.
     * @param moduleNames A set of module names within the component that claim this path.
     * @param path The normalized string representation of the exclusive path.
     */
    record ExclusivePath(String componentName, Set<String> moduleNames, String path) {

        @Override
        public String toString() {
            return "[[" + componentName + "] " + moduleNames + " [" + path + "]]";
        }
    }

    /**
     * @brief Builds a consolidated list of unique exclusive paths from a collection of exclusive file entitlements.
     *        It aggregates exclusive paths, checks for conflicts (a path being exclusive to multiple components),
     *        and sorts them for efficient searching.
     * @param exclusiveFileEntitlements A list of {@link ExclusiveFileEntitlement} objects, each detailing exclusive access granted.
     * @param pathLookup An instance of {@link PathLookup} to resolve paths.
     * @param comparison A {@link FileAccessTreeComparison} instance for path-specific comparisons.
     * @return A sorted list of {@link ExclusivePath} objects representing all unique exclusive paths.
     * @throws IllegalArgumentException if a path is found to be exclusively claimed by more than one component.
     * Algorithm: Iterates through all exclusive file entitlements, resolves and normalizes their paths,
     *            and aggregates them into a map to detect and prevent duplicate exclusive claims before sorting.
     * Time Complexity: O(M * K * L + P log P) where M is the number of exclusive file entitlements, K is the
     *                  average number of file data entries per entitlement, L is the average number of resolved
     *                  paths per file data entry, and P is the total number of unique exclusive paths.
     */
    static List<ExclusivePath> buildExclusivePathList(
        List<ExclusiveFileEntitlement> exclusiveFileEntitlements,
        PathLookup pathLookup,
        FileAccessTreeComparison comparison
    ) {
        Map<String, ExclusivePath> exclusivePaths = new HashMap<>();
        /**
         * Block Logic: Iterates through each exclusive file entitlement (`efe`) to process its associated file data.
         * Functional Utility: This loop ensures that all defined exclusive entitlements are considered for path aggregation.
         * Invariant: Each `efe` from `exclusiveFileEntitlements` is processed exactly once.
         */
        for (ExclusiveFileEntitlement efe : exclusiveFileEntitlements) {
            /**
             * Block Logic: Iterates through each {@link FilesEntitlement.FileData} entry within the current
             *              exclusive file entitlement.
             * Functional Utility: Extracts individual file data configurations to determine specific paths.
             * Invariant: All `FileData` objects associated with the current `efe` are examined.
             */
            for (FilesEntitlement.FileData fd : efe.filesEntitlement().filesData()) {
                /**
                 * Block Logic: Filters file data entries to only process those explicitly marked as exclusive.
                 * Pre-condition: `fd` must not be null and represents a file data configuration.
                 * Functional Utility: Ensures that only paths designated for exclusive access are further resolved and aggregated.
                 * Invariant: Only `FileData` objects with `exclusive()` returning true proceed to path resolution.
                 */
                if (fd.exclusive()) {
                    List<Path> paths = fd.resolvePaths(pathLookup).toList();
                    /**
                     * Block Logic: Iterates through each resolved concrete file system {@link Path} derived from
                     *              the current exclusive file data entry.
                     * Functional Utility: Normalizes each path and aggregates it into a map, checking for conflicts.
                     * Invariant: All resolved paths for `fd` are normalized and added to the `exclusivePaths` map.
                     */
                    for (Path path : paths) {
                        String normalizedPath = normalizePath(path);
                        var exclusivePath = exclusivePaths.computeIfAbsent(
                            normalizedPath,
                            k -> new ExclusivePath(efe.componentName(), new HashSet<>(), normalizedPath)
                        );
                        /**
                         * Block Logic: Detects and prevents conflicts where the same normalized path is exclusively
                         *              claimed by different components.
                         * Pre-condition: `exclusivePath` exists in the map for `normalizedPath`.
                         * Functional Utility: Enforces the constraint that a path can only be exclusively managed
                         *                     by a single component.
                         * Invariant: An {@link IllegalArgumentException} is thrown if a conflict is detected,
                         *            preventing inconsistent entitlement configurations.
                         */
                        if (exclusivePath.componentName().equals(efe.componentName()) == false) {
                            throw new IllegalArgumentException(
                                "Path ["
                                    + normalizedPath
                                    + "] is already exclusive to ["
                                    + exclusivePath.componentName()
                                    + "]"
                                    + exclusivePath.moduleNames
                                    + ", cannot add exclusive access for ["
                                    + efe.componentName()
                                    + "]["
                                    + efe.moduleName
                                    + "]"
                            );
                        }
                        exclusivePath.moduleNames.add(efe.moduleName());
                    }
                }
            }
        }
        return exclusivePaths.values().stream().sorted(comparing(ExclusivePath::path, comparison.pathComparator())).distinct().toList();
    }

    /**
     * @brief Validates that there are no duplicate or overlapping exclusive paths within the provided list.
     *        This ensures the integrity of the exclusive path configuration, preventing ambiguities in access control.
     * @param exclusivePaths A sorted list of {@link ExclusivePath} objects to validate.
     * @param comparison A {@link FileAccessTreeComparison} instance for path-specific comparisons.
     * @throws IllegalArgumentException if any duplicate or overlapping exclusive paths are found.
     * Algorithm: Linear scan of a sorted list to check for immediate duplicates or parent-child overlaps.
     * Time Complexity: O(N) where N is the number of exclusive paths.
     */
    static void validateExclusivePaths(List<ExclusivePath> exclusivePaths, FileAccessTreeComparison comparison) {
        /**
         * Block Logic: Skips validation if the list of exclusive paths is empty, as there are no paths to check for conflicts.
         * Pre-condition: The `exclusivePaths` list is expected to be pre-sorted.
         * Functional Utility: Optimizes by avoiding unnecessary iteration when no exclusive paths are defined.
         * Invariant: If the list is empty, this block ensures early exit without any validation steps.
         */
        if (exclusivePaths.isEmpty() == false) {
            ExclusivePath currentExclusivePath = exclusivePaths.get(0);
            /**
             * Block Logic: Iterates through the sorted list of exclusive paths, comparing each path with its predecessor.
             * Functional Utility: Detects duplicate or overlapping exclusive path claims to maintain a consistent access policy.
             * Invariant: `currentExclusivePath` always holds the last validated exclusive path, and `nextPath`
             *            is the path currently being evaluated for conflicts.
             */
            for (int i = 1; i < exclusivePaths.size(); ++i) {
                ExclusivePath nextPath = exclusivePaths.get(i);
                /**
                 * Block Logic: Checks for conflicts where the `nextPath` is either identical to or a child of the `currentExclusivePath`.
                 * Pre-condition: `currentExclusivePath` and `nextPath` are valid {@link ExclusivePath} objects from a sorted list.
                 * Functional Utility: Ensures that no two exclusive path entries in the list represent the same physical
                 *                     path or a direct hierarchical overlap (parent-child relationship).
                 * Invariant: An `IllegalArgumentException` is thrown if a conflict (duplicate or overlap) is detected,
                 *            preventing the system from operating with an ambiguous access policy.
                 */
                if (comparison.samePath(currentExclusivePath.path(), nextPath.path)
                    || comparison.isParent(currentExclusivePath.path(), nextPath.path())) {
                    throw new IllegalArgumentException(
                        "duplicate/overlapping exclusive paths found in files entitlements: " + currentExclusivePath + " and " + nextPath
                    );
                }
                currentExclusivePath = nextPath;
            }
        }
    }

    @SuppressForbidden(reason = "we need the separator as a char, not a string")
    static char separatorChar() {
        return File.separatorChar;
    }

    private static final Logger logger = LogManager.getLogger(FileAccessTree.class);
    private static final String FILE_SEPARATOR = getDefaultFileSystem().getSeparator();
    static final FileAccessTreeComparison DEFAULT_COMPARISON = Platform.LINUX.isCurrent()
        ? new CaseSensitiveComparison(separatorChar())
        : new CaseInsensitiveComparison(separatorChar());

    private final FileAccessTreeComparison comparison;
    /**
     * lists paths that are forbidden for this component+module because some other component has granted exclusive access to one of its
     * modules
     */
    private final String[] exclusivePaths;
    /**
     * lists paths for which the component has granted read or read_write access to the module
     */
    private final String[] readPaths;
    /**
     * lists paths for which the component has granted read_write access to the module
     */
    private final String[] writePaths;

    private static String[] buildUpdatedAndSortedExclusivePaths(
        String componentName,
        String moduleName,
        List<ExclusivePath> exclusivePaths,
        FileAccessTreeComparison comparison
    ) {
        List<String> updatedExclusivePaths = new ArrayList<>();
        /**
         * Block Logic: Iterates through the global list of `exclusivePaths` to identify which paths are
         *              not exclusively claimed by the current `componentName` and `moduleName` combination.
         * Functional Utility: Filters out paths that are "self-exclusive" to avoid redundant restrictions.
         * Invariant: Each `exclusivePath` from the input list is evaluated once.
         */
        for (ExclusivePath exclusivePath : exclusivePaths) {
            /**
             * Block Logic: Adds an exclusive path to the `updatedExclusivePaths` list if it is not exclusively
             *              granted by the specified `componentName` and `moduleName`.
             * Pre-condition: `exclusivePath` is a valid {@link ExclusivePath} object.
             * Functional Utility: Ensures that only paths exclusive to other components or modules (or globally exclusive)
             *                     are included, representing real restrictions for the current context.
             * Invariant: Only genuinely external exclusive paths are propagated to the output list.
             */
            if (exclusivePath.componentName().equals(componentName) == false || exclusivePath.moduleNames().contains(moduleName) == false) {
                updatedExclusivePaths.add(exclusivePath.path());
            }
        }
        updatedExclusivePaths.sort(comparison.pathComparator());
        return updatedExclusivePaths.toArray(new String[0]);
    }

    /**
     * @brief Constructs a new {@link FileAccessTree} instance, initializing it with granted read, write, and exclusive paths.
     *        This constructor processes file entitlements, resolves paths, handles symbolic links, and prunes redundant paths.
     * @param filesEntitlement The {@link FilesEntitlement} object containing the raw file access grants.
     * @param pathLookup An instance of {@link PathLookup} to resolve paths based on their base directories.
     * @param componentPaths A {@link Collection} of base paths of the component, used to grant default read access.
     * @param sortedExclusivePaths An array of paths that are exclusively controlled by other components/modules.
     * @param comparison A {@link FileAccessTreeComparison} instance for platform-specific path comparisons.
     * Algorithm: Iterates through file entitlements, resolves paths, normalizes them, handles symlinks, and then prunes and sorts the resulting lists of paths.
     * Time Complexity: Dominated by path resolution, symlink checking, and sorting, which can be O(N log N) where N is the number of paths.
     */
    FileAccessTree(
        FilesEntitlement filesEntitlement,
        PathLookup pathLookup,
        Collection<Path> componentPaths,
        String[] sortedExclusivePaths,
        FileAccessTreeComparison comparison
    ) {
        this.comparison = comparison;
        List<String> readPaths = new ArrayList<>();
        List<String> writePaths = new ArrayList<>();
        BiConsumer<Path, Mode> addPath = (path, mode) -> {
            var normalized = normalizePath(path);
            /**
             * Block Logic: Conditionally adds the `normalized` path to the `writePaths` list.
             * Functional Utility: Records paths that have been explicitly granted `READ_WRITE` access.
             * Pre-condition: `normalized` is a canonical string representation of a file path; `mode` indicates the access level.
             * Invariant: `writePaths` will only contain paths for which `READ_WRITE` access is granted.
             */
            if (mode == READ_WRITE) {
                writePaths.add(normalized);
            }
            // Always add to readPaths as READ_WRITE implies READ.
            readPaths.add(normalized);
        };
        BiConsumer<Path, Mode> addPathAndMaybeLink = (path, mode) -> {
            addPath.accept(path, mode);
            // also try to follow symlinks. Lucene does this and writes to the target path.
            /**
             * Block Logic: Verifies the existence of the given {@link Path} on the filesystem.
             * Functional Utility: Prevents {@link IOException} from being thrown by `toRealPath()` if the path does not exist,
             *                     ensuring robust handling of symbolic links.
             * Pre-condition: `path` is a valid {@link Path} object.
             * Invariant: Only existing paths are subjected to `toRealPath()` to resolve symlinks.
             */
            if (Files.exists(path)) {
                try {
                    Path realPath = path.toRealPath();
                    /**
                     * Block Logic: Compares the resolved `realPath` with the original `path` to detect symbolic links.
                     * Functional Utility: If a symbolic link is identified (i.e., `realPath` differs), the true
                     *                     physical path is also added to the access lists to ensure proper entitlement.
                     * Pre-condition: `realPath` has been successfully obtained from `path.toRealPath()`.
                     * Invariant: Both the original path and its dereferenced real path (if it's a symlink) are
                     *            registered for access control, consistent with Lucene's behavior.
                     */
                    if (realPath.equals(path) == false) {
                        addPath.accept(realPath, mode);
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        };
        /**
         * Block Logic: Processes each {@link FilesEntitlement.FileData} entry to determine the paths and their access modes.
         * Functional Utility: Extracts and resolves the file system paths specified in the entitlement, applying
         *                     platform-specific filtering and handling symbolic links.
         * Invariant: Each `fileData` is checked for platform relevance and then its paths are resolved and added.
         */
        for (FilesEntitlement.FileData fileData : filesEntitlement.filesData()) {
            var platform = fileData.platform();
            /**
             * Block Logic: Skips processing the current file data if it is platform-specific and does not match the current platform.
             * Functional Utility: Allows for platform-conditional entitlement rules, preventing the application of
             *                     irrelevant or incompatible path grants.
             * Pre-condition: `platform` may be null or a valid {@link Platform} object.
             * Invariant: Only file data relevant to the current platform (or general data) is processed.
             */
            if (platform != null && platform.isCurrent() == false) {
                continue;
            }
            var mode = fileData.mode();
            var paths = fileData.resolvePaths(pathLookup);
            paths.forEach(path -> {
                /**
                 * Block Logic: Skips processing if a resolved path is null.
                 * Functional Utility: Handles potential issues where `resolvePaths` might return null entries,
                 *                     preventing `IOException` during path processing.
                 * Pre-condition: `path` is an individual resolved path from `fileData.resolvePaths`.
                 * Invariant: Ensures that only valid (non-null) paths are passed to `addPathAndMaybeLink`.
                 */
                if (path == null) {
                    // TODO: null paths shouldn't be allowed, but they can occur due to repo paths
                    return;
                }
                addPathAndMaybeLink.accept(path, mode);
            });
        }

        // everything has access to the temp dir, config dir, to their own dir (their own jar files) and the jdk
        pathLookup.getBaseDirPaths(TEMP).forEach(tempPath -> addPathAndMaybeLink.accept(tempPath, READ_WRITE));
        // TODO: this grants read access to the config dir for all modules until explicit read entitlements can be added
        pathLookup.getBaseDirPaths(CONFIG).forEach(configPath -> addPathAndMaybeLink.accept(configPath, Mode.READ));
        /**
         * Block Logic: Grants read access to each component's base path provided in the collection.
         * Functional Utility: Ensures that the component can always read from its own installation directories.
         * Pre-condition: `componentPaths` is a collection of valid {@link Path} objects.
         * Invariant: Each path in `componentPaths` is added to the paths with read access, and symlinks are resolved if present.
         */
        componentPaths.forEach(p -> addPathAndMaybeLink.accept(p, Mode.READ));

        // TODO: watcher uses javax.activation which looks for known mime types configuration, should this be global or explicit in watcher?
        Path jdk = Paths.get(System.getProperty("java.home"));
        addPathAndMaybeLink.accept(jdk.resolve("conf"), Mode.READ);

        readPaths.sort(comparison.pathComparator());
        writePaths.sort(comparison.pathComparator());

        this.exclusivePaths = sortedExclusivePaths;
        this.readPaths = pruneSortedPaths(readPaths, comparison).toArray(new String[0]);
        this.writePaths = pruneSortedPaths(writePaths, comparison).toArray(new String[0]);

        logger.debug(
            () -> Strings.format(
                "Created FileAccessTree with paths: exclusive [%s], read [%s], write [%s]",
                String.join(",", this.exclusivePaths),
                String.join(",", this.readPaths),
                String.join(",", this.writePaths)
            )
        );
    }

    /**
     * @brief Prunes a sorted list of paths by removing redundant child paths if their parent is already present in the list.
     *        This optimization prevents unnecessary checks against child paths when access is already determined by a parent.
     * @param paths A sorted list of path strings.
     * @param comparison A {@link FileAccessTreeComparison} instance for path-specific comparisons.
     * @return A new list containing only the pruned, non-redundant paths.
     * Algorithm: Linear scan of a sorted list, comparing adjacent paths for parent-child relationships.
     * Time Complexity: O(N) where N is the number of paths in the input list.
     */
    static List<String> pruneSortedPaths(List<String> paths, FileAccessTreeComparison comparison) {
        List<String> prunedReadPaths = new ArrayList<>();
        /**
         * Block Logic: Initializes the path pruning process only if the input list of paths is not empty.
         * Functional Utility: Avoids unnecessary processing for an empty list, returning an empty list efficiently.
         * Pre-condition: `paths` is a sorted list.
         * Invariant: If the list is empty, no pruning is performed.
         */
        if (paths.isEmpty() == false) {
            String currentPath = paths.get(0);
            prunedReadPaths.add(currentPath);
            /**
             * Block Logic: Iterates through the sorted `paths` list, starting from the second element, to identify
             *              and remove redundant (child or duplicate) paths.
             * Functional Utility: Ensures that the `prunedReadPaths` list contains only the most general paths,
             *                     where a parent path implicitly covers its child paths.
             * Invariant: `currentPath` always refers to the last path added to `prunedReadPaths`, serving as the
             *            reference for comparison with subsequent paths.
             */
            for (int i = 1; i < paths.size(); ++i) {
                String nextPath = paths.get(i);
                /**
                 * Block Logic: Determines if `nextPath` is a new, non-redundant path (not same as or child of `currentPath`).
                 * Functional Utility: Adds `nextPath` to the `prunedReadPaths` only if it is not the same as `currentPath`
                 *                     and is not a child of `currentPath`, thereby keeping only the most significant paths.
                 * Pre-condition: `currentPath` and `nextPath` are valid, normalized path strings from a sorted list.
                 * Invariant: If `nextPath` is deemed unique and independent, it is added to the pruned list and becomes
                 *            the new `currentPath` for subsequent comparisons.
                 */
                if (comparison.samePath(currentPath, nextPath) == false && comparison.isParent(currentPath, nextPath) == false) {
                    prunedReadPaths.add(nextPath);
                    currentPath = nextPath;
                }
            }
        }
        return prunedReadPaths;
    }

    /**
     * @brief Factory method to create a {@link FileAccessTree} for a specific component and module.
     *        This method handles the construction of the tree, including the derivation of exclusive paths.
     * @param componentName The name of the component requesting the FileAccessTree.
     * @param moduleName The name of the module requesting the FileAccessTree.
     * @param filesEntitlement The raw file entitlements for this module.
     * @param pathLookup The {@link PathLookup} instance for path resolution.
     * @param componentPaths A {@link Collection} of base paths of the component, used to grant default read access.
     * @param exclusivePaths A global list of all exclusive paths.
     * @return A new {@link FileAccessTree} instance configured for the specified component and module.
     */
    static FileAccessTree of(
        String componentName,
        String moduleName,
        FilesEntitlement filesEntitlement,
        PathLookup pathLookup,
        Collection<Path> componentPaths,
        List<ExclusivePath> exclusivePaths
    ) {
        return new FileAccessTree(
            filesEntitlement,
            pathLookup,
            componentPaths,
            buildUpdatedAndSortedExclusivePaths(componentName, moduleName, exclusivePaths, DEFAULT_COMPARISON),
            DEFAULT_COMPARISON
        );
    }

    /**
     * @brief A special factory method to create a {@link FileAccessTree} instance without any exclusive path restrictions.
     *        This is useful for scenarios like quick validation or when default, unrestricted file access is required.
     * @param filesEntitlement The raw file entitlements for this module.
     * @param pathLookup The {@link PathLookup} instance for path resolution.
     * @param componentPaths A {@link Collection} of base paths of the component, used to grant default read access.
     * @return A new {@link FileAccessTree} instance with no exclusive paths configured.
     */
    public static FileAccessTree withoutExclusivePaths(
        FilesEntitlement filesEntitlement,
        PathLookup pathLookup,
        Collection<Path> componentPaths
    ) {
        return new FileAccessTree(filesEntitlement, pathLookup, componentPaths, new String[0], DEFAULT_COMPARISON);
    }

    /**
     * @brief Checks if the given path is allowed for read access based on the configured entitlements.
     *        It normalizes the path and performs a lookup against the internal read and exclusive path lists.
     * @param path The path to check for read access.
     * @return `true` if read access is granted, `false` otherwise.
     * Algorithm: Path normalization followed by binary search on sorted read and exclusive path arrays.
     * Time Complexity: O(log N) where N is the number of read/exclusive paths.
     */
    public boolean canRead(Path path) {
        var normalizedPath = normalizePath(path);
        var canRead = checkPath(normalizedPath, readPaths);
        logger.trace(() -> Strings.format("checking [%s] (normalized to [%s]) for read: %b", path, normalizedPath, canRead));
        return canRead;
    }

    /**
     * @brief Checks if the given path is allowed for write access based on the configured entitlements.
     *        It normalizes the path and performs a lookup against the internal write and exclusive path lists.
     * @param path The path to check for write access.
     * @return `true` if write access is granted, `false` otherwise.
     * Algorithm: Path normalization followed by binary search on sorted write and exclusive path arrays.
     * Time Complexity: O(log N) where N is the number of write/exclusive paths.
     */
    public boolean canWrite(Path path) {
        var normalizedPath = normalizePath(path);
        var canWrite = checkPath(normalizedPath, writePaths);
        logger.trace(() -> Strings.format("checking [%s] (normalized to [%s]) for write: %b", path, normalizedPath, canWrite));
        return canWrite;
    }

    /**
     * @brief Normalizes a given {@link Path} into a canonical string representation suitable for entitlement checks.
     *        This involves converting to an absolute path, normalizing it, and ensuring a consistent file separator.
     *        It also removes any trailing file separators to maintain consistency.
     * @param path The {@link Path} object to normalize.
     * @return The "canonical" form of the given {@code path} as a string.
     * Algorithm: Path conversion, normalization, and string manipulation.
     * Time Complexity: O(L) where L is the length of the path string.
     */
    static String normalizePath(Path path) {
        // Note that toAbsolutePath produces paths separated by the default file separator,
        // so on Windows, if the given path uses forward slashes, this consistently
        // converts it to backslashes.
        String result = path.toAbsolutePath().normalize().toString();
        /**
         * Block Logic: Removes any trailing file separators from the normalized path string.
         * Invariant: The `result` string will not end with a file separator after this loop.
         */
        while (result.endsWith(FILE_SEPARATOR)) {
            // Inline: Repeatedly truncates the `result` string if it ends with a file separator, ensuring a consistent canonical path format.
            result = result.substring(0, result.length() - FILE_SEPARATOR.length());
        }
        return result;
    }

    /**
     * @brief Performs the core logic for checking if a given `path` is allowed based on a list of granted paths and a list of exclusive paths.
     *        This method uses binary search on sorted arrays to efficiently determine access.
     * @param path The normalized path string to check.
     * @param paths An array of normalized paths that are granted for a specific operation (read or write).
     * @return `true` if the path is granted and not exclusive, `false` otherwise.
     * Algorithm: Binary search on the `exclusivePaths` array first, then on the `paths` array.
     * Time Complexity: O(log N) where N is the length of the `paths` and `exclusivePaths` arrays.
     */
    private boolean checkPath(String path, String[] paths) {
        /**
         * Block Logic: If the granted paths array is empty, no access is possible.
         * Pre-condition: `paths` is a valid string array.
         * Invariant: Returns `false` immediately if there are no granted paths.
         */
        if (paths.length == 0) {
            return false;
        }

        int endx = Arrays.binarySearch(exclusivePaths, path, comparison.pathComparator());
        /**
         * Block Logic: Determines if the `path` is explicitly listed in `exclusivePaths` or is a child of an exclusive path.
         * Pre-condition: `exclusivePaths` is a sorted array. `endx` is the result of a binary search.
         * Invariant: If the path or its parent is exclusive, access is denied.
         * Inline: `-endx - 2` is used to calculate the insertion point index when `path` is not found,
         *         allowing a check for parent directories in the sorted `exclusivePaths` array.
         */
        if (endx < -1 && comparison.isParent(exclusivePaths[-endx - 2], path) || endx >= 0) {
            return false;
        }

        int ndx = Arrays.binarySearch(paths, path, comparison.pathComparator());
        /**
         * Block Logic: Determines if the `path` is explicitly listed in the `paths` array or is a child of a granted path.
         * Pre-condition: `paths` is a sorted array. `ndx` is the result of a binary search.
         * Invariant: If the path or its parent is granted, access is allowed.
         * Inline: `-ndx - 2` is used to calculate the insertion point index when `path` is not found,
         *         allowing a check for parent directories in the sorted `paths` array.
         */
        if (ndx < -1) {
            return comparison.isParent(paths[-ndx - 2], path);
        }
        return ndx >= 0;
    }

    @Override
    /**
     * @brief Compares this {@link FileAccessTree} instance with another object for equality.
     *        Equality is determined by comparing the deep equality of their `readPaths` and `writePaths` arrays.
     * @param o The object to compare with this instance.
     * @return `true` if the objects are equal (have deeply equal read and write paths), `false` otherwise.
     * Algorithm: Object identity check, class type check, then deep equality comparison of internal path arrays.
     * Time Complexity: O(N) where N is the number of elements in the path arrays, due to `Objects.deepEquals`.
     */
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        FileAccessTree that = (FileAccessTree) o;
        return Objects.deepEquals(readPaths, that.readPaths) && Objects.deepEquals(writePaths, that.writePaths);
    }

    @Override
    /**
     * @brief Computes the hash code for this {@link FileAccessTree} instance.
     *        The hash code is derived from the hash codes of the `readPaths` and `writePaths` arrays.
     * @return The hash code for this object.
     * Algorithm: Combines the hash codes of the deeply evaluated read and write path arrays.
     * Time Complexity: O(N) where N is the number of elements in the path arrays, due to `Arrays.hashCode`.
     */
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(readPaths), Arrays.hashCode(writePaths));
    }
}
