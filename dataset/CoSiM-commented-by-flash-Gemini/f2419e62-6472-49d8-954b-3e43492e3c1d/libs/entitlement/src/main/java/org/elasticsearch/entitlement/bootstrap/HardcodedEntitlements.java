/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.bootstrap;

import org.elasticsearch.core.Booleans;
import org.elasticsearch.entitlement.runtime.policy.Policy;
import org.elasticsearch.entitlement.runtime.policy.PolicyUtils;
import org.elasticsearch.entitlement.runtime.policy.Scope;
import org.elasticsearch.entitlement.runtime.policy.entitlements.CreateClassLoaderEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.Entitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.ExitVMEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.InboundNetworkEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.LoadNativeLibrariesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.ManageThreadsEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.OutboundNetworkEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.ReadStoreAttributesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.SetHttpsConnectionPropertiesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.WriteSystemPropertiesEntitlement;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.CONFIG;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.DATA;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.LIB;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.LOGS;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.MODULES;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.PLUGINS;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.SHARED_REPO;
import static org.elasticsearch.entitlement.runtime.policy.Platform.LINUX;
import static org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement.Mode.READ;
import static org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement.Mode.READ_WRITE;

/**
 * @brief Provides a centralized definition of hardcoded entitlements for various Elasticsearch components.
 *
 * This class serves as a repository for default {@link Policy} configurations,
 * including file system access rules, network permissions, and other system-level
 * entitlements required by the Elasticsearch server, its modules, and special agents (like APM).
 * It ensures that essential functionalities operate within predefined security boundaries
 * and prevents unauthorized access to critical resources.
 *
 * Functional Utility: Establishes a baseline of security permissions that are considered
 *                     safe and necessary for the core operation of Elasticsearch, which can
 *                     then be patched or augmented by plugin-specific policies.
 * Architecture: Separates the declaration of core entitlements from their enforcement
 *               mechanism, providing a clear and auditable set of default permissions.
 */
class HardcodedEntitlements {

        /**

         * @brief Creates a list of {@link Scope} objects defining the entitlements for the Elasticsearch server.

         * @param pidFile The path to the PID file, used for specific file entitlements.

         * @return A {@link List} of {@link Scope}s representing the base entitlements for the server process.

         * Functional Utility: Centralizes the definition of all necessary permissions for core Elasticsearch

         *                     functionality, including file system access, class loading, network operations,

         *                     and thread management. This includes platform-specific entitlements (e.g., Linux).

         * Post-condition: A comprehensive list of server-level entitlements is returned, ready to be applied.

         */

        private static List<Scope> createServerEntitlements(Path pidFile) {

    

            List<Scope> serverScopes = new ArrayList<>();

            List<FilesEntitlement.FileData> serverModuleFileDatas = new ArrayList<>();

            /**

             * Block Logic: Populates `serverModuleFileDatas` with specific file access entitlements for various Elasticsearch directories.

             * Functional Utility: Defines the default read and write permissions for critical directories like plugins, modules,

             *                     configuration, logs, data, and shared repositories. This ensures the server can operate

             *                     correctly within its own ecosystem.

             * Invariant: Each `FileData` entry specifies a base directory path and the required access mode (READ or READ_WRITE).

             */

            Collections.addAll(

                serverModuleFileDatas,

                // Base ES directories

                /**

                 * Functional Utility: Grants read access to the plugins directory.

                 * Invariant: Plugins must be readable by the server.

                 */

                FilesEntitlement.FileData.ofBaseDirPath(PLUGINS, READ),

                /**

                 * Functional Utility: Grants read access to the modules directory.

                 * Invariant: Modules must be readable by the server.

                 */

                FilesEntitlement.FileData.ofBaseDirPath(MODULES, READ),

                /**

                 * Functional Utility: Grants read access to the configuration directory.

                 * Invariant: Configuration files must be readable by the server.

                 */

                FilesEntitlement.FileData.ofBaseDirPath(CONFIG, READ),

                /**

                 * Functional Utility: Grants read/write access to the logs directory.

                 * Invariant: The server must be able to write logs.

                 */

                FilesEntitlement.FileData.ofBaseDirPath(LOGS, READ_WRITE),

                /**

                 * Functional Utility: Grants read access to the library directory.

                 * Invariant: Libraries must be readable by the server.

                 */

                FilesEntitlement.FileData.ofBaseDirPath(LIB, READ),

                /**

                 * Functional Utility: Grants read/write access to the data directory.

                 * Invariant: The server must be able to read and write data.

                 */

                FilesEntitlement.FileData.ofBaseDirPath(DATA, READ_WRITE),

                /**

                 * Functional Utility: Grants read/write access to shared repository directories.

                 * Invariant: Shared repositories must be accessible for read and write operations.

                 */

                FilesEntitlement.FileData.ofBaseDirPath(SHARED_REPO, READ_WRITE),

                // exclusive settings file

                /**

                 * Functional Utility: Grants exclusive read/write access to the `operator/settings.json` file within the config directory.

                 * Invariant: This specific file is treated as exclusive to prevent conflicts.

                 */

                FilesEntitlement.FileData.ofRelativePath(Path.of("operator/settings.json"), CONFIG, READ_WRITE).withExclusive(true),

                // OS release on Linux

                /**

                 * Functional Utility: Grants read access to common Linux OS release files.

                 * Invariant: The server may need to read OS release information on Linux platforms.

                 */

                FilesEntitlement.FileData.ofPath(Path.of("/etc/os-release"), READ).withPlatform(LINUX),

                FilesEntitlement.FileData.ofPath(Path.of("/etc/system-release"), READ).withPlatform(LINUX),

                FilesEntitlement.FileData.ofPath(Path.of("/usr/lib/os-release"), READ).withPlatform(LINUX),

                // read max virtual memory areas

                /**

                 * Functional Utility: Grants read access to Linux process memory information.

                 * Invariant: The server may need to query system memory limits.

                 */

                FilesEntitlement.FileData.ofPath(Path.of("/proc/sys/vm/max_map_count"), READ).withPlatform(LINUX),

                FilesEntitlement.FileData.ofPath(Path.of("/proc/meminfo"), READ).withPlatform(LINUX),

                // load averages on Linux

                /**

                 * Functional Utility: Grants read access to Linux load average information.

                 * Invariant: The server may need to monitor system load.

                 */

                FilesEntitlement.FileData.ofPath(Path.of("/proc/loadavg"), READ).withPlatform(LINUX),

                // control group stats on Linux. cgroup v2 stats are in an unpredicable

                // location under `/sys/fs/cgroup`, so unfortunately we have to allow

                // read access to the entire directory hierarchy.

                /**

                 * Functional Utility: Grants read access to Linux cgroup information.

                 * Invariant: The server needs to access control group statistics, including those in potentially

                 *            unpredictable locations under `/sys/fs/cgroup` for cgroup v2.

                 */

                FilesEntitlement.FileData.ofPath(Path.of("/proc/self/cgroup"), READ).withPlatform(LINUX),

                FilesEntitlement.FileData.ofPath(Path.of("/sys/fs/cgroup/"), READ).withPlatform(LINUX),

                // // io stats on Linux

                /**

                 * Functional Utility: Grants read access to Linux I/O statistics and mount information.

                 * Invariant: The server needs to collect I/O related metrics.

                 */

                FilesEntitlement.FileData.ofPath(Path.of("/proc/self/mountinfo"), READ).withPlatform(LINUX),

                FilesEntitlement.FileData.ofPath(Path.of("/proc/diskstats"), READ).withPlatform(LINUX)

            );

            /**

             * Block Logic: Conditionally adds read/write entitlement for the PID file if it is specified.

             * Functional Utility: Ensures that the server process has the necessary permissions to manage

             *                     its PID file, which is critical for process lifecycle management.

             * Pre-condition: `pidFile` is a {@link Path} object (non-null).

             * Invariant: If a PID file is used, the entitlement to access it is included.

             */

            if (pidFile != null) {

                serverModuleFileDatas.add(FilesEntitlement.FileData.ofPath(pidFile, READ_WRITE));

            }

    

            /**

             * Block Logic: Aggregates various entitlements into distinct scopes for different core Elasticsearch modules.

             * Functional Utility: Organizes permissions by logical module boundaries, making the policy

             *                     more modular and easier to understand, manage, and extend.

             * Invariant: Each added scope defines a set of specific entitlements for a particular module or component.

             */

            Collections.addAll(

                serverScopes,

                /**

                 * Functional Utility: Defines entitlements for the `org.elasticsearch.base` module.

                 * Invariant: Grants {@link CreateClassLoaderEntitlement} and file access to data and shared repo directories.

                 */

                new Scope(

                    "org.elasticsearch.base",

                    List.of(

                        new CreateClassLoaderEntitlement(),

                        new FilesEntitlement(

                            List.of(

                                // TODO: what in es.base is accessing shared repo?

                                FilesEntitlement.FileData.ofBaseDirPath(SHARED_REPO, READ_WRITE),

                                FilesEntitlement.FileData.ofBaseDirPath(DATA, READ_WRITE)

                            )

                        )

                    )

                ),

                /**

                 * Functional Utility: Defines entitlements for the `org.elasticsearch.xcontent` module.

                 * Invariant: Grants {@link CreateClassLoaderEntitlement}.

                 */

                new Scope("org.elasticsearch.xcontent", List.of(new CreateClassLoaderEntitlement())),

                /**

                 * Functional Utility: Defines entitlements for the `org.elasticsearch.server` module.

                 * Invariant: Grants a broad set of permissions essential for the server's operation,

                 *            including VM exit, attribute reading, class loading, network, native libraries,

                 *            thread management, and file system access as defined in `serverModuleFileDatas`.

                 */

                new Scope(

                    "org.elasticsearch.server",

                    List.of(

                        new ExitVMEntitlement(),

                        new ReadStoreAttributesEntitlement(),

                        new CreateClassLoaderEntitlement(),

                        new InboundNetworkEntitlement(),

                        new LoadNativeLibrariesEntitlement(),

                        new ManageThreadsEntitlement(),

                        new FilesEntitlement(serverModuleFileDatas)

                    )

                ),

                /**

                 * Functional Utility: Defines entitlements for the `java.desktop` module.

                 * Invariant: Grants {@link LoadNativeLibrariesEntitlement}, likely for UI-related or legacy components.

                 */

                new Scope("java.desktop", List.of(new LoadNativeLibrariesEntitlement())),

                /**

                 * Functional Utility: Defines entitlements for the `org.apache.httpcomponents.httpclient` module.

                 * Invariant: Grants {@link OutboundNetworkEntitlement} for HTTP client operations.

                 */

                new Scope(

                    "org.apache.httpcomponents.httpclient",

                    List.of(new OutboundNetworkEntitlement())

                ),

                /**

                 * Functional Utility: Defines entitlements for the `org.apache.lucene.core` module.

                 * Invariant: Grants permissions for native libraries, thread management, and file access

                 *            to config and data directories, essential for Lucene's indexing and search capabilities.

                 */

                new Scope(

                    "org.apache.lucene.core",

                    List.of(

                        new LoadNativeLibrariesEntitlement(),

                        new ManageThreadsEntitlement(),

                        new FilesEntitlement(

                            List.of(

                                FilesEntitlement.FileData.ofBaseDirPath(CONFIG, READ),

                                FilesEntitlement.FileData.ofBaseDirPath(DATA, READ_WRITE)

                            )

                        )

                    )

                ),

                /**

                 * Functional Utility: Defines entitlements for the `org.apache.lucene.misc` module.

                 * Invariant: Grants file access to data directories and permission to read store attributes.

                 */

                new Scope(

                    "org.apache.lucene.misc",

                    List.of(

                        new FilesEntitlement(List.of(FilesEntitlement.FileData.ofBaseDirPath(DATA, READ_WRITE))),

                        new ReadStoreAttributesEntitlement()

                    )

                ),

                /**

                 * Functional Utility: Defines entitlements for the `org.apache.logging.log4j.core` module.

                 * Invariant: Grants thread management and file access to log directories,

                 *            necessary for logging operations.

                 */

                new Scope(

                    "org.apache.logging.log4j.core",

                    List.of(

                        new ManageThreadsEntitlement(),

                        new FilesEntitlement(List.of(FilesEntitlement.FileData.ofBaseDirPath(LOGS, READ_WRITE)))

                    )

                ),

                /**

                 * Functional Utility: Defines entitlements for the `org.elasticsearch.nativeaccess` module.

                 * Invariant: Grants permissions for loading native libraries and file access to data directories,

                 *            critical for native code integration.

                 */

                new Scope(

                    "org.elasticsearch.nativeaccess",

                    List.of(

                        new LoadNativeLibrariesEntitlement(),

                        new FilesEntitlement(List.of(FilesEntitlement.FileData.ofBaseDirPath(DATA, READ_WRITE)))

                    )

                )

            );

    

            // conditionally add FIPS entitlements if FIPS only functionality is enforced

            /**

             * Block Logic: Conditionally adds FIPS-related entitlements if FIPS-only functionality is enforced.

             * Functional Utility: Ensures that when the system is operating in FIPS-approved mode,

             *                     the necessary permissions for Bouncy Castle FIPS modules (TLS, Core)

             *                     are granted, including access to trust stores and network.

             * Pre-condition: The system property "org.bouncycastle.fips.approved_only" is set to "true".

             * Invariant: FIPS-specific scopes are added to `serverScopes` only if FIPS mode is enabled.

             */

            if (Booleans.parseBoolean(System.getProperty("org.bouncycastle.fips.approved_only"), false)) {

                // if custom trust store is set, grant read access to its location, otherwise use the default JDK trust store

                String trustStore = System.getProperty("javax.net.ssl.trustStore");

                Path trustStorePath = trustStore != null

                    ? Path.of(trustStore)

                    : Path.of(System.getProperty("java.home")).resolve("lib/security/jssecacerts");

    

                Collections.addAll(

                    serverScopes,

                    /**

                     * Functional Utility: Defines entitlements for the `org.bouncycastle.fips.tls` module under FIPS mode.

                     * Invariant: Grants file read access to the trust store, thread management, and outbound network access.

                     */

                    new Scope(

                        "org.bouncycastle.fips.tls",

                        List.of(

                            new FilesEntitlement(List.of(FilesEntitlement.FileData.ofPath(trustStorePath, READ))),

                            new ManageThreadsEntitlement(),

                            new OutboundNetworkEntitlement()

                        )

                    ),

                    /**

                     * Functional Utility: Defines entitlements for the `org.bouncycastle.fips.core` module under FIPS mode.

                     * Invariant: Grants file read access to the library directory for checksum validation and thread management.

                     */

                    new Scope(

                        "org.bouncycastle.fips.core",

                        // read to lib dir is required for checksum validation

                        List.of(

                            new FilesEntitlement(List.of(FilesEntitlement.FileData.ofBaseDirPath(LIB, READ))),

                            new ManageThreadsEntitlement()

                        )

                    )

                );

            }

            return serverScopes;

        }

    /**
     * @brief Creates the final server {@link Policy} by potentially merging default entitlements with a patch policy.
     * @param pidFile The path to the PID file, used by {@link #createServerEntitlements(Path)}.
     * @param serverPolicyPatch An optional {@link Policy} containing additional scopes to merge into the default server policy.
     * @return A consolidated {@link Policy} for the server.
     * Functional Utility: Provides the complete set of entitlements that will govern the server's operations,
     *                     allowing for customization or extension of the hardcoded defaults.
     * Post-condition: A {@link Policy} instance is returned that combines base server entitlements with any provided patch.
     */
    static Policy serverPolicy(Path pidFile, Policy serverPolicyPatch) {
        var serverScopes = createServerEntitlements(pidFile);
        return new Policy(
            "server",
            serverPolicyPatch == null ? serverScopes : PolicyUtils.mergeScopes(serverScopes, serverPolicyPatch.scopes())
        );
    }

    // agents run without a module, so this is a special hack for the apm agent
    // this should be removed once https://github.com/elastic/elasticsearch/issues/109335 is completed
    // See also modules/apm/src/main/plugin-metadata/entitlement-policy.yaml
    /**
     * @brief Defines hardcoded entitlements specifically for agents (e.g., APM agent) that run without a dedicated module.
     * @return A {@link List} of {@link Entitlement} objects for agent processes.
     * Functional Utility: Provides the necessary permissions for an agent to operate within the JVM,
     *                     including class loading, thread management, network access, system property
     *                     writes, native library loading, and limited file system access (e.g., logs, procfs).
     * Rationale: Agents often operate outside the standard module system and require explicit, carefully
     *            defined permissions to perform their monitoring or instrumentation tasks without
     *            compromising the security of the main application. This also addresses known
     *            limitations with dynamic module access.
     * Invariant: This set of entitlements is a temporary measure and is expected to be integrated
     *            into a more robust module-based entitlement system in future versions.
     */
    static List<Entitlement> agentEntitlements() {
        return List.of(
            new CreateClassLoaderEntitlement(),
            new ManageThreadsEntitlement(),
            new SetHttpsConnectionPropertiesEntitlement(),
            new OutboundNetworkEntitlement(),
            new WriteSystemPropertiesEntitlement(Set.of("AsyncProfiler.safemode")),
            new LoadNativeLibrariesEntitlement(),
            new FilesEntitlement(
                List.of(
                    FilesEntitlement.FileData.ofBaseDirPath(LOGS, READ_WRITE),
                    FilesEntitlement.FileData.ofPath(Path.of("/proc/meminfo"), READ),
                    FilesEntitlement.FileData.ofPath(Path.of("/sys/fs/cgroup/"), READ)
                )
            )
        );
    }
}
