/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

/**
 * @file RemoteClusterService.java
 * @brief Provides core functionality for managing and interacting with remote Elasticsearch clusters.
 * This service handles remote cluster connections, credential management, index grouping, and request routing
 * across multiple clusters, supporting both single and multi-project environments.
 * It ensures robust error handling and proper resource management for inter-cluster communication.
 */
package org.elasticsearch.transport;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.Build;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.OriginalIndices;
import org.elasticsearch.action.support.CountDownActionListener;
import org.elasticsearch.action.support.IndicesOptions;
import org.elasticsearch.action.support.PlainActionFuture;
import org.elasticsearch.action.support.RefCountingListener;
import org.elasticsearch.action.support.RefCountingRunnable;
import org.elasticsearch.client.internal.RemoteClusterClient;
import org.elasticsearch.cluster.metadata.IndexNameExpressionResolver;
import org.elasticsearch.cluster.metadata.ProjectId;
import org.elasticsearch.cluster.node.DiscoveryNode;
import org.elasticsearch.cluster.node.DiscoveryNodeRole;
import org.elasticsearch.cluster.project.DefaultProjectResolver;
import org.elasticsearch.cluster.project.ProjectResolver;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.settings.ClusterSettings;
import org.elasticsearch.common.settings.SecureSetting;
import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.util.concurrent.ConcurrentCollections;
import org.elasticsearch.common.util.concurrent.EsExecutors;
import org.elasticsearch.core.FixForMultiProject;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.indices.IndicesExpressionGrouper;
import org.elasticsearch.node.ReportingService;
import org.elasticsearch.transport.RemoteClusterCredentialsManager.UpdateRemoteClusterCredentialsResult;

import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.elasticsearch.common.settings.Setting.boolSetting;
import static org.elasticsearch.common.settings.Setting.enumSetting;
import static org.elasticsearch.common.settings.Setting.timeSetting;
import static org.elasticsearch.transport.RemoteClusterPortSettings.REMOTE_CLUSTER_SERVER_ENABLED;

/**
 * @brief Basic service for accessing remote clusters via gateway nodes.
 * This class provides functionality for managing connections to remote Elasticsearch clusters,
 * handling remote requests, and grouping indices across clusters. It supports features like
 * connection timeouts, node attribute filtering, and secure credential management for inter-cluster communication.
 * It extends {@link RemoteClusterAware} for common remote cluster functionalities,
 * implements {@link Closeable} for resource management,
 * {@link ReportingService} for cluster information, and
 * {@link IndicesExpressionGrouper} for index grouping logic.
 */
public final class RemoteClusterService extends RemoteClusterAware
    implements
        Closeable,
        ReportingService<RemoteClusterServerInfo>,
        IndicesExpressionGrouper {

    private static final Logger logger = LogManager.getLogger(RemoteClusterService.class);

    /**
     * @var REMOTE_INITIAL_CONNECTION_TIMEOUT_SETTING
     * @brief The initial connection timeout for remote cluster connections.
     * This setting defines how long to wait when establishing the first connection to a remote cluster.
     */
    public static final Setting<TimeValue> REMOTE_INITIAL_CONNECTION_TIMEOUT_SETTING = Setting.positiveTimeSetting(
        "cluster.remote.initial_connect_timeout",
        TimeValue.timeValueSeconds(30),
        Setting.Property.NodeScope
    );

    /**
     * @var REMOTE_NODE_ATTRIBUTE
     * @brief Node attribute used to select gateway nodes in remote clusters.
     * This setting allows specifying a node attribute name (e.g., "gateway") whose value
     * is expected to be boolean. Nodes with {@code node.attr.gateway: true} would be eligible
     * as gateway nodes for remote cluster connections.
     */
    public static final Setting<String> REMOTE_NODE_ATTRIBUTE = Setting.simpleString(
        "cluster.remote.node.attr",
        Setting.Property.NodeScope
    );

    /**
     * @var REMOTE_CLUSTER_SKIP_UNAVAILABLE
     * @brief Affix setting to control whether to skip unavailable remote clusters.
     * If set to {@code true}, requests to a disconnected remote cluster will fail immediately
     * without attempting to re-establish the connection.
     */
    public static final Setting.AffixSetting<Boolean> REMOTE_CLUSTER_SKIP_UNAVAILABLE = Setting.affixKeySetting(
        "cluster.remote.",
        "skip_unavailable",
        (ns, key) -> boolSetting(key, true, new RemoteConnectionEnabled<>(ns, key), Setting.Property.Dynamic, Setting.Property.NodeScope)
    );

    /**
     * @var REMOTE_CLUSTER_PING_SCHEDULE
     * @brief Affix setting for the transport ping schedule to remote clusters.
     * Defines the interval at which ping requests are sent to remote clusters to maintain connection.
     */
    public static final Setting.AffixSetting<TimeValue> REMOTE_CLUSTER_PING_SCHEDULE = Setting.affixKeySetting(
        "cluster.remote.",
        "transport.ping_schedule",
        (ns, key) -> timeSetting(
            key,
            TransportSettings.PING_SCHEDULE,
            new RemoteConnectionEnabled<>(ns, key),
            Setting.Property.Dynamic,
            Setting.Property.NodeScope
        )
    );

    /**
     * @var REMOTE_CLUSTER_COMPRESS
     * @brief Affix setting to enable or disable transport compression for remote cluster connections.
     * This setting controls whether data transmitted to remote clusters should be compressed.
     */
    public static final Setting.AffixSetting<Compression.Enabled> REMOTE_CLUSTER_COMPRESS = Setting.affixKeySetting(
        "cluster.remote.",
        "transport.compress",
        (ns, key) -> enumSetting(
            Compression.Enabled.class,
            key,
            TransportSettings.TRANSPORT_COMPRESS,
            new RemoteConnectionEnabled<>(ns, key),
            Setting.Property.Dynamic,
            Setting.Property.NodeScope
        )
    );

    /**
     * @var REMOTE_CLUSTER_COMPRESSION_SCHEME
     * @brief Affix setting to define the compression scheme used for remote cluster transport.
     * Specifies the algorithm used for compressing data if compression is enabled.
     */
    public static final Setting.AffixSetting<Compression.Scheme> REMOTE_CLUSTER_COMPRESSION_SCHEME = Setting.affixKeySetting(
        "cluster.remote.",
        "transport.compression_scheme",
        (ns, key) -> enumSetting(
            Compression.Scheme.class,
            key,
            TransportSettings.TRANSPORT_COMPRESSION_SCHEME,
            new RemoteConnectionEnabled<>(ns, key),
            Setting.Property.Dynamic,
            Setting.Property.NodeScope
        )
    );

    /**
     * @var REMOTE_CLUSTER_CREDENTIALS
     * @brief Affix setting for secure credentials used to connect to remote clusters.
     * Stores sensitive information like usernames and passwords for authentication with remote clusters.
     */
    public static final Setting.AffixSetting<SecureString> REMOTE_CLUSTER_CREDENTIALS = Setting.affixKeySetting(
        "cluster.remote.",
        "credentials",
        key -> SecureSetting.secureString(key, null)
    );

    /**
     * @var REMOTE_CLUSTER_HANDSHAKE_ACTION_NAME
     * @brief The action name for remote cluster handshake requests.
     * Used for internal communication to establish and verify connections between remote clusters.
     */
    public static final String REMOTE_CLUSTER_HANDSHAKE_ACTION_NAME = "cluster:internal/remote_cluster/handshake";

    private final boolean enabled; /**< Flag indicating if remote cluster client functionality is enabled on this node. */
    private final boolean remoteClusterServerEnabled; /**< Flag indicating if the remote cluster server is enabled. */

    /**
     * @brief Checks if remote cluster client functionality is enabled.
     * @return True if enabled, false otherwise.
     */
    public boolean isEnabled() {
        return enabled;
    }

    /**
     * @brief Checks if the remote cluster server is enabled.
     * @return True if enabled, false otherwise.
     */
    public boolean isRemoteClusterServerEnabled() {
        return remoteClusterServerEnabled;
    }

    private final TransportService transportService; /**< The TransportService instance for handling network communications. */
    private final Map<ProjectId, Map<String, RemoteClusterConnection>> remoteClusters; /**< Map of remote cluster connections, keyed by ProjectId and cluster alias. */
    private final RemoteClusterCredentialsManager remoteClusterCredentialsManager; /**< Manager for remote cluster credentials. */
    private final ProjectResolver projectResolver; /**< Resolver for project-related information. */

    /**
     * @brief Constructs a new RemoteClusterService.
     * Initializes the service with given settings and transport service, sets up remote cluster
     * connection maps, and registers handshake request handlers if the remote cluster server is enabled.
     * @param settings The node settings.
     * @param transportService The TransportService instance.
     * @FixForMultiProject(description = "Inject the ProjectResolver instance.")
     */
    RemoteClusterService(Settings settings, TransportService transportService) {
        super(settings);
        this.enabled = DiscoveryNode.isRemoteClusterClient(settings);
        this.remoteClusterServerEnabled = REMOTE_CLUSTER_SERVER_ENABLED.get(settings);
        this.transportService = transportService;
        this.projectResolver = DefaultProjectResolver.INSTANCE;
        this.remoteClusters = projectResolver.supportsMultipleProjects()
            ? ConcurrentCollections.newConcurrentMap()
            : Map.of(ProjectId.DEFAULT, ConcurrentCollections.newConcurrentMap());
        this.remoteClusterCredentialsManager = new RemoteClusterCredentialsManager(settings);
        if (remoteClusterServerEnabled) {
            registerRemoteClusterHandshakeRequestHandler(transportService);
        }
    }

    public DiscoveryNode getLocalNode() {
        return transportService.getLocalNode();
    }

    /**
     * Group indices by cluster alias mapped to OriginalIndices for that cluster.
     * @param remoteClusterNames Set of configured remote cluster names.
     * @param indicesOptions IndicesOptions to clarify how the index expressions should be parsed/applied
     * @param indices Multiple index expressions as string[].
     * @param returnLocalAll whether to support the _all functionality needed by _search
     *        (See https://github.com/elastic/elasticsearch/pull/33899). If true, and no indices are specified,
     *        then a Map with one entry for the local cluster with an empty index array is returned.
     *        If false, an empty map is returned when no indices are specified.
     * @return Map keyed by cluster alias having OriginalIndices as the map value parsed from the String[] indices argument
     */
    /**
     * @brief Group indices by cluster alias mapped to OriginalIndices for that cluster.
     * @param remoteClusterNames Set of configured remote cluster names.
     * @param indicesOptions IndicesOptions to clarify how the index expressions should be parsed/applied
     * @param indices Multiple index expressions as string[].
     * @param returnLocalAll whether to support the _all functionality needed by _search
     *        (See https://github.com/elastic/elasticsearch/pull/33899). If true, and no indices are specified,
     *        then a Map with one entry for the local cluster with an empty index array is returned.
     *        If false, an empty map is returned when no indices are specified.
     * @return Map keyed by cluster alias having OriginalIndices as the map value parsed from the String[] indices argument
     *
     * Functional Utility: This method acts as a central dispatcher for grouping indices based on remote cluster configurations
     *                     and specified options. It handles the logic for determining whether to include local indices
     *                     or fallback to an empty set.
     * Pre-condition: `remoteClusterNames` should be a valid set of cluster aliases; `indicesOptions` and `indices` should
     *                be properly initialized.
     * Post-condition: Returns a map where keys are cluster aliases (including a special key for the local cluster)
     *                 and values are `OriginalIndices` objects representing the indices relevant to that cluster.
     */
        final Map<String, OriginalIndices> originalIndicesMap = new HashMap<>();
        final Map<String, List<String>> groupedIndices;
        /*
         * returnLocalAll is used to control whether we'd like to fallback to the local cluster.
         * While this is acceptable in a few cases, there are cases where we should not fallback to the local
         * cluster. Consider _resolve/cluster where the specified patterns do not match any remote clusters.
         * Falling back to the local cluster and returning its details in such cases is not ok. This is why
         * TransportResolveClusterAction sets returnLocalAll to false wherever it uses groupIndices().
         *
         * If such a fallback isn't allowed and the given indices match a pattern whose semantics mean that
         * it's ok to return an empty result (denoted via ["*", "-*"]), empty groupIndices.
         */
        // Block Logic: Determines whether to proceed with index grouping or to return an empty map based on `returnLocalAll`
        //              and if the indices expression signifies 'none'. This prevents unwanted local fallbacks
        //              when specific remote-only behaviors are intended.
        if (returnLocalAll == false && IndexNameExpressionResolver.isNoneExpression(indices)) {
            groupedIndices = Map.of();
        } else {
            groupedIndices = groupClusterIndices(remoteClusterNames, indices);
        }

        // Block Logic: Populates the `originalIndicesMap` based on whether `groupedIndices` is empty.
        //              If empty and `returnLocalAll` is true, it defaults to the local cluster's all indices.
        // Invariant: `originalIndicesMap` will contain `OriginalIndices` for either the local cluster or the
        //            grouped remote clusters, depending on the input parameters and grouping result.
        if (groupedIndices.isEmpty()) {
            if (returnLocalAll) {
                // Inline: Assigns an empty array for the local cluster if no indices are specified and local fallback is enabled.
                originalIndicesMap.put(LOCAL_CLUSTER_GROUP_KEY, new OriginalIndices(Strings.EMPTY_ARRAY, indicesOptions));
            }
        } else {
            // Block Logic: Iterates through the grouped indices and converts them into `OriginalIndices` objects
            //              for each cluster alias, populating the `originalIndicesMap`.
            for (Map.Entry<String, List<String>> entry : groupedIndices.entrySet()) {
                String clusterAlias = entry.getKey();
                List<String> originalIndices = entry.getValue();
                originalIndicesMap.put(clusterAlias, new OriginalIndices(originalIndices.toArray(new String[0]), indicesOptions));
            }
        }
        return originalIndicesMap;
    }

    /**
     * If no indices are specified, then a Map with one entry for the local cluster with an empty index array is returned.
     * For details see {@code groupIndices(IndicesOptions indicesOptions, String[] indices, boolean returnLocalAll)}
     * @param remoteClusterNames Set of configured remote cluster names.
     * @param indicesOptions IndicesOptions to clarify how the index expressions should be parsed/applied
     * @param indices Multiple index expressions as string[].
     * @return Map keyed by cluster alias having OriginalIndices as the map value parsed from the String[] indices argument
     */
    /**
     * @brief Group indices by cluster alias, defaulting to local cluster fallback if no indices are specified.
     * If no indices are specified, then a Map with one entry for the local cluster with an empty index array is returned.
     * For details see {@link #groupIndices(Set, IndicesOptions, String[], boolean)}
     * @param remoteClusterNames Set of configured remote cluster names.
     * @param indicesOptions IndicesOptions to clarify how the index expressions should be parsed/applied
     * @param indices Multiple index expressions as string[].
     * @return Map keyed by cluster alias having OriginalIndices as the map value parsed from the String[] indices argument
     *
     * Functional Utility: This is a convenience overload of the primary `groupIndices` method,
     *                     providing a simplified interface when the default `returnLocalAll` behavior (`true`) is desired.
     * Pre-condition: `remoteClusterNames` should be a valid set of cluster aliases; `indicesOptions` and `indices` should
     *                be properly initialized.
     * Post-condition: Returns a map similar to the main `groupIndices` method, with local fallback implicitly enabled.
     */
        return groupIndices(remoteClusterNames, indicesOptions, indices, true);
    }

    /**
     * @brief Group indices by cluster alias, retrieving registered remote cluster names automatically.
     * @param indicesOptions IndicesOptions to clarify how the index expressions should be parsed/applied
     * @param indices Multiple index expressions as string[].
     * @param returnLocalAll whether to support the _all functionality needed by _search.
     *        (See {@link #groupIndices(Set, IndicesOptions, String[], boolean)} for details).
     * @return Map keyed by cluster alias having OriginalIndices as the map value parsed from the String[] indices argument
     *
     * Functional Utility: Simplifies calling the primary `groupIndices` method by automatically
     *                     fetching the set of currently registered remote cluster names.
     * Pre-condition: `indicesOptions` and `indices` should be properly initialized. Remote clusters must be configured.
     * Post-condition: Returns a map similar to the main `groupIndices` method, with the remote cluster
     *                 names dynamically retrieved.
     */
        return groupIndices(getRegisteredRemoteClusterNames(), indicesOptions, indices, returnLocalAll);
    }

    /**
     * @brief Group indices by cluster alias, automatically retrieving registered remote cluster names and allowing local fallback.
     * This is the simplest overload, implicitly setting {@code returnLocalAll} to {@code true}.
     * @param indicesOptions IndicesOptions to clarify how the index expressions should be parsed/applied
     * @param indices Multiple index expressions as string[].
     * @return Map keyed by cluster alias having OriginalIndices as the map value parsed from the String[] indices argument
     *
     * Functional Utility: Provides the most streamlined way to group indices, suitable for common use cases
     *                     where default behavior (auto-retrieving cluster names and allowing local fallback) is acceptable.
     * Pre-condition: `indicesOptions` and `indices` should be properly initialized. Remote clusters must be configured.
     * Post-condition: Returns a map similar to the main `groupIndices` method.
     */
        return groupIndices(getRegisteredRemoteClusterNames(), indicesOptions, indices, true);
    }

    @Override
    /**
     * @brief Retrieves the names of all remote clusters that are currently configured.
     * Functional Utility: Provides a consolidated list of remote clusters available for interaction.
     * Post-condition: Returns a {@link Set} of strings, each representing a configured remote cluster's alias.
     */
    public Set<String> getConfiguredClusters() {
        return getRegisteredRemoteClusterNames();
    }

    /**
     * Returns the registered remote cluster names.
     */
    @FixForMultiProject(description = "Analyze use cases, determine possible need for cluster scoped and project scoped versions.")
    /**
     * @brief Returns the names of all currently registered remote clusters.
     * Functional Utility: Provides a direct way to query the aliases of all remote clusters that have been configured.
     * @return A {@link Set} of strings, where each string is the alias of a registered remote cluster.
     */
    @FixForMultiProject(description = "Analyze use cases, determine possible need for cluster scoped and project scoped versions.")
    public Set<String> getRegisteredRemoteClusterNames() {
        return getConnectionsMapForCurrentProject().keySet();
    }

    /**
     * @brief Returns a connection to the given node on the given remote cluster.
     * @param node The {@link DiscoveryNode} representing the target node within the remote cluster.
     * @param cluster The alias of the remote cluster.
     * @return A {@link Transport.Connection} to the specified node in the remote cluster.
     * @throws IllegalArgumentException if the remote cluster is unknown.
     *
     * Functional Utility: Provides direct access to a transport connection for a specific node in a remote cluster,
     *                     enabling targeted communication within the inter-cluster network.
     * Pre-condition: The `cluster` alias must be known and configured. The `node` must be a valid node within that cluster.
     */
    public Transport.Connection getConnection(DiscoveryNode node, String cluster) {
        return getRemoteClusterConnection(cluster).getConnection(node);
    }

    /**
     * Ensures that the given cluster alias is connected. If the cluster is connected this operation
     * will invoke the listener immediately.
     */
    /**
     * @brief Ensures that the given cluster alias is connected. If the cluster is already connected, this operation
     *        will invoke the listener immediately. Otherwise, it will attempt to establish a connection.
     * @param clusterAlias The alias of the remote cluster to connect to.
     * @param listener The {@link ActionListener} to be notified upon connection success or failure.
     *
     * Functional Utility: Provides an asynchronous mechanism to guarantee connectivity to a remote cluster
     *                     before subsequent operations, handling existing connections efficiently.
     * Pre-condition: `clusterAlias` must refer to a configured remote cluster.
     * Post-condition: The listener will be invoked, indicating either a successful connection or an error.
     */
    void ensureConnected(String clusterAlias, ActionListener<Void> listener) {
        final RemoteClusterConnection remoteClusterConnection;
        // Block Logic: Attempts to retrieve the remote cluster connection. If the cluster alias is unknown,
        //              it catches the NoSuchRemoteClusterException and notifies the listener of the failure.
        // Pre-condition: `clusterAlias` is provided.
        // Post-condition: `remoteClusterConnection` is initialized, or `listener` is failed with an exception.
        try {
            remoteClusterConnection = getRemoteClusterConnection(clusterAlias);
        } catch (NoSuchRemoteClusterException e) {
            listener.onFailure(e);
            return;
        }
        remoteClusterConnection.ensureConnected(listener);
    }

    /**
     * Returns whether the cluster identified by the provided alias is configured to be skipped when unavailable
     */
    /**
     * @brief Returns whether the cluster identified by the provided alias is configured to be skipped when unavailable.
     * @param clusterAlias The alias of the remote cluster to check.
     * @return `true` if the remote cluster is configured to be skipped when unavailable, `false` otherwise.
     *
     * Functional Utility: Allows callers to determine the behavior of requests to a remote cluster when it is disconnected.
     * Pre-condition: `clusterAlias` must refer to a configured remote cluster.
     * Invariant: The returned value reflects the current dynamic setting for `REMOTE_CLUSTER_SKIP_UNAVAILABLE` for the given alias.
     */
    public boolean isSkipUnavailable(String clusterAlias) {
        return getRemoteClusterConnection(clusterAlias).isSkipUnavailable();
    }

    /**
     * @brief Returns an arbitrary active connection to the specified remote cluster.
     * This method does not attempt to re-establish a connection if none is available.
     * @param cluster The alias of the remote cluster.
     * @return A {@link Transport.Connection} to the remote cluster.
     * @throws NoSuchRemoteClusterException if the remote cluster is not configured.
     * @throws ConnectTransportException if no active connection is available for the remote cluster.
     *
     * Functional Utility: Provides quick access to an established connection for sending requests to a remote cluster
     *                     when immediate availability is assumed or previously checked.
     * Pre-condition: The `cluster` alias must be known and a connection to it must be active.
     */
    public Transport.Connection getConnection(String cluster) {
        return getRemoteClusterConnection(cluster).getConnection();
    }

    /**
     * Unlike {@link #getConnection(String)} this method might attempt to re-establish a remote connection if there is no connection
     * available before returning a connection to the remote cluster.
     *
     * @param clusterAlias    the remote cluster
     * @param ensureConnected whether requests should wait for a connection attempt when there isn't available connection
     * @param listener        a listener that will be notified the connection or failure
     */
    /**
     * @brief Conditionally ensures connectivity to a remote cluster and retrieves a connection.
     * Unlike {@link #getConnection(String)}, this method might attempt to re-establish a remote connection if there is no connection
     * available before returning a connection to the remote cluster.
     * @param clusterAlias The alias of the remote cluster.
     * @param ensureConnected If `true`, the method will wait for a connection attempt to complete if no connection is available.
     *                        If `false`, it will fail immediately if no connection is available, but trigger a background reconnect.
     * @param listener An {@link ActionListener} that will be notified with the {@link Transport.Connection} or a failure.
     *
     * Functional Utility: Provides flexible control over connection establishment behavior, allowing for either
     *                     blocking waits for connectivity or immediate failure with background reconnection.
     * Pre-condition: `clusterAlias` must refer to a configured remote cluster.
     * Post-condition: The listener is invoked with a connection if successful, or an exception if unsuccessful.
     *                 If `ensureConnected` is `false` and a connection is not immediately available, a reconnect
     *                 attempt is made in the background.
     */
    public void maybeEnsureConnectedAndGetConnection(
        String clusterAlias,
        boolean ensureConnected,
        ActionListener<Transport.Connection> listener
    ) {
        // Block Logic: Creates a wrapper listener that handles the delegation of failure and the retrieval of the connection.
        //              If `ensureConnected` is false and a connection is not immediately available, it triggers a
        //              background reconnection attempt without waiting.
        // Functional Utility: Centralizes the logic for handling connection attempts and potential failures,
        //                     providing a consistent response mechanism to the original listener.
        ActionListener<Void> ensureConnectedListener = listener.delegateFailureAndWrap(
            (l, nullValue) -> ActionListener.completeWith(l, () -> {
                try {
                    return getConnection(clusterAlias);
                } catch (ConnectTransportException e) {
                    // Block Logic: If `ensureConnected` is false, a connection failure here triggers a background
                    //              reconnection attempt without blocking the current request.
                    if (ensureConnected == false) {
                        // trigger another connection attempt, but don't wait for it to complete
                        ensureConnected(clusterAlias, ActionListener.noop());
                    }
                    throw e;
                }
            })
        );
        // Block Logic: Determines whether to initiate an active connection attempt or to immediately
        //              proceed with the connection retrieval (which might fail if no connection exists).
        // Pre-condition: `ensureConnectedListener` is ready to receive a response.
        if (ensureConnected) {
            ensureConnected(clusterAlias, ensureConnectedListener);
        } else {
            ensureConnectedListener.onResponse(null);
        }
    }

    /**
     * @brief Retrieves the {@link RemoteClusterConnection} object for a specified remote cluster alias.
     * @param cluster The alias of the remote cluster.
     * @return The {@link RemoteClusterConnection} instance associated with the given alias.
     * @throws IllegalArgumentException if remote cluster client functionality is not enabled on this node.
     * @throws NoSuchRemoteClusterException if the specified cluster alias is not found among the configured remote clusters.
     *
     * Functional Utility: Serves as a gateway to access the connection management details and status of a specific remote cluster.
     * Pre-condition: `cluster` must be a non-null, non-empty string representing a remote cluster alias.
     */
    public RemoteClusterConnection getRemoteClusterConnection(String cluster) {
        if (enabled == false) {
            throw new IllegalArgumentException(
                "this node does not have the " + DiscoveryNodeRole.REMOTE_CLUSTER_CLIENT_ROLE.roleName() + " role"
            );
        }
        @FixForMultiProject(description = "Verify all callers will have the proper context set for resolving the origin project ID.")
        RemoteClusterConnection connection = getConnectionsMapForCurrentProject().get(cluster);
        if (connection == null) {
            throw new NoSuchRemoteClusterException(cluster);
        }
        return connection;
    }

    @Override
    /**
     * @brief Registers consumers for dynamic cluster setting updates.
     * Functional Utility: Enables the {@link RemoteClusterService} to react to changes in cluster settings
     *                     related to remote connections, such as `REMOTE_CLUSTER_SKIP_UNAVAILABLE`.
     * @param clusterSettings The {@link ClusterSettings} instance to register update consumers with.
     * Pre-condition: `clusterSettings` must be initialized.
     */
    public void listenForUpdates(ClusterSettings clusterSettings) {
        super.listenForUpdates(clusterSettings);
        clusterSettings.addAffixUpdateConsumer(REMOTE_CLUSTER_SKIP_UNAVAILABLE, this::updateSkipUnavailable, (alias, value) -> {});
    }

    /**
     * @brief Dynamically updates the 'skip unavailable' setting for a specific remote cluster.
     * This method is a consumer for changes to the {@link #REMOTE_CLUSTER_SKIP_UNAVAILABLE} setting.
     * @param clusterAlias The alias of the remote cluster whose setting is being updated.
     * @param skipUnavailable The new boolean value for the 'skip unavailable' setting.
     *
     * Functional Utility: Allows the service to adapt its behavior regarding disconnected remote clusters
     *                     in real-time, based on cluster setting updates.
     * Pre-condition: The method is invoked in response to a change in the `REMOTE_CLUSTER_SKIP_UNAVAILABLE` setting
     *                for the given `clusterAlias`.
     */
    private synchronized void updateSkipUnavailable(String clusterAlias, Boolean skipUnavailable) {
        RemoteClusterConnection remote = getConnectionsMapForCurrentProject().get(clusterAlias);
        if (remote != null) {
            remote.setSkipUnavailable(skipUnavailable);
        }
    }

    /**
     * @brief Updates the secure credentials for remote clusters and rebuilds connections where necessary.
     * This method is triggered when secure settings related to remote cluster credentials are updated.
     * It rebuilds connections only for clusters whose credentials were newly added or removed, not just modified.
     * @param settingsSupplier A {@link Supplier} that provides the updated {@link Settings}.
     * @param listener An {@link ActionListener} to be notified upon completion of the credential update process.
     *
     * Functional Utility: Ensures that remote cluster connections operate with the most up-to-date
     *                     authentication details, enhancing security and maintaining connectivity.
     * Pre-condition: `settingsSupplier` provides valid settings, and `listener` is prepared to handle completion.
     * Post-condition: Connections to affected remote clusters are rebuilt, and the listener is informed.
     */
    @FixForMultiProject(description = "Refactor as needed to support project specific changes to linked remotes.")
    public synchronized void updateRemoteClusterCredentials(Supplier<Settings> settingsSupplier, ActionListener<Void> listener) {
        final var projectId = projectResolver.getProjectId();
        final Settings settings = settingsSupplier.get();
        final UpdateRemoteClusterCredentialsResult result = remoteClusterCredentialsManager.updateClusterCredentials(settings);
        // We only need to rebuild connections when a credential was newly added or removed for a cluster alias, not if the credential
        // value was updated. Therefore, only consider added or removed aliases
        final int totalConnectionsToRebuild = result.addedClusterAliases().size() + result.removedClusterAliases().size();
        // Block Logic: Checks if any credentials were added or removed. If not, no connection rebuilding is necessary,
        //              and the listener is immediately notified.
        // Pre-condition: `result` contains the outcome of the credential update.
        if (totalConnectionsToRebuild == 0) {
            logger.debug("project [{}] no connection rebuilding required after credentials update", projectId);
            listener.onResponse(null);
            return;
        }
        logger.info("project [{}] rebuilding [{}] connections after credentials update", projectId, totalConnectionsToRebuild);
        // Block Logic: Uses a RefCountingRunnable to manage the asynchronous rebuilding of multiple connections.
        //              This ensures the listener is only called after all necessary connection rebuilds have
        //              either completed or failed.
        // Invariant: The `listener` will be invoked exactly once after all relevant `maybeRebuildConnectionOnCredentialsChange`
        //            calls have completed.
        try (var connectionRefs = new RefCountingRunnable(() -> listener.onResponse(null))) {
            // Block Logic: Iterates through each cluster alias for which new credentials were added,
            //              triggering a potential connection rebuild for each.
            for (var clusterAlias : result.addedClusterAliases()) {
                maybeRebuildConnectionOnCredentialsChange(projectId, clusterAlias, settings, connectionRefs);
            }
            // Block Logic: Iterates through each cluster alias for which credentials were removed,
            //              triggering a potential connection rebuild for each.
            for (var clusterAlias : result.removedClusterAliases()) {
                maybeRebuildConnectionOnCredentialsChange(projectId, clusterAlias, settings, connectionRefs);
            }
        }
    }

    /**
     * @brief Conditionally rebuilds a remote cluster connection after credentials have changed.
     * This method checks if a connection for the given `clusterAlias` already exists. If not,
     * no rebuild is necessary. If it exists, it triggers an update to the remote cluster connection.
     * @param projectId The {@link ProjectId} associated with the remote cluster.
     * @param clusterAlias The alias of the remote cluster.
     * @param settings The updated {@link Settings} containing the new credential information.
     * @param connectionRefs A {@link RefCountingRunnable} to track the completion of connection rebuilds.
     *
     * Functional Utility: Manages the lifecycle of remote connections in response to credential updates,
     *                     ensuring that existing connections are properly refreshed or re-established.
     * Pre-condition: `projectId`, `clusterAlias`, `settings`, and `connectionRefs` are valid.
     * Post-condition: An existing connection for `clusterAlias` is either rebuilt or updated,
     *                 or the method returns if no existing connection requires rebuilding.
     */
    private void maybeRebuildConnectionOnCredentialsChange(
        ProjectId projectId,
        String clusterAlias,
        Settings settings,
        RefCountingRunnable connectionRefs
    ) {
        final var connectionsMap = getConnectionsMapForProject(projectId);
        // Block Logic: Checks if a connection for the given cluster alias already exists in the map.
        //              If it does not, there's no existing connection to rebuild, so the method returns.
        // Pre-condition: `connectionsMap` is initialized.
        if (false == connectionsMap.containsKey(clusterAlias)) {
            // A credential was added or removed before a remote connection was configured.
            // Without an existing connection, there is nothing to rebuild.
            logger.info(
                "project [{}] no connection rebuild required for remote cluster [{}] after credentials change",
                projectId,
                clusterAlias
            );
            return;
        }

        updateRemoteCluster(projectId, clusterAlias, settings, true, ActionListener.releaseAfter(new ActionListener<>() {
            @Override
            public void onResponse(RemoteClusterConnectionStatus status) {
                // Block Logic: Logs successful update of the remote cluster connection status.
                logger.info(
                    "project [{}] remote cluster connection [{}] updated after credentials change: [{}]",
                    projectId,
                    clusterAlias,
                    status
                );
            }

            @Override
            public void onFailure(Exception e) {
                // Block Logic: Logs a warning if the connection rebuild fails, ensuring the primary
                //              credential reload process isn't falsely marked as failed.
                // Invariant: Failures here indicate issues with connection re-establishment, not
                //            the secure settings reload itself.
                // We don't want to return an error to the upstream listener here since a connection rebuild failure
                // does *not* imply a failure to reload secure settings; however, that's how it would surface in the reload-settings call.
                // Instead, we log a warning which is also consistent with how we handle remote cluster settings updates (logging instead of
                // returning an error)
                logger.warn(
                    () -> "project ["
                        + projectId
                        + "] failed to update remote cluster connection ["
                        + clusterAlias
                        + "] after credentials change",
                    e
                );
            }
        }, connectionRefs.acquire()));
    }

    /**
     * @brief Updates the configuration of a remote cluster.
     * This protected method handles the logic for updating remote cluster settings and
     * synchronously waits for the update to complete within a timeout period.
     * @param clusterAlias The alias of the remote cluster to update.
     * @param settings The new {@link Settings} for the remote cluster.
     *
     * Functional Utility: Provides a synchronous entry point for internal components to
     *                     update remote cluster configurations, ensuring the update process
     *                     is complete before proceeding.
     * Pre-condition: `clusterAlias` refers to a valid remote cluster, and `settings` contains the desired updates.
     * Invariant: The method attempts to update the cluster and logs any issues, potentially blocking
     *            for a short duration.
     */
    @Override
    protected void updateRemoteCluster(String clusterAlias, Settings settings) {
        @FixForMultiProject(description = "ES-12270: Refactor as needed to support project specific changes to linked remotes.")
        final var projectId = projectResolver.getProjectId();
        // Inline: Initializes a latch to await the completion of the asynchronous update process.
        CountDownLatch latch = new CountDownLatch(1);
        // Block Logic: Calls the internal private method to perform the actual remote cluster update.
        //              A listener is provided to decrement the latch upon completion (success or failure).
        updateRemoteCluster(projectId, clusterAlias, settings, false, ActionListener.runAfter(new ActionListener<>() {
            @Override
            public void onResponse(RemoteClusterConnectionStatus status) {
                logger.info("project [{}] remote cluster connection [{}] updated: {}", projectId, clusterAlias, status);
            }

            @Override
            public void onFailure(Exception e) {
                logger.warn(() -> "project [" + projectId + " failed to update remote cluster connection [" + clusterAlias + "]", e);
            }
        }, latch::countDown));

        // Block Logic: Attempts to wait for the latch to count down, signifying the completion of the
        //              remote cluster update. A timeout is enforced to prevent indefinite blocking.
        // Invariant: The method will either complete after the update or after the timeout,
        //            logging appropriate messages for timeout or interruption.
        try {
            // Wait 10 seconds for a connections. We must use a latch instead of a future because we
            // are on the cluster state thread and our custom future implementation will throw an
            // assertion.
            if (latch.await(10, TimeUnit.SECONDS) == false) {
                logger.warn(
                    "project [{}] failed to update remote cluster connection [{}] within {}",
                    projectId,
                    clusterAlias,
                    TimeValue.timeValueSeconds(10)
                );
            }
        } catch (InterruptedException e) {
            // Inline: Restores the interrupted status of the current thread.
            Thread.currentThread().interrupt();
        } catch (TimeoutException ex) {
            logger.warn("project [{}] failed to update remote cluster connection [{}] within {}", projectId, clusterAlias, TimeValue.timeValueSeconds(10));
        } catch (Exception e) {
            logger.warn("project [" + projectId + "] failed to update remote cluster connection [" + clusterAlias + "]", e);
        }
    }

    // Package-access for testing.
    /**
     * @brief Updates the configuration of a remote cluster and notifies a listener.
     * This package-private method is used primarily for testing and provides an asynchronous way to
     * update a remote cluster's settings.
     * @param clusterAlias The alias of the remote cluster to update.
     * @param newSettings The new {@link Settings} for the remote cluster.
     * @param listener An {@link ActionListener} to be notified with the {@link RemoteClusterConnectionStatus}.
     *
     * Functional Utility: Allows for controlled, asynchronous updates to remote cluster configurations,
     *                     particularly useful in scenarios where immediate feedback via a listener is required.
     * Pre-condition: `clusterAlias` is a valid remote cluster alias, `newSettings` contains the updates,
     *                and `listener` is prepared to receive the status.
     * Post-condition: The remote cluster's configuration is updated, and the listener is invoked with the status.
     */
    @FixForMultiProject(description = "Refactor to supply the project ID associated with the alias and settings, or eliminate this method.")
    void updateRemoteCluster(String clusterAlias, Settings newSettings, ActionListener<RemoteClusterConnectionStatus> listener) {
        updateRemoteCluster(projectResolver.getProjectId(), clusterAlias, newSettings, false, listener);
    }

    /**
     * @brief Internal method to update, add, or remove a remote cluster connection.
     * This method orchestrates the lifecycle of a remote cluster connection based on the provided settings.
     * It handles closing existing connections, creating new ones, or rebuilding if necessary.
     * @param projectId The {@link ProjectId} associated with the remote cluster.
     * @param clusterAlias The alias of the remote cluster.
     * @param newSettings The updated {@link Settings} for the remote cluster.
     * @param forceRebuild If `true`, forces a rebuild of the connection even if settings haven't changed.
     * @param listener An {@link ActionListener} to be notified with the {@link RemoteClusterConnectionStatus}.
     *
     * Functional Utility: Centralizes the complex logic for managing remote cluster connections,
     *                     ensuring proper resource management and state transitions (connected, disconnected, reconnected).
     * Algorithm:
     * 1. Check for `LOCAL_CLUSTER_GROUP_KEY` alias, throwing an exception if used.
     * 2. Retrieve the existing `RemoteClusterConnection` for `clusterAlias`.
     * 3. Determine if the connection should be disabled based on `newSettings`. If so, close and remove it.
     * 4. If no existing connection, create a new one, add it to the map, and ensure it's connected.
     * 5. If an existing connection needs rebuilding (due to `forceRebuild` or changed settings), close the old,
     *    create a new, and ensure it's connected.
     * 6. Otherwise (no changes), notify the listener with `UNCHANGED`.
     * Pre-condition: `projectId`, `clusterAlias`, `newSettings`, `forceRebuild`, and `listener` are valid.
     * Post-condition: The remote cluster connection state is updated according to `newSettings`,
     *                 and the listener is invoked with the appropriate status.
     */
    private synchronized void updateRemoteCluster(
        ProjectId projectId,
        String clusterAlias,
        Settings newSettings,
        boolean forceRebuild,
        ActionListener<RemoteClusterConnectionStatus> listener
    ) {
        // Block Logic: Ensures that the special `LOCAL_CLUSTER_GROUP_KEY` is not used as a remote cluster alias.
        // Pre-condition: `clusterAlias` is provided.
        if (LOCAL_CLUSTER_GROUP_KEY.equals(clusterAlias)) {
            throw new IllegalArgumentException("remote clusters must not have the empty string as its key");
        }

        final var connectionMap = getConnectionsMapForProject(projectId);
        RemoteClusterConnection remote = connectionMap.get(clusterAlias);
        // Block Logic: Determines if the remote connection should be disabled based on the `newSettings`.
        //              If disabled, the existing connection is closed, removed from the map, and the listener
        //              is notified of the DISCONNECTED status.
        if (RemoteConnectionStrategy.isConnectionEnabled(clusterAlias, newSettings) == false) {
            try {
                IOUtils.close(remote);
            } catch (IOException e) {
                logger.warn("project [" + projectId + "] failed to close remote cluster connections for cluster: " + clusterAlias, e);
            }
            connectionMap.remove(clusterAlias);
            listener.onResponse(RemoteClusterConnectionStatus.DISCONNECTED);
            return;
        }

        // Block Logic: Handles the scenario where a new remote cluster is being configured.
        //              A new `RemoteClusterConnection` is created, added to the map, and an initial
        //              connection attempt is made.
        if (remote == null) {
            // this is a new cluster we have to add a new representation
            Settings finalSettings = Settings.builder().put(this.settings, false).put(newSettings, false).build();
            remote = new RemoteClusterConnection(finalSettings, clusterAlias, transportService, remoteClusterCredentialsManager);
            connectionMap.put(clusterAlias, remote);
            remote.ensureConnected(listener.map(ignored -> RemoteClusterConnectionStatus.CONNECTED));
        } else if (forceRebuild || remote.shouldRebuildConnection(newSettings)) {
            // Block Logic: Handles changes to connection configuration that require tearing down and
            //              rebuilding the existing connection. The old connection is closed, removed,
            //              a new one is created, added, and connected.
            try {
                IOUtils.close(remote);
            } catch (IOException e) {
                logger.warn("project [" + projectId + "] failed to close remote cluster connections for cluster: " + clusterAlias, e);
            }
            connectionMap.remove(clusterAlias);
            Settings finalSettings = Settings.builder().put(this.settings, false).put(newSettings, false).build();
            remote = new RemoteClusterConnection(finalSettings, clusterAlias, transportService, remoteClusterCredentialsManager);
            connectionMap.put(clusterAlias, remote);
            remote.ensureConnected(listener.map(ignored -> RemoteClusterConnectionStatus.RECONNECTED));
        } else {
            // Block Logic: If no changes to connection configuration are detected, the listener is notified
            //              with the UNCHANGED status.
            listener.onResponse(RemoteClusterConnectionStatus.UNCHANGED);
        }
    }

    enum RemoteClusterConnectionStatus {
        CONNECTED,
        DISCONNECTED,
        RECONNECTED,
        UNCHANGED
    }

    /**
     * Connects to all remote clusters in a blocking fashion. This should be called on node startup to establish an initial connection
     * to all configured seed nodes.
     */
    /**
     * @brief Connects to all remote clusters in a blocking fashion.
     * This method should be called on node startup to establish an initial connection
     * to all configured seed nodes, ensuring remote communication is ready.
     *
     * Functional Utility: Critical for bootstrapping the remote cluster communication infrastructure
     *                     at the beginning of a node's lifecycle.
     * Pre-condition: Remote clusters are configured in the node settings.
     * Post-condition: Attempts to connect to all configured remote clusters, blocking until
     *                 connections are established or a timeout occurs.
     */
    void initializeRemoteClusters() {
        @FixForMultiProject(description = "Refactor for initializing connections to linked projects for each origin project supported.")
        final var projectId = projectResolver.getProjectId();
        final TimeValue timeValue = REMOTE_INITIAL_CONNECTION_TIMEOUT_SETTING.get(settings);
        final PlainActionFuture<Void> future = new PlainActionFuture<>();
        Set<String> enabledClusters = RemoteClusterAware.getEnabledRemoteClusters(settings);

        // Block Logic: If no remote clusters are enabled in the settings, there's nothing to initialize,
        //              so the method returns early.
        // Pre-condition: `enabledClusters` contains the aliases of remote clusters to connect to.
        if (enabledClusters.isEmpty()) {
            return;
        }

        // Inline: Initializes a CountDownActionListener to track the completion of multiple asynchronous update operations.
        CountDownActionListener listener = new CountDownActionListener(enabledClusters.size(), future);
        // Block Logic: Iterates through each enabled remote cluster and triggers an update operation.
        //              Each update operation will eventually cause the `CountDownActionListener` to count down.
        for (String clusterAlias : enabledClusters) {
            updateRemoteCluster(projectId, clusterAlias, settings, false, listener.map(ignored -> null));
        }

        // Block Logic: If `enabledClusters` was empty initially (which should be caught by the previous `if`),
        //              the future is immediately completed to prevent hanging.
        if (enabledClusters.isEmpty()) {
            future.onResponse(null);
        }

        // Block Logic: Blocks the current thread until all remote clusters have attempted to connect
        //              or a timeout occurs. Handles InterruptedException and TimeoutException.
        // Invariant: The method will not proceed past this block until all connections are attempted
        //            or the initial connection timeout expires.
        try {
            future.get(timeValue.millis(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            // Inline: Restores the interrupted status of the current thread.
            Thread.currentThread().interrupt();
        } catch (TimeoutException ex) {
            logger.warn("project [{}] failed to connect to remote clusters within {}", projectId, timeValue.toString());
        } catch (Exception e) {
            logger.warn("project [" + projectId + "] failed to connect to remote clusters", e);
        }
    }

    @Override
    /**
     * @brief Closes all active remote cluster connections and releases associated resources.
     * Functional Utility: Ensures proper shutdown and resource cleanup for all managed remote cluster connections.
     * @throws IOException if an I/O error occurs during the closing of connections.
     * Post-condition: All {@link RemoteClusterConnection} instances held by this service are closed.
     */
    public void close() throws IOException {
        IOUtils.close(remoteClusters.values().stream().flatMap(map -> map.values().stream()).collect(Collectors.toList()));
    }

    /**
     * @brief Provides a stream of information about all currently active remote cluster connections.
     * Functional Utility: Allows for introspection into the state and details of each remote connection managed by the service.
     * @return A {@link Stream} of {@link RemoteConnectionInfo} objects, each representing an active remote cluster connection.
     */
    @FixForMultiProject(description = "Analyze use cases, determine possible need for cluster scoped and project scoped versions.")
    public Stream<RemoteConnectionInfo> getRemoteConnectionInfos() {
        return getConnectionsMapForCurrentProject().values().stream().map(RemoteClusterConnection::getConnectionInfo);
    }

    @Override
    /**
     * @brief Provides information about the remote cluster server capabilities of this node.
     * Functional Utility: Reports whether the remote cluster server is enabled and, if so, its bound remote access address.
     * @return A {@link RemoteClusterServerInfo} object if the remote cluster server is enabled, otherwise `null`.
     */
    public RemoteClusterServerInfo info() {
        if (remoteClusterServerEnabled) {
            return new RemoteClusterServerInfo(transportService.boundRemoteAccessAddress());
        } else {
            return null;
        }
    }

    /**
     * Collects all nodes of the given clusters and returns / passes a (clusterAlias, nodeId) to {@link DiscoveryNode}
     * function on success.
     */
    /**
     * @brief Collects all {@link DiscoveryNode}s from the specified remote clusters.
     * This method asynchronously collects all nodes of the given clusters and returns/passes
     * a (clusterAlias, nodeId) to {@link DiscoveryNode} function on success.
     * @param clusters A {@link Set} of remote cluster aliases from which to collect nodes.
     * @param listener An {@link ActionListener} that will be notified with a {@link BiFunction}
     *                 to look up DiscoveryNodes by cluster alias and node ID, or with a failure.
     *
     * Functional Utility: Provides a mechanism to obtain a comprehensive view of all available
     *                     nodes across multiple remote clusters, essential for routing requests or
     *                     gathering cluster topology information.
     * Pre-condition: Remote cluster client functionality must be enabled on this node, and
     *                `clusters` must contain valid remote cluster aliases.
     * Post-condition: The listener is invoked with a function to query nodes, or with an exception
     *                 if any specified cluster is unknown or if the remote cluster client is disabled.
     */
    public void collectNodes(Set<String> clusters, ActionListener<BiFunction<String, String, DiscoveryNode>> listener) {
        // Block Logic: Checks if the remote cluster client functionality is enabled on this node.
        //              If not, it throws an `IllegalArgumentException` as this operation requires it.
        // Pre-condition: This node is configured as a remote cluster client.
        if (enabled == false) {
            throw new IllegalArgumentException(
                "this node does not have the " + DiscoveryNodeRole.REMOTE_CLUSTER_CLIENT_ROLE.roleName() + " role"
            );
        }
        @FixForMultiProject(description = "Analyze usages and determine if the project ID must be provided.")
        final var projectConnectionsMap = getConnectionsMapForCurrentProject();
        final var connectionsMap = new HashMap<String, RemoteClusterConnection>();
        // Block Logic: Iterates through the requested cluster aliases, retrieves their corresponding
        //              `RemoteClusterConnection` objects, and populates a local `connectionsMap`.
        //              If any cluster alias is unknown, it fails the listener immediately.
        // Pre-condition: `clusters` contains valid remote cluster aliases.
        for (String cluster : clusters) {
            final var connection = projectConnectionsMap.get(cluster);
            if (connection == null) {
                listener.onFailure(new NoSuchRemoteClusterException(cluster));
                return;
            }
            connectionsMap.put(cluster, connection);
        }

        final Map<String, Function<String, DiscoveryNode>> clusterMap = new HashMap<>();
        // Block Logic: Creates a final listener that maps the collected node lookups to the expected
        //              BiFunction output, and uses `RefCountingListener` to manage multiple asynchronous
        //              node collection tasks.
        // Invariant: The `finalListener` is only completed after all individual cluster node collection
        //            operations have finished.
        try (var refs = new RefCountingListener(finalListener)) {
            connectionsMap.forEach((cluster, connection) -> connection.collectNodes(refs.acquire(nodeLookup -> {
                synchronized (clusterMap) {
                    clusterMap.put(cluster, nodeLookup);
                }
            })));
        }
    }

    /**
     * Specifies how to behave when executing a request against a disconnected remote cluster.
     */
    public enum DisconnectedStrategy {
        /**
         * Always try and reconnect before executing a request, waiting for {@link TransportSettings#CONNECT_TIMEOUT} before failing if the
         * remote cluster is totally unresponsive.
         */
        RECONNECT_IF_DISCONNECTED,

        /**
         * Fail the request immediately if the remote cluster is disconnected (but also trigger another attempt to reconnect to the remote
         * cluster in the background so that the next request might succeed).
         */
        FAIL_IF_DISCONNECTED,

        /**
         * Behave according to the {@link #REMOTE_CLUSTER_SKIP_UNAVAILABLE} setting for this remote cluster: if this setting is
         * {@code false} (the default) then behave like {@link #RECONNECT_IF_DISCONNECTED}, but if it is {@code true} then behave like
         * {@link #FAIL_IF_DISCONNECTED}.
         */
        RECONNECT_UNLESS_SKIP_UNAVAILABLE
    }

    /**
     * Returns a client to the remote cluster if the given cluster alias exists.
     *
     * @param clusterAlias         the cluster alias the remote cluster is registered under
     * @param responseExecutor     the executor to use to process the response
     * @param disconnectedStrategy how to handle the situation where the remote cluster is disconnected when executing a request
     * @throws IllegalArgumentException if the given clusterAlias doesn't exist
     */
    /**
     * @brief Returns a client to the remote cluster if the given cluster alias exists.
     * This client can be used to send requests to the remote cluster, with behavior governed by the `disconnectedStrategy`.
     * @param clusterAlias The alias of the remote cluster the client should connect to.
     * @param responseExecutor The {@link Executor} to use for processing responses from the remote cluster.
     * @param disconnectedStrategy Defines how to behave when executing a request against a disconnected remote cluster.
     * @return A {@link RemoteClusterClient} instance configured for the specified remote cluster.
     * @throws IllegalArgumentException if the remote cluster client functionality is not enabled on this node,
     *                                  or if the given `clusterAlias` does not exist.
     *
     * Functional Utility: Provides a robust interface for sending requests to remote clusters,
     *                     abstracting away connection management and handling strategies for
     *                     disconnected states.
     * Pre-condition: Remote cluster client role is enabled on this node, and `clusterAlias` corresponds
     *                to a configured remote cluster.
     */
    public RemoteClusterClient getRemoteClusterClient(
        String clusterAlias,
        Executor responseExecutor,
        DisconnectedStrategy disconnectedStrategy
    ) {
        // Block Logic: Verifies that the remote cluster client functionality is enabled on this node.
        //              If not, an `IllegalArgumentException` is thrown, preventing operations that require this role.
        // Pre-condition: This node has the `REMOTE_CLUSTER_CLIENT_ROLE`.
        if (transportService.getRemoteClusterService().isEnabled() == false) {
            throw new IllegalArgumentException(
                "this node does not have the " + DiscoveryNodeRole.REMOTE_CLUSTER_CLIENT_ROLE.roleName() + " role"
            );
        }
        // Block Logic: Checks if the provided `clusterAlias` is a recognized and configured remote cluster.
        //              If not, a `NoSuchRemoteClusterException` is thrown.
        // Pre-condition: `clusterAlias` corresponds to an existing remote cluster configuration.
        if (transportService.getRemoteClusterService().getRegisteredRemoteClusterNames().contains(clusterAlias) == false) {
            throw new NoSuchRemoteClusterException(clusterAlias);
        }
        // Block Logic: Uses a switch expression to determine the appropriate `ensureConnected` boolean value
        //              based on the `DisconnectedStrategy`. This controls whether requests will attempt
        //              to reconnect or fail immediately if the remote cluster is disconnected.
        // Invariant: The returned `RemoteClusterAwareClient` is configured with the correct connection behavior.
        return new RemoteClusterAwareClient(transportService, clusterAlias, responseExecutor, switch (disconnectedStrategy) {
            case RECONNECT_IF_DISCONNECTED -> true;
            case FAIL_IF_DISCONNECTED -> false;
            case RECONNECT_UNLESS_SKIP_UNAVAILABLE -> transportService.getRemoteClusterService().isSkipUnavailable(clusterAlias) == false;
        });
    }

    /**
     * @brief Registers the request handler for remote cluster handshake requests.
     * This static method sets up the necessary infrastructure for nodes to perform a handshake
     * when establishing a connection to a remote cluster, verifying compatibility and exchanging node information.
     * @param transportService The {@link TransportService} instance to register the handler with.
     *
     * Functional Utility: Establishes a foundational communication protocol for secure and compatible
     *                     inter-cluster communication.
     * Pre-condition: `transportService` is initialized and ready to register request handlers.
     * Post-condition: The {@link #REMOTE_CLUSTER_HANDSHAKE_ACTION_NAME} action is registered,
     *                 and incoming handshake requests will be processed by the provided lambda.
     */
    static void registerRemoteClusterHandshakeRequestHandler(TransportService transportService) {
        transportService.registerRequestHandler(
            REMOTE_CLUSTER_HANDSHAKE_ACTION_NAME,
            EsExecutors.DIRECT_EXECUTOR_SERVICE,
            false,
            false,
            TransportService.HandshakeRequest::new,
            (request, channel, task) -> {
                // Block Logic: Validates that the channel used for the handshake request has the expected
                //              profile name for remote cluster communication. This ensures secure and
                //              correct protocol usage.
                // Pre-condition: The channel's profile name is set.
                if (false == RemoteClusterPortSettings.REMOTE_CLUSTER_PROFILE.equals(channel.getProfileName())) {
                    throw new IllegalArgumentException(
                        Strings.format(
                            "remote cluster handshake action requires channel profile to be [%s], but got [%s]",
                            RemoteClusterPortSettings.REMOTE_CLUSTER_PROFILE,
                            channel.getProfileName()
                        )
                    );
                }
                logger.trace("handling remote cluster handshake request");
                channel.sendResponse(
                    new TransportService.HandshakeResponse(
                        transportService.getLocalNode().getVersion(),
                        Build.current().hash(),
                        transportService.getLocalNode().withTransportAddress(transportService.boundRemoteAccessAddress().publishAddress()),
                        transportService.clusterName
                    )
                );
            }
        );
    }

    /**
     * Returns the map of connections for the {@link ProjectId} currently returned by the {@link ProjectResolver}.
     */
    /**
     * @brief Retrieves the map of {@link RemoteClusterConnection}s for the project ID currently resolved by the {@link ProjectResolver}.
     * Functional Utility: Provides the correct context-specific map of remote cluster connections, especially important
     *                     in multi-project environments where connections might be isolated per project.
     * @return A {@link Map} where keys are cluster aliases (String) and values are {@link RemoteClusterConnection} objects,
     *         specific to the current project.
     * Pre-condition: The {@link ProjectResolver} is correctly configured and can resolve the current {@link ProjectId}.
     */
    private Map<String, RemoteClusterConnection> getConnectionsMapForCurrentProject() {
        return getConnectionsMapForProject(projectResolver.getProjectId());
    }

    /**
     * Returns the map of connections for the given {@link ProjectId}.
     */
    /**
     * @brief Retrieves the map of {@link RemoteClusterConnection}s for a given {@link ProjectId}.
     * Handles the instantiation of new maps for project IDs if multiple projects are supported.
     * @param projectId The {@link ProjectId} for which to retrieve the connections map.
     * @return A {@link Map} where keys are cluster aliases (String) and values are {@link RemoteClusterConnection} objects.
     *
     * Functional Utility: Manages the isolation and retrieval of remote cluster connections on a per-project basis,
     *                     supporting multi-tenancy or multi-project architectures.
     * Pre-condition: `projectId` is a valid identifier.
     * Invariant: In a multi-project environment, a new concurrent map is created for a projectId if it doesn't exist.
     *            In a single-project environment, only the default project ID is expected.
     */
    private Map<String, RemoteClusterConnection> getConnectionsMapForProject(ProjectId projectId) {
        if (projectResolver.supportsMultipleProjects()) {
            assert ProjectId.DEFAULT.equals(projectId) == false : "The default project ID should not be used in multi-project environment";
            return remoteClusters.computeIfAbsent(projectId, unused -> ConcurrentCollections.newConcurrentMap());
        }
        assert ProjectId.DEFAULT.equals(projectId) : "Only the default project ID should be used when multiple projects are not supported";
        return remoteClusters.get(projectId);
    }

    /**
     * @class RemoteConnectionEnabled
     * @brief A setting validator that ensures a remote cluster connection is enabled before allowing certain settings to be configured.
     * Functional Utility: Enforces a dependency between the overall enablement of a remote cluster connection
     *                     and the configuration of its specific settings (e.g., ping schedule, compression).
     *                     This prevents misconfigurations of disabled remote clusters.
     * @param <T> The type of the setting value being validated.
     */
    private static class RemoteConnectionEnabled<T> implements Setting.Validator<T> {

        private final String clusterAlias;
        private final String key;

        /**
         * @brief Constructs a new `RemoteConnectionEnabled` validator.
         * @param clusterAlias The alias of the remote cluster for which the setting is being validated.
         * @param key The full key of the setting being validated (e.g., "cluster.remote.my_cluster.transport.ping_schedule").
         * Functional Utility: Initializes the validator with the context needed to check the remote connection's enablement.
         */
        private RemoteConnectionEnabled(String clusterAlias, String key) {
            this.clusterAlias = clusterAlias;
            this.key = key;
        }

        @Override
        public void validate(T value) {}

        @Override
        /**
         * @brief Validates the setting, ensuring the remote cluster connection is enabled if the setting is present.
         * @param value The value of the setting.
         * @param settings A map of all current settings.
         * @param isPresent `true` if the setting is present in the provided settings, `false` otherwise.
         * @throws IllegalArgumentException if the setting is present but the remote cluster connection is not enabled.
         * Functional Utility: Implements the core validation logic, checking the `RemoteConnectionStrategy`
         *                     to ensure consistency.
         * Pre-condition: `settings` contains the necessary `RemoteConnectionStrategy` setting for the `clusterAlias`.
         */
        public void validate(T value, Map<Setting<?>, Object> settings, boolean isPresent) {
            if (isPresent && RemoteConnectionStrategy.isConnectionEnabled(clusterAlias, settings) == false) {
                throw new IllegalArgumentException("Cannot configure setting [" + key + "] if remote cluster is not enabled.");
            }
        }

        @Override
        /**
         * @brief Returns an iterator over the settings that this validator depends on.
         * Functional Utility: Declares the dependencies of this validator, allowing the settings system
         *                     to correctly order validation and update processes.
         * @return An {@link Iterator} of {@link Setting} objects that this validator needs to inspect.
         */
        public Iterator<Setting<?>> settings() {
            return Stream.concat(
                Stream.of(RemoteConnectionStrategy.REMOTE_CONNECTION_MODE.getConcreteSettingForNamespace(clusterAlias)),
                settingsStream()
            ).iterator();
        }

        /**
         * @brief Provides a stream of enablement settings for various connection strategies.
         * Functional Utility: Helper method to gather all relevant enablement settings across different
         *                     `RemoteConnectionStrategy.ConnectionStrategy` types.
         * @return A {@link Stream} of {@link Setting} objects related to connection enablement.
         */
        private Stream<Setting<?>> settingsStream() {
            return Arrays.stream(RemoteConnectionStrategy.ConnectionStrategy.values())
                .flatMap(strategy -> strategy.getEnablementSettings().get())
                .map(as -> as.getConcreteSettingForNamespace(clusterAlias));
        }
    };
}
