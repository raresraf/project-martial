/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.ingest.geoip;

import com.maxmind.db.NodeCache;

import org.elasticsearch.cluster.metadata.ProjectId;
import org.elasticsearch.common.cache.Cache;
import org.elasticsearch.common.cache.CacheBuilder;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.ingest.geoip.stats.CacheStats;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.function.LongSupplier;

/**
 * @2db6c093-4fb2-459a-a2ca-2192b0b1f0be/modules/ingest-geoip/src/main/java/org/elasticsearch/ingest/geoip/GeoIpCache.java
 * @brief High-performance, in-memory cache for deserialized GeoIP database lookups.
 * 
 * Functional Intent: Reduces CPU overhead by caching fully deserialized JSON objects 
 * from MaxMind databases, avoiding redundant parsing on every request. It uses a 
 * compound key (project, IP, database path) to support multi-tenancy and 
 * multi-database environments.
 */
public final class GeoIpCache {

    /**
     * Functional Utility: Sentinel object to represent negative cache hits (null results).
     * Logic: Allows the cache to distinguish between "not searched" and "searched but not found", 
     * preventing repeated database lookups for non-existent IP metadata.
     */
    // visible for testing
    static final Object NO_RESULT = new Object() {
        @Override
        public String toString() {
            return "NO_RESULT";
        }
    };

    private final LongSupplier relativeNanoTimeProvider;
    private final Cache<CacheKey, Object> cache;
    private final AtomicLong hitsTimeInNanos = new AtomicLong(0);
    private final AtomicLong missesTimeInNanos = new AtomicLong(0);

    /**
     * @brief Internal constructor for dependency injection of time providers.
     * @param maxSize Maximum weighted size of the cache.
     * @param relativeNanoTimeProvider Supplier for nanosecond-precision timing.
     */
    GeoIpCache(long maxSize, LongSupplier relativeNanoTimeProvider) {
        // Block Logic: Input validation for cache capacity.
        if (maxSize < 0) {
            throw new IllegalArgumentException("geoip max cache size must be 0 or greater");
        }
        this.relativeNanoTimeProvider = relativeNanoTimeProvider;
        this.cache = CacheBuilder.<CacheKey, Object>builder().setMaximumWeight(maxSize).build();
    }

    /**
     * @brief Public constructor utilizing the system monotonic clock.
     * @param maxSize Maximum weighted size of the cache.
     */
    GeoIpCache(long maxSize) {
        this(maxSize, System::nanoTime);
    }

    /**
     * Block Logic: Thread-safe, non-locking population of cache entries.
     * Logic: 
     * 1. Probes the cache for an existing result.
     * 2. On miss: Executes the provided retrieval function, wraps nulls in the 
     *    NO_RESULT sentinel, and commits the result to the cache.
     * 3. Aggregates timing metrics for performance monitoring.
     * 
     * @param <RESPONSE> The expected response type.
     * @param projectId Tenant identifier.
     * @param ip The IP address to resolve.
     * @param databasePath Path to the MaxMind database used.
     * @param retrieveFunction Logic to execute on cache miss.
     * @return The resolved GeoIP metadata, or null if not found.
     */
    @SuppressWarnings("unchecked")
    <RESPONSE> RESPONSE putIfAbsent(ProjectId projectId, String ip, String databasePath, Function<String, RESPONSE> retrieveFunction) {
        CacheKey cacheKey = new CacheKey(projectId, ip, databasePath);
        long cacheStart = relativeNanoTimeProvider.getAsLong();
        Object response = cache.get(cacheKey);
        long cacheRequestTime = relativeNanoTimeProvider.getAsLong() - cacheStart;

        // Block Logic: Cache miss handling.
        if (response == null) {
            long retrieveStart = relativeNanoTimeProvider.getAsLong();
            response = retrieveFunction.apply(ip);
            
            // Logic: Canonicalize null responses to the internal sentinel.
            if (response == null) {
                response = NO_RESULT;
            }
            cache.put(cacheKey, response);
            long databaseRequestAndCachePutTime = relativeNanoTimeProvider.getAsLong() - retrieveStart;
            missesTimeInNanos.addAndGet(cacheRequestTime + databaseRequestAndCachePutTime);
        } else {
            hitsTimeInNanos.addAndGet(cacheRequestTime);
        }

        // Functional Utility: Unwraps sentinel objects before returning to the caller.
        if (response == NO_RESULT) {
            return null;
        } else {
            return (RESPONSE) response;
        }
    }

    /**
     * @brief Direct accessor for internal state (testing only).
     */
    Object get(ProjectId projectId, String ip, String databasePath) {
        CacheKey cacheKey = new CacheKey(projectId, ip, databasePath);
        return cache.get(cacheKey);
    }

    /**
     * Block Logic: Selective cache invalidation.
     * Logic: Iterates through all cached keys and removes those associated with 
     * a specific project and database file, facilitating clean cleanup when 
     * databases are updated or projects are deleted.
     * 
     * @param projectId Target project.
     * @param databaseFile Path to the database being purged.
     * @return Count of invalidated entries.
     */
    public int purgeCacheEntriesForDatabase(ProjectId projectId, Path databaseFile) {
        String databasePath = databaseFile.toString();
        int counter = 0;
        // Invariant: Only entries matching both criteria are invalidated.
        for (CacheKey key : cache.keys()) {
            if (key.projectId.equals(projectId) && key.databasePath.equals(databasePath)) {
                cache.invalidate(key);
                counter++;
            }
        }
        return counter;
    }

    /**
     * @brief Returns the total number of items currently in the cache.
     */
    public int count() {
        return cache.count();
    }

    /**
     * Block Logic: Performance metric aggregation.
     * Logic: Combines internal cache implementation stats with high-level 
     * timing metrics gathered during the putIfAbsent cycle.
     * 
     * @return Snapshot of current cache performance.
     */
    public CacheStats getCacheStats() {
        Cache.CacheStats stats = cache.stats();
        return new CacheStats(
            cache.count(),
            stats.getHits(),
            stats.getMisses(),
            stats.getEvictions(),
            TimeValue.nsecToMSec(hitsTimeInNanos.get()),
            TimeValue.nsecToMSec(missesTimeInNanos.get())
        );
    }

    /**
     * @brief Immutable record for uniquely identifying cached GeoIP results.
     */
    private record CacheKey(ProjectId projectId, String ip, String databasePath) {}
}
