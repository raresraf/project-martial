/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.common.cache;

import org.elasticsearch.core.Tuple;

import java.lang.reflect.Array;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.ToLongBiFunction;

/**
 * A simple concurrent cache.
 * <p>
 * Cache is a simple concurrent cache that supports time-based and weight-based evictions, with notifications for all
 * evictions. The design goals for this cache were simplicity and read performance. This means that we are willing to
 * accept reduced write performance in exchange for easy-to-understand code. Cache statistics for hits, misses and
 * evictions are exposed.
 * <p>
 * The design of the cache is relatively simple. The cache is segmented into 256 segments which are backed by HashMaps.
 * Each segment is protected by a re-entrant read/write lock. The read/write locks permit multiple concurrent readers
 * without contention, and the segments gives us write throughput without impacting readers (so readers are blocked only
 * if they are reading a segment that a writer is writing to).
 * <p>
 * The LRU functionality is backed by a single doubly-linked list chaining the entries in order of insertion. This
 * LRU list is protected by a lock that serializes all writes to it. There are opportunities for improvements
 * here if write throughput is a concern.
 * <ol>
 * <li>LRU list mutations could be inserted into a blocking queue that a single thread is reading from
 * and applying to the LRU list.</li>
 * <li>Promotions could be deferred for entries that were "recently" promoted.</li>
 * <li>Locks on the list could be taken per node being modified instead of globally.</li>
 * </ol>
 * <p>
 * Evictions only occur after a mutation to the cache (meaning an entry promotion, a cache insertion, or a manual
 * invalidation) or an explicit call to {@link #refresh()}.
 *
 * @param <K> The type of the keys
 * @param <V> The type of the values
 */
public class Cache<K, V> {

    private final LongAdder hits = new LongAdder();

    private final LongAdder misses = new LongAdder();

    private final LongAdder evictions = new LongAdder();

    // positive if entries have an expiration
    private long expireAfterAccessNanos = -1;

    // true if entries can expire after access
    private boolean entriesExpireAfterAccess;

    // positive if entries have an expiration after write
    private long expireAfterWriteNanos = -1;

    // true if entries can expire after initial insertion
    private boolean entriesExpireAfterWrite;

    // the number of entries in the cache
    private int count = 0;

    // the weight of the entries in the cache
    private long weight = 0;

    // the maximum weight that this cache supports
    private long maximumWeight = -1;

    // the weigher of entries
    private ToLongBiFunction<K, V> weigher = (k, v) -> 1;

    // the removal callback
    private RemovalListener<K, V> removalListener = notification -> {};

    // use CacheBuilder to construct
    /**
     * @brief [Functional Utility for Cache]: Describe purpose here.
     */
    Cache() {}

    /**
     * @brief [Functional Utility for setExpireAfterAccessNanos]: Describe purpose here.
     */
    void setExpireAfterAccessNanos(long expireAfterAccessNanos) {
        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (expireAfterAccessNanos <= 0) {
            throw new IllegalArgumentException("expireAfterAccessNanos <= 0");
        }
        this.expireAfterAccessNanos = expireAfterAccessNanos;
        this.entriesExpireAfterAccess = true;
    }

    // public for testing
    /**
     * @brief [Functional Utility for getExpireAfterAccessNanos]: Describe purpose here.
     */
    public long getExpireAfterAccessNanos() {
        return this.expireAfterAccessNanos;
    }

    /**
     * @brief [Functional Utility for setExpireAfterWriteNanos]: Describe purpose here.
     */
    void setExpireAfterWriteNanos(long expireAfterWriteNanos) {
        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (expireAfterWriteNanos <= 0) {
            throw new IllegalArgumentException("expireAfterWriteNanos <= 0");
        }
        this.expireAfterWriteNanos = expireAfterWriteNanos;
        this.entriesExpireAfterWrite = true;
    }

    // pkg-private for testing
    /**
     * @brief [Functional Utility for getExpireAfterWriteNanos]: Describe purpose here.
     */
    long getExpireAfterWriteNanos() {
        return this.expireAfterWriteNanos;
    }

    /**
     * @brief [Functional Utility for setMaximumWeight]: Describe purpose here.
     */
    void setMaximumWeight(long maximumWeight) {
        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (maximumWeight < 0) {
            throw new IllegalArgumentException("maximumWeight < 0");
        }
        this.maximumWeight = maximumWeight;
    }

    /**
     * @brief [Functional Utility for setWeigher]: Describe purpose here.
     */
    void setWeigher(ToLongBiFunction<K, V> weigher) {
        Objects.requireNonNull(weigher);
        this.weigher = weigher;
    }

    /**
     * @brief [Functional Utility for setRemovalListener]: Describe purpose here.
     */
    void setRemovalListener(RemovalListener<K, V> removalListener) {
        Objects.requireNonNull(removalListener);
        this.removalListener = removalListener;
    }

    /**
     * The relative time used to track time-based evictions.
     *
     * @return the current relative time
     */
    /**
     * @brief [Functional Utility for now]: Describe purpose here.
     */
    protected long now() {
        // System.nanoTime takes non-negligible time, so we only use it if we need it
        // use System.nanoTime because we want relative time, not absolute time
        return entriesExpireAfterAccess || entriesExpireAfterWrite ? System.nanoTime() : 0;
    }

    // the state of an entry in the LRU list
    enum State {
        NEW,
        EXISTING,
        DELETED
    }

    private static final class Entry<K, V> {
        final K key;
        final V value;
        final long writeTime;
        volatile long accessTime;
        Entry<K, V> before;
        Entry<K, V> after;
        State state = State.NEW;

        /**
         * @brief [Functional Utility for Entry]: Describe purpose here.
         */
        Entry(K key, V value, long writeTime) {
            this.key = key;
            this.value = value;
            this.writeTime = this.accessTime = writeTime;
        }
    }

    /**
     * A cache segment.
     * <p>
     * A CacheSegment is backed by a HashMap and is protected by a read/write lock.
     */
    private final class CacheSegment {
        // read/write lock protecting mutations to the segment
        final ReadWriteLock segmentLock = new ReentrantReadWriteLock();

        final Lock readLock = segmentLock.readLock();
        final Lock writeLock = segmentLock.writeLock();

        Map<K, CompletableFuture<Entry<K, V>>> map;

        /**
         * get an entry from the segment; expired entries will be returned as null but not removed from the cache until the LRU list is
         * pruned or a manual {@link Cache#refresh()} is performed however a caller can take action using the provided callback
         *
         * @param key       the key of the entry to get from the cache
         * @param now       the access time of this entry
         * @param eagerEvict whether entries should be eagerly evicted on expiration
         * @return the entry if there was one, otherwise null
         */
        Entry<K, V> get(K key, long now, boolean eagerEvict) {
            CompletableFuture<Entry<K, V>> future;
            readLock.lock();
            try {
                future = map == null ? null : map.get(key);
            } finally {
                readLock.unlock();
            }
            /**
             * @brief [Functional Utility for if]: Describe purpose here.
             */
            if (future != null) {
                Entry<K, V> entry;
                try {
                    entry = future.get();
                } catch (ExecutionException e) {
                    assert future.isCompletedExceptionally();
                    misses.increment();
                    return null;
                } catch (InterruptedException e) {
                    throw new IllegalStateException(e);
                }
                // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                // Invariant: State condition that holds true before and after each iteration/execution
                if (isExpired(entry, now)) {
                    misses.increment();
                    /**
                     * @brief [Functional Utility for if]: Describe purpose here.
                     */
                    if (eagerEvict) {
                        lruLock.lock();
                        try {
                            evictEntry(entry);
                        } finally {
                            lruLock.unlock();
                        }
                    }
                    return null;
                } else {
                    hits.increment();
                    entry.accessTime = now;
                    return entry;
                }
            } else {
                misses.increment();
                return null;
            }
        }

        /**
         * put an entry into the segment
         *
         * @param key   the key of the entry to add to the cache
         * @param value the value of the entry to add to the cache
         * @param now   the access time of this entry
         * @return a tuple of the new entry and the existing entry, if there was one otherwise null
         */
        Tuple<Entry<K, V>, Entry<K, V>> put(K key, V value, long now) {
            Entry<K, V> entry = new Entry<>(key, value, now);
            Entry<K, V> existing = null;
            writeLock.lock();
            try {
                try {
                    /**
                     * @brief [Functional Utility for if]: Describe purpose here.
                     */
                    if (map == null) {
                        map = new HashMap<>();
                    }
                    CompletableFuture<Entry<K, V>> future = map.put(key, CompletableFuture.completedFuture(entry));
                    /**
                     * @brief [Functional Utility for if]: Describe purpose here.
                     */
                    if (future != null) {
                        existing = future.handle((ok, ex) -> ok).get();
                    }
                } catch (ExecutionException | InterruptedException e) {
                    throw new IllegalStateException(e);
                }
            } finally {
                writeLock.unlock();
            }
            return Tuple.tuple(entry, existing);
        }

        /**
         * remove an entry from the segment
         *
         * @param key       the key of the entry to remove from the cache
         */
        /**
         * @brief [Functional Utility for remove]: Describe purpose here.
         */
        void remove(K key) {
            CompletableFuture<Entry<K, V>> future;
            writeLock.lock();
            try {
                /**
                 * @brief [Functional Utility for if]: Describe purpose here.
                 */
                if (map == null) {
                    future = null;
                } else {
                    future = map.remove(key);
                    // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    // Invariant: State condition that holds true before and after each iteration/execution
                    if (map.isEmpty()) {
                        map = null;
                    }
                }
            } finally {
                writeLock.unlock();
            }
            /**
             * @brief [Functional Utility for if]: Describe purpose here.
             */
            if (future != null) {
                evictions.increment();
                notifyWithInvalidated(future);
            }
        }

        /**
         * remove an entry from the segment iff the future is done and the value is equal to the
         * expected value
         *
         * @param key the key of the entry to remove from the cache
         * @param value the value expected to be associated with the key
         * @param notify whether to trigger a removal notification if the entry has been removed
         */
        /**
         * @brief [Functional Utility for remove]: Describe purpose here.
         */
        void remove(K key, V value, boolean notify) {
            CompletableFuture<Entry<K, V>> future;
            boolean removed = false;
            writeLock.lock();
            try {
                future = map == null ? null : map.get(key);
                try {
                    /**
                     * @brief [Functional Utility for if]: Describe purpose here.
                     */
                    if (future != null) {
                        // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                        // Invariant: State condition that holds true before and after each iteration/execution
                        if (future.isDone()) {
                            Entry<K, V> entry = future.get();
                            // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                            // Invariant: State condition that holds true before and after each iteration/execution
                            if (Objects.equals(value, entry.value)) {
                                removed = map.remove(key, future);
                                // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                                // Invariant: State condition that holds true before and after each iteration/execution
                                if (map.isEmpty()) {
                                    map = null;
                                }
                            }
                        }
                    }
                } catch (ExecutionException | InterruptedException e) {
                    throw new IllegalStateException(e);
                }
            } finally {
                writeLock.unlock();
            }

            /**
             * @brief [Functional Utility for if]: Describe purpose here.
             */
            if (future != null && removed) {
                evictions.increment();
                /**
                 * @brief [Functional Utility for if]: Describe purpose here.
                 */
                if (notify) {
                    notifyWithInvalidated(future);
                }
            }
        }

    }

    public static final int NUMBER_OF_SEGMENTS = 256;
    @SuppressWarnings("unchecked")
    private final CacheSegment[] segments = (CacheSegment[]) Array.newInstance(CacheSegment.class, NUMBER_OF_SEGMENTS);

    {
        /**
         * @brief [Functional Utility for for]: Describe purpose here.
         */
        for (int i = 0; i < segments.length; i++) {
            segments[i] = new CacheSegment();
        }
    }

    Entry<K, V> head;
    Entry<K, V> tail;

    // lock protecting mutations to the LRU list
    private final ReentrantLock lruLock = new ReentrantLock();

    /**
     * Returns the value to which the specified key is mapped, or null if this map contains no mapping for the key.
     *
     * @param key the key whose associated value is to be returned
     * @return the value to which the specified key is mapped, or null if this map contains no mapping for the key
     */
    /**
     * @brief [Functional Utility for get]: Describe purpose here.
     */
    public V get(K key) {
        return get(key, now(), false);
    }

    /**
     * @brief [Functional Utility for get]: Describe purpose here.
     */
    private V get(K key, long now, boolean eagerEvict) {
        CacheSegment segment = getCacheSegment(key);
        Entry<K, V> entry = segment.get(key, now, eagerEvict);
        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (entry == null) {
            return null;
        } else {
            promote(entry, now);
            return entry.value;
        }
    }

    /**
     * If the specified key is not already associated with a value (or is mapped to null), attempts to compute its
     * value using the given mapping function and enters it into this map unless null. The load method for a given key
     * will be invoked at most once.
     *
     * Use of different {@link CacheLoader} implementations on the same key concurrently may result in only the first
     * loader function being called and the second will be returned the result provided by the first including any exceptions
     * thrown during the execution of the first.
     *
     * @param key    the key whose associated value is to be returned or computed for if non-existent
     * @param loader the function to compute a value given a key
     * @return the current (existing or computed) non-null value associated with the specified key
     * @throws ExecutionException thrown if loader throws an exception or returns a null value
     */
    /**
     * @brief [Functional Utility for computeIfAbsent]: Describe purpose here.
     */
    public V computeIfAbsent(K key, CacheLoader<K, V> loader) throws ExecutionException {
        long now = now();
        // we have to eagerly evict expired entries or our putIfAbsent call below will fail
        V value = get(key, now, true);
        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (value == null) {
            // we need to synchronize loading of a value for a given key; however, holding the segment lock while
            // invoking load can lead to deadlock against another thread due to dependent key loading; therefore, we
            // need a mechanism to ensure that load is invoked at most once, but we are not invoking load while holding
            // the segment lock; to do this, we atomically put a future in the map that can load the value, and then
            // get the value from this future on the thread that won the race to place the future into the segment map
            final CacheSegment segment = getCacheSegment(key);
            CompletableFuture<Entry<K, V>> future;
            CompletableFuture<Entry<K, V>> completableFuture = new CompletableFuture<>();

            segment.writeLock.lock();
            try {
                /**
                 * @brief [Functional Utility for if]: Describe purpose here.
                 */
                if (segment.map == null) {
                    segment.map = new HashMap<>();
                }
                future = segment.map.putIfAbsent(key, completableFuture);
            } finally {
                segment.writeLock.unlock();
            }

            BiFunction<? super Entry<K, V>, Throwable, ? extends V> handler = (ok, ex) -> {
                /**
                 * @brief [Functional Utility for if]: Describe purpose here.
                 */
                if (ok != null) {
                    promote(ok, now);
                    return ok.value;
                } else {
                    segment.writeLock.lock();
                    try {
                        CompletableFuture<Entry<K, V>> sanity = segment.map == null ? null : segment.map.get(key);
                        // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                        // Invariant: State condition that holds true before and after each iteration/execution
                        if (sanity != null && sanity.isCompletedExceptionally()) {
                            segment.map.remove(key);
                            // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                            // Invariant: State condition that holds true before and after each iteration/execution
                            if (segment.map.isEmpty()) {
                                segment.map = null;
                            }
                        }
                    } finally {
                        segment.writeLock.unlock();
                    }
                    return null;
                }
            };

            CompletableFuture<V> completableValue;
            /**
             * @brief [Functional Utility for if]: Describe purpose here.
             */
            if (future == null) {
                future = completableFuture;
                completableValue = future.handle(handler);
                V loaded;
                try {
                    loaded = loader.load(key);
                } catch (Exception e) {
                    future.completeExceptionally(e);
                    throw new ExecutionException(e);
                }
                /**
                 * @brief [Functional Utility for if]: Describe purpose here.
                 */
                if (loaded == null) {
                    NullPointerException npe = new NullPointerException("loader returned a null value");
                    future.completeExceptionally(npe);
                    throw new ExecutionException(npe);
                } else {
                    future.complete(new Entry<>(key, loaded, now));
                }
            } else {
                completableValue = future.handle(handler);
            }

            try {
                value = completableValue.get();
                // check to ensure the future hasn't been completed with an exception
                if (future.isCompletedExceptionally()) {
                    future.get(); // call get to force the exception to be thrown for other concurrent callers
                    throw new IllegalStateException("the future was completed exceptionally but no exception was thrown");
                }
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
        }
        return value;
    }

    /**
     * Associates the specified value with the specified key in this map. If the map previously contained a mapping for
     * the key, the old value is replaced.
     *
     * @param key   key with which the specified value is to be associated
     * @param value value to be associated with the specified key
     */
    /**
     * @brief [Functional Utility for put]: Describe purpose here.
     */
    public void put(K key, V value) {
        long now = now();
        put(key, value, now);
    }

    /**
     * @brief [Functional Utility for put]: Describe purpose here.
     */
    private void put(K key, V value, long now) {
        CacheSegment segment = getCacheSegment(key);
        Tuple<Entry<K, V>, Entry<K, V>> tuple = segment.put(key, value, now);
        boolean replaced = false;
        lruLock.lock();
        try {
            // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            // Invariant: State condition that holds true before and after each iteration/execution
            if (tuple.v2() != null && tuple.v2().state == State.EXISTING) {
                // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                // Invariant: State condition that holds true before and after each iteration/execution
                if (unlink(tuple.v2())) {
                    replaced = true;
                }
            }
            promote(tuple.v1(), now);
        } finally {
            lruLock.unlock();
        }
        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (replaced) {
            removalListener.onRemoval(
                new RemovalNotification<>(tuple.v2().key, tuple.v2().value, RemovalNotification.RemovalReason.REPLACED)
            );
        }
    }

    /**
     * @brief [Functional Utility for notifyWithInvalidated]: Describe purpose here.
     */
    private void notifyWithInvalidated(CompletableFuture<Entry<K, V>> f) {
        try {
            Entry<K, V> entry = f.get();
            lruLock.lock();
            try {
                delete(entry, RemovalNotification.RemovalReason.INVALIDATED);
            } finally {
                lruLock.unlock();
            }
        } catch (ExecutionException e) {
            // ok
        } catch (InterruptedException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Invalidate the association for the specified key. A removal notification will be issued for invalidated
     * entries with {@link org.elasticsearch.common.cache.RemovalNotification.RemovalReason} INVALIDATED.
     *
     * @param key the key whose mapping is to be invalidated from the cache
     */
    /**
     * @brief [Functional Utility for invalidate]: Describe purpose here.
     */
    public void invalidate(K key) {
        CacheSegment segment = getCacheSegment(key);
        segment.remove(key);
    }

    /**
     * Invalidate the entry for the specified key and value. If the value provided is not equal to the value in
     * the cache, no removal will occur. A removal notification will be issued for invalidated
     * entries with {@link org.elasticsearch.common.cache.RemovalNotification.RemovalReason} INVALIDATED.
     *
     * @param key the key whose mapping is to be invalidated from the cache
     * @param value the expected value that should be associated with the key
     */
    /**
     * @brief [Functional Utility for invalidate]: Describe purpose here.
     */
    public void invalidate(K key, V value) {
        CacheSegment segment = getCacheSegment(key);
        segment.remove(key, value, true);
    }

    /**
     * Invalidate all cache entries. A removal notification will be issued for invalidated entries with
     * {@link org.elasticsearch.common.cache.RemovalNotification.RemovalReason} INVALIDATED.
     */
    /**
     * @brief [Functional Utility for invalidateAll]: Describe purpose here.
     */
    public void invalidateAll() {
        Entry<K, V> h;

        boolean[] haveSegmentLock = new boolean[NUMBER_OF_SEGMENTS];
        lruLock.lock();
        try {
            try {
                /**
                 * @brief [Functional Utility for for]: Describe purpose here.
                 */
                for (int i = 0; i < NUMBER_OF_SEGMENTS; i++) {
                    segments[i].segmentLock.writeLock().lock();
                    haveSegmentLock[i] = true;
                }
                h = head;
                /**
                 * @brief [Functional Utility for for]: Describe purpose here.
                 */
                for (CacheSegment segment : segments) {
                    segment.map = null;
                }
                Entry<K, V> current = head;
                /**
                 * @brief [Functional Utility for while]: Describe purpose here.
                 */
                while (current != null) {
                    current.state = State.DELETED;
                    current = current.after;
                }
                head = tail = null;
                count = 0;
                weight = 0;
            } finally {
                /**
                 * @brief [Functional Utility for for]: Describe purpose here.
                 */
                for (int i = NUMBER_OF_SEGMENTS - 1; i >= 0; i--) {
                    /**
                     * @brief [Functional Utility for if]: Describe purpose here.
                     */
                    if (haveSegmentLock[i]) {
                        segments[i].segmentLock.writeLock().unlock();
                    }
                }
            }
        } finally {
            lruLock.unlock();
        }
        /**
         * @brief [Functional Utility for while]: Describe purpose here.
         */
        while (h != null) {
            removalListener.onRemoval(new RemovalNotification<>(h.key, h.value, RemovalNotification.RemovalReason.INVALIDATED));
            h = h.after;
        }
    }

    /**
     * Force any outstanding size-based and time-based evictions to occur
     */
    public void refresh() {
        long now = now();
        lruLock.lock();
        try {
            evict(now);
        } finally {
            lruLock.unlock();
        }
    }

    /**
     * The number of entries in the cache.
     *
     * @return the number of entries in the cache
     */
    /**
     * @brief [Functional Utility for count]: Describe purpose here.
     */
    public int count() {
        return count;
    }

    /**
     * The weight of the entries in the cache.
     *
     * @return the weight of the entries in the cache
     */
    /**
     * @brief [Functional Utility for weight]: Describe purpose here.
     */
    public long weight() {
        return weight;
    }

    /**
     * An LRU sequencing of the keys in the cache that supports removal. This sequence is not protected from mutations
     * to the cache (except for {@link Iterator#remove()}. The result of iteration under any other mutation is
     * undefined.
     *
     * @return an LRU-ordered {@link Iterable} over the keys in the cache
     */
    /**
     * @brief [Functional Utility for keys]: Describe purpose here.
     */
    public Iterable<K> keys() {
        return () -> new Iterator<>() {
            private final CacheIterator iterator = new CacheIterator(head);

            @Override
            /**
             * @brief [Functional Utility for hasNext]: Describe purpose here.
             */
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            /**
             * @brief [Functional Utility for next]: Describe purpose here.
             */
            public K next() {
                return iterator.next().key;
            }

            @Override
            /**
             * @brief [Functional Utility for remove]: Describe purpose here.
             */
            public void remove() {
                iterator.remove();
            }
        };
    }

    /**
     * An LRU sequencing of the values in the cache. This sequence is not protected from mutations
     * to the cache (except for {@link Iterator#remove()}. The result of iteration under any other mutation is
     * undefined.
     *
     * @return an LRU-ordered {@link Iterable} over the values in the cache
     */
    /**
     * @brief [Functional Utility for values]: Describe purpose here.
     */
    public Iterable<V> values() {
        return () -> new Iterator<>() {
            private final CacheIterator iterator = new CacheIterator(head);

            @Override
            /**
             * @brief [Functional Utility for hasNext]: Describe purpose here.
             */
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            /**
             * @brief [Functional Utility for next]: Describe purpose here.
             */
            public V next() {
                return iterator.next().value;
            }

            @Override
            /**
             * @brief [Functional Utility for remove]: Describe purpose here.
             */
            public void remove() {
                iterator.remove();
            }
        };
    }

    /**
     * Performs an action for each cache entry in the cache. While iterating over the cache entries this method is protected from mutations
     * that occurs within the same cache segment by acquiring the segment's read lock during all the iteration. As such, the specified
     * consumer should not try to modify the cache. Modifications that occur in already traveled segments won't been seen by the consumer
     * but modification that occur in non yet traveled segments should be.
     *
     * @param consumer the {@link Consumer}
     */
    /**
     * @brief [Functional Utility for forEach]: Describe purpose here.
     */
    public void forEach(BiConsumer<K, V> consumer) {
        /**
         * @brief [Functional Utility for for]: Describe purpose here.
         */
        for (CacheSegment segment : segments) {
            segment.readLock.lock();
            try {
                /**
                 * @brief [Functional Utility for if]: Describe purpose here.
                 */
                if (segment.map == null) {
                    continue;
                }
                // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                // Invariant: State condition that holds true before and after each iteration/execution
                for (CompletableFuture<Entry<K, V>> future : segment.map.values()) {
                    try {
                        // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                        // Invariant: State condition that holds true before and after each iteration/execution
                        if (future != null && future.isDone()) {
                            final Entry<K, V> entry = future.get();
                            consumer.accept(entry.key, entry.value);
                        }
                    } catch (ExecutionException | InterruptedException e) {
                        throw new IllegalStateException(e);
                    }
                }
            } finally {
                segment.readLock.unlock();
            }
        }
    }

    private class CacheIterator implements Iterator<Entry<K, V>> {
        private Entry<K, V> current;
        private Entry<K, V> next;

        /**
         * @brief [Functional Utility for CacheIterator]: Describe purpose here.
         */
        CacheIterator(Entry<K, V> head) {
            current = null;
            next = head;
        }

        @Override
        /**
         * @brief [Functional Utility for hasNext]: Describe purpose here.
         */
        public boolean hasNext() {
            return next != null;
        }

        @Override
        public Entry<K, V> next() {
            current = next;
            next = next.after;
            return current;
        }

        @Override
        /**
         * @brief [Functional Utility for remove]: Describe purpose here.
         */
        public void remove() {
            Entry<K, V> entry = current;
            /**
             * @brief [Functional Utility for if]: Describe purpose here.
             */
            if (entry != null) {
                CacheSegment segment = getCacheSegment(entry.key);
                segment.remove(entry.key, entry.value, false);
                lruLock.lock();
                try {
                    current = null;
                    delete(entry, RemovalNotification.RemovalReason.INVALIDATED);
                } finally {
                    lruLock.unlock();
                }
            }
        }
    }

    /**
     * The cache statistics tracking hits, misses and evictions. These are taken on a best-effort basis meaning that
     * they could be out-of-date mid-flight.
     *
     * @return the current cache statistics
     */
    /**
     * @brief [Functional Utility for stats]: Describe purpose here.
     */
    public CacheStats stats() {
        return new CacheStats(this.hits.sum(), misses.sum(), evictions.sum());
    }

    public static class CacheStats {
        private final long hits;
        private final long misses;
        private final long evictions;

        /**
         * @brief [Functional Utility for CacheStats]: Describe purpose here.
         */
        public CacheStats(long hits, long misses, long evictions) {
            this.hits = hits;
            this.misses = misses;
            this.evictions = evictions;
        }

        /**
         * @brief [Functional Utility for getHits]: Describe purpose here.
         */
        public long getHits() {
            return hits;
        }

        /**
         * @brief [Functional Utility for getMisses]: Describe purpose here.
         */
        public long getMisses() {
            return misses;
        }

        /**
         * @brief [Functional Utility for getEvictions]: Describe purpose here.
         */
        public long getEvictions() {
            return evictions;
        }
    }

    /**
     * @brief [Functional Utility for promote]: Describe purpose here.
     */
    private void promote(Entry<K, V> entry, long now) {
        boolean promoted = true;
        lruLock.lock();
        try {
            /**
             * @brief [Functional Utility for switch]: Describe purpose here.
             */
            switch (entry.state) {
                case DELETED -> promoted = false;
                case EXISTING -> relinkAtHead(entry);
                case NEW -> linkAtHead(entry);
            }
            /**
             * @brief [Functional Utility for if]: Describe purpose here.
             */
            if (promoted) {
                evict(now);
            }
        } finally {
            lruLock.unlock();
        }
    }

    /**
     * @brief [Functional Utility for evict]: Describe purpose here.
     */
    private void evict(long now) {
        assert lruLock.isHeldByCurrentThread();

        // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        // Invariant: State condition that holds true before and after each iteration/execution
        while (tail != null && shouldPrune(tail, now)) {
            evictEntry(tail);
        }
    }

    /**
     * @brief [Functional Utility for evictEntry]: Describe purpose here.
     */
    private void evictEntry(Entry<K, V> entry) {
        assert lruLock.isHeldByCurrentThread();

        CacheSegment segment = getCacheSegment(entry.key);
        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (segment != null) {
            segment.remove(entry.key, entry.value, false);
        }
        delete(entry, RemovalNotification.RemovalReason.EVICTED);
    }

    /**
     * @brief [Functional Utility for delete]: Describe purpose here.
     */
    private void delete(Entry<K, V> entry, RemovalNotification.RemovalReason removalReason) {
        assert lruLock.isHeldByCurrentThread();

        // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        // Invariant: State condition that holds true before and after each iteration/execution
        if (unlink(entry)) {
            removalListener.onRemoval(new RemovalNotification<>(entry.key, entry.value, removalReason));
        }
    }

    /**
     * @brief [Functional Utility for shouldPrune]: Describe purpose here.
     */
    private boolean shouldPrune(Entry<K, V> entry, long now) {
        return exceedsWeight() || isExpired(entry, now);
    }

    /**
     * @brief [Functional Utility for exceedsWeight]: Describe purpose here.
     */
    private boolean exceedsWeight() {
        return maximumWeight != -1 && weight > maximumWeight;
    }

    /**
     * @brief [Functional Utility for isExpired]: Describe purpose here.
     */
    private boolean isExpired(Entry<K, V> entry, long now) {
        return (entriesExpireAfterAccess && now - entry.accessTime > expireAfterAccessNanos)
            || (entriesExpireAfterWrite && now - entry.writeTime > expireAfterWriteNanos);
    }

    /**
     * @brief [Functional Utility for unlink]: Describe purpose here.
     */
    private boolean unlink(Entry<K, V> entry) {
        assert lruLock.isHeldByCurrentThread();

        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (entry.state == State.EXISTING) {
            final Entry<K, V> before = entry.before;
            final Entry<K, V> after = entry.after;

            /**
             * @brief [Functional Utility for if]: Describe purpose here.
             */
            if (before == null) {
                // removing the head
                assert head == entry;
                head = after;
                /**
                 * @brief [Functional Utility for if]: Describe purpose here.
                 */
                if (head != null) {
                    head.before = null;
                }
            } else {
                // removing inner element
                before.after = after;
                entry.before = null;
            }

            /**
             * @brief [Functional Utility for if]: Describe purpose here.
             */
            if (after == null) {
                // removing tail
                assert tail == entry;
                tail = before;
                /**
                 * @brief [Functional Utility for if]: Describe purpose here.
                 */
                if (tail != null) {
                    tail.after = null;
                }
            } else {
                // removing inner element
                after.before = before;
                entry.after = null;
            }

            count--;
            weight -= weigher.applyAsLong(entry.key, entry.value);
            entry.state = State.DELETED;
            return true;
        } else {
            return false;
        }
    }

    /**
     * @brief [Functional Utility for linkAtHead]: Describe purpose here.
     */
    private void linkAtHead(Entry<K, V> entry) {
        assert lruLock.isHeldByCurrentThread();

        Entry<K, V> h = head;
        entry.before = null;
        entry.after = head;
        head = entry;
        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (h == null) {
            tail = entry;
        } else {
            h.before = entry;
        }

        count++;
        weight += weigher.applyAsLong(entry.key, entry.value);
        entry.state = State.EXISTING;
    }

    /**
     * @brief [Functional Utility for relinkAtHead]: Describe purpose here.
     */
    private void relinkAtHead(Entry<K, V> entry) {
        assert lruLock.isHeldByCurrentThread();

        /**
         * @brief [Functional Utility for if]: Describe purpose here.
         */
        if (head != entry) {
            unlink(entry);
            linkAtHead(entry);
        }
    }

    /**
     * @brief [Functional Utility for getCacheSegment]: Describe purpose here.
     */
    private CacheSegment getCacheSegment(K key) {
        return segments[key.hashCode() & 0xff];
    }
}
