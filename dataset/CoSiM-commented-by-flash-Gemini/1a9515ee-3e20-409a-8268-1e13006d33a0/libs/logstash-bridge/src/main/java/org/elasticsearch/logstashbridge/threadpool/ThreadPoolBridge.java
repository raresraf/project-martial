/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.logstashbridge.threadpool;

import org.elasticsearch.logstashbridge.StableBridgeAPI;
import org.elasticsearch.logstashbridge.common.SettingsBridge;
import org.elasticsearch.telemetry.metric.MeterRegistry;
import org.elasticsearch.threadpool.DefaultBuiltInExecutorBuilders;
import org.elasticsearch.threadpool.ThreadPool;

import java.util.concurrent.TimeUnit;

/**
 * An external bridge for {@link ThreadPool}
 */
public class ThreadPoolBridge extends StableBridgeAPI.ProxyInternal<ThreadPool> {

    /**
     * @brief [Functional Utility for ThreadPoolBridge]: Describe purpose here.
     * @param settingsBridge: [Description]
     * @return [ReturnType]: [Description]
     */
    public ThreadPoolBridge(final SettingsBridge settingsBridge) {
        this(new ThreadPool(settingsBridge.toInternal(), MeterRegistry.NOOP, new DefaultBuiltInExecutorBuilders()));
    }

    /**
     * @brief [Functional Utility for ThreadPoolBridge]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
    public ThreadPoolBridge(final ThreadPool delegate) {
        super(delegate);
    }

    /**
     * @brief [Functional Utility for terminate]: Describe purpose here.
     * @param pool: [Description]
     * @param timeout: [Description]
     * @param timeUnit: [Description]
     * @return [ReturnType]: [Description]
     */
    public static boolean terminate(final ThreadPoolBridge pool, final long timeout, final TimeUnit timeUnit) {
        return ThreadPool.terminate(pool.toInternal(), timeout, timeUnit);
    }

    /**
     * @brief [Functional Utility for relativeTimeInMillis]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public long relativeTimeInMillis() {
        return internalDelegate.relativeTimeInMillis();
    }

    /**
     * @brief [Functional Utility for absoluteTimeInMillis]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public long absoluteTimeInMillis() {
        return internalDelegate.absoluteTimeInMillis();
    }
}
