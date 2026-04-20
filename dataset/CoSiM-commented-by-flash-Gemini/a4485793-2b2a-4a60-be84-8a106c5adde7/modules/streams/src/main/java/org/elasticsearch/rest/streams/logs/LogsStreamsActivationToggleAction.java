/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.rest.streams.logs;

import org.elasticsearch.action.ActionType;
import org.elasticsearch.action.support.master.AcknowledgedRequest;
import org.elasticsearch.action.support.master.AcknowledgedResponse;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.tasks.CancellableTask;
import org.elasticsearch.tasks.Task;
import org.elasticsearch.tasks.TaskId;

import java.io.IOException;
import java.util.Map;

/**
 * @a4485793-2b2a-4a60-be84-8a106c5adde7/modules/streams/src/main/java/org/elasticsearch/rest/streams/logs/LogsStreamsActivationToggleAction.java
 * @brief Action definition for toggling the activation state of log streams.
 * 
 * Functional Intent: Defines the administrative action used to enable or disable 
 * automated log stream management within the cluster. It provides a standard 
 * request/response pattern for cluster-level state changes.
 */
public class LogsStreamsActivationToggleAction {

    public static ActionType<AcknowledgedResponse> INSTANCE = new ActionType<>("cluster:admin/streams/logs/toggle");

    /**
     * @brief Request object for the logs stream toggle action.
     * 
     * Functional Intent: Encapsulates the desired state (enabled/disabled) and 
     * required timeouts for the master node and acknowledgement.
     */
    public static class Request extends AcknowledgedRequest<Request> {

        private final boolean enable;

        /**
         * @brief Constructs a new request.
         * @param masterNodeTimeout Timeout for the master node to process the request.
         * @param ackTimeout Timeout for receiving acknowledgement from all required nodes.
         * @param enable True to enable log streams, false to disable.
         */
        public Request(TimeValue masterNodeTimeout, TimeValue ackTimeout, boolean enable) {
            super(masterNodeTimeout, ackTimeout);
            this.enable = enable;
        }

        /**
         * @brief Deserializes a request from a StreamInput.
         * @param in The stream to read from.
         * @throws IOException If a serialization error occurs.
         */
        public Request(StreamInput in) throws IOException {
            super(in);
            this.enable = in.readBoolean();
        }

        /**
         * Block Logic: Serializes the request state to a StreamOutput.
         * Logic: Writes the base acknowledged request parameters followed by the 'enable' boolean flag.
         */
        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeBoolean(enable);
        }

        @Override
        public String toString() {
            return "LogsStreamsActivationToggleAction.Request{" + "enable=" + enable + '}';
        }

        /**
         * Functional Utility: Accessor for the desired activation state.
         * @return True if the request intent is to enable log streams.
         */
        public boolean shouldEnable() {
            return enable;
        }

        /**
         * Block Logic: Creates a task instance to represent this asynchronous request.
         * Logic: Returns a CancellableTask initialized with the provided TaskId's ID 
         * and other metadata, identifying it as a log streams activation toggle request.
         */
        @Override
        public Task createTask(TaskId taskId, String type, String action, TaskId parentTaskId, Map<String, String> headers) {
            return new CancellableTask(taskId.getId(), type, action, "Logs streams activation toggle request", parentTaskId, headers);
        }
    }
}
