/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.logstashbridge.ingest;

import org.elasticsearch.ingest.IngestDocument;
import org.elasticsearch.ingest.LogstashInternalBridge;
import org.elasticsearch.logstashbridge.StableBridgeAPI;
import org.elasticsearch.logstashbridge.script.MetadataBridge;
import org.elasticsearch.logstashbridge.script.TemplateScriptBridge;

import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;

/**
 * An external bridge for {@link IngestDocument} that proxies calls through a real {@link IngestDocument}
 */
public class IngestDocumentBridge extends StableBridgeAPI.ProxyInternal<IngestDocument> {

    public static final class Constants {
        public static final String METADATA_VERSION_FIELD_NAME = IngestDocument.Metadata.VERSION.getFieldName();

        private Constants() {}
    }

    /**
     * @brief [Functional Utility for fromInternalNullable]: Describe purpose here.
     * @param ingestDocument: [Description]
     * @return [ReturnType]: [Description]
     */
    public static IngestDocumentBridge fromInternalNullable(final IngestDocument ingestDocument) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (ingestDocument == null) {
    /**
     * @brief [Functional description for field null]: Describe purpose here.
     */
            return null;
        }
        return new IngestDocumentBridge(ingestDocument);
    }

    /**
     * @brief [Functional Utility for IngestDocumentBridge]: Describe purpose here.
     * @param Map<String: [Description]
     * @param sourceAndMetadata: [Description]
     * @param Map<String: [Description]
     * @param ingestMetadata: [Description]
     * @return [ReturnType]: [Description]
     */
    public IngestDocumentBridge(final Map<String, Object> sourceAndMetadata, final Map<String, Object> ingestMetadata) {
        this(new IngestDocument(sourceAndMetadata, ingestMetadata));
    }

    /**
     * @brief [Functional Utility for IngestDocumentBridge]: Describe purpose here.
     * @param inner: [Description]
     * @return [ReturnType]: [Description]
     */
    private IngestDocumentBridge(IngestDocument inner) {
        super(inner);
    }

    /**
     * @brief [Functional Utility for getMetadata]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public MetadataBridge getMetadata() {
        return new MetadataBridge(internalDelegate.getMetadata());
    }

    public Map<String, Object> getSource() {
        return internalDelegate.getSource();
    }

    /**
     * @brief [Functional Utility for updateIndexHistory]: Describe purpose here.
     * @param index: [Description]
     * @return [ReturnType]: [Description]
     */
    public boolean updateIndexHistory(final String index) {
        return internalDelegate.updateIndexHistory(index);
    }

    /**
     * @brief [Functional Utility for getIndexHistory]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public Set<String> getIndexHistory() {
        return Set.copyOf(internalDelegate.getIndexHistory());
    }

    /**
     * @brief [Functional Utility for isReroute]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public boolean isReroute() {
        return LogstashInternalBridge.isReroute(internalDelegate);
    }

    /**
     * @brief [Functional Utility for resetReroute]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public void resetReroute() {
        LogstashInternalBridge.resetReroute(internalDelegate);
    }

    public Map<String, Object> getIngestMetadata() {
        return internalDelegate.getIngestMetadata();
    }

    public <T> T getFieldValue(final String fieldName, final Class<T> type) {
        return internalDelegate.getFieldValue(fieldName, type);
    }

    public <T> T getFieldValue(final String fieldName, final Class<T> type, final boolean ignoreMissing) {
        return internalDelegate.getFieldValue(fieldName, type, ignoreMissing);
    }

    /**
     * @brief [Functional Utility for renderTemplate]: Describe purpose here.
     * @param templateScriptFactory: [Description]
     * @return [ReturnType]: [Description]
     */
    public String renderTemplate(final TemplateScriptBridge.Factory templateScriptFactory) {
        return internalDelegate.renderTemplate(templateScriptFactory.toInternal());
    }

    /**
     * @brief [Functional Utility for setFieldValue]: Describe purpose here.
     * @param path: [Description]
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
    public void setFieldValue(final String path, final Object value) {
        internalDelegate.setFieldValue(path, value);
    }

    /**
     * @brief [Functional Utility for removeField]: Describe purpose here.
     * @param path: [Description]
     * @return [ReturnType]: [Description]
     */
    public void removeField(final String path) {
        internalDelegate.removeField(path);
    }

    /**
     * @brief [Functional Utility for executePipeline]: Describe purpose here.
     * @param pipelineBridge: [Description]
     * @param BiConsumer<IngestDocumentBridge: [Description]
     * @param handler: [Description]
     * @return [ReturnType]: [Description]
     */
    public void executePipeline(final PipelineBridge pipelineBridge, final BiConsumer<IngestDocumentBridge, Exception> handler) {
        this.internalDelegate.executePipeline(pipelineBridge.toInternal(),
                                              (ingestDocument, e) -> {
                                                  handler.accept(IngestDocumentBridge.fromInternalNullable(ingestDocument), e);
                                              });
    }
}
