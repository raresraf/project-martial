/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.lookup;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @brief Functional description of the SourceProviderTests class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class SourceProviderTests extends ESTestCase {

    public void testStoredFieldsSourceProvider() throws IOException {
        try (Directory dir = newDirectory(); RandomIndexWriter iw = new RandomIndexWriter(random(), dir)) {
            Document doc = new Document();
            doc.add(new StringField("field", "value", Field.Store.YES));
            doc.add(new StoredField("_source", new BytesRef("{\"field\": \"value\"}")));
            iw.addDocument(doc);

            try (IndexReader reader = iw.getReader()) {
                LeafReaderContext readerContext = reader.leaves().get(0);

                SourceProvider sourceProvider = SourceProvider.fromStoredFields();
                Source source = sourceProvider.getSource(readerContext, 0);

                assertNotNull(source.internalSourceRef());

                // Source should be preserved if we pass in the same reader and document
                Source s2 = sourceProvider.getSource(readerContext, 0);
                assertSame(s2, source);
            }
        }
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void testConcurrentStoredFieldsSourceProvider() throws IOException {
        int numDocs = 350;
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergePolicy(NoMergePolicy.INSTANCE);

        ExecutorService executorService = Executors.newFixedThreadPool(4);
        try (Directory dir = newDirectory(); RandomIndexWriter iw = new RandomIndexWriter(random(), dir, iwc)) {

            Document doc = new Document();
            for (int i = 0; i < numDocs; i++) {
                doc.clear();
                doc.add(new StoredField("_source", new BytesRef("{ \"id\" : " + i + "}")));
                iw.addDocument(doc);
                if (random().nextInt(35) == 7) {
                    iw.commit();
                }
            }
            iw.commit();

            IndexReader reader = iw.getReader();
            IndexSearcher searcher = new IndexSearcher(reader, executorService);

            int numIterations = 20;
            for (int i = 0; i < numIterations; i++) {
                searcher.search(new MatchAllDocsQuery(), assertingCollectorManager());
            }

            reader.close();
        }
        executorService.shutdown();
    }

    private static class SourceAssertingCollector implements Collector {

        final SourceProvider sourceProvider;

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param sourceProvider: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
        private SourceAssertingCollector(SourceProvider sourceProvider) {
            this.sourceProvider = sourceProvider;
        }

        @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param context: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
        public LeafCollector getLeafCollector(LeafReaderContext context) {
            return new LeafCollector() {
                @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param scorer: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
                public void setScorer(Scorable scorer) {

                }

                @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param doc: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
                public void collect(int doc) throws IOException {
                    Source source = sourceProvider.getSource(context, doc);
                    assertEquals(doc + context.docBase, source.source().get("id"));
                }
            };
        }

        @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
        public ScoreMode scoreMode() {
            return ScoreMode.COMPLETE;
        }
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    private static CollectorManager<SourceAssertingCollector, ?> assertingCollectorManager() {
        SourceProvider sourceProvider = SourceProvider.fromStoredFields();
        return new CollectorManager<>() {
            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public SourceAssertingCollector newCollector() {
                return new SourceAssertingCollector(sourceProvider);
            }

            @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param collectors: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
            public Object reduce(Collection<SourceAssertingCollector> collectors) {
                return 0;
            }
        };
    }
}
