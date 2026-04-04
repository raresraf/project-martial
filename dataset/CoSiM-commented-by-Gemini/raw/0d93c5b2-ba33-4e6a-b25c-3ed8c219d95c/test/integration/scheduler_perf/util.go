/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package benchmark provides utilities for running scheduler performance benchmarks.
// It includes functions for setting up a test environment with an apiserver and scheduler,
// creating test pods, and collecting performance metrics like scheduling latency and throughput.
package benchmark

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"path"
	"sort"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/integration/util"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	dateFormat                = "2006-01-02T15:04:05Z"
	testNamespace             = "sched-test"
	setupNamespace            = "sched-setup"
	throughputSampleFrequency = time.Second
)

var dataItemsDir = flag.String("data-items-dir", "", "destination directory for storing generated data items for perf dashboard")

// mustSetupScheduler starts a Kubernetes API server and a scheduler for integration testing.
// It configures a high-throughput client rate limiter.
// It returns a shutdown function to clean up the started components, a pod informer, and a clientset.
func mustSetupScheduler() (util.ShutdownFunc, coreinformers.PodInformer, clientset.Interface) {
	// Functional Utility: Start an isolated API server for the test.
	apiURL, apiShutdown := util.StartApiserver()
	// Functional Utility: Create a high QPS Kubernetes client to avoid client-side throttling.
	clientSet := clientset.NewForConfigOrDie(&restclient.Config{
		Host:          apiURL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}},
		QPS:           5000.0,
		Burst:         5000,
	})
	// Functional Utility: Start an isolated scheduler instance connected to the test API server.
	_, podInformer, schedulerShutdown := util.StartScheduler(clientSet)
	// Functional Utility: Start a fake controller for PersistentVolumes to satisfy pod claims.
	fakePVControllerShutdown := util.StartFakePVController(clientSet)

	// Invariant: The shutdown function ensures that all test components are terminated in reverse order of creation.
	shutdownFunc := func() {
		fakePVControllerShutdown()
		schedulerShutdown()
		apiShutdown()
	}

	return shutdownFunc, podInformer, clientSet
}

// getScheduledPods returns a list of pods that have been successfully assigned to a node.
// It filters pods from a set of specified namespaces. If no namespaces are provided, it checks all namespaces.
// The pod informer provides a cached, efficient way to access pod state.
func getScheduledPods(podInformer coreinformers.PodInformer, namespaces ...string) ([]*v1.Pod, error) {
	pods, err := podInformer.Lister().List(labels.Everything())
	if err != nil {
		return nil, err
	}

	s := sets.NewString(namespaces...)
	scheduled := make([]*v1.Pod, 0, len(pods))
	// Block Logic: Iterate through all pods in the cache.
	for i := range pods {
		pod := pods[i]
		// Pre-condition: A pod is considered scheduled if its NodeName field is not empty.
		// It must also belong to one of the specified namespaces if any were provided.
		if len(pod.Spec.NodeName) > 0 && (len(s) == 0 || s.Has(pod.Namespace)) {
			scheduled = append(scheduled, pod)
		}
	}
	return scheduled, nil
}

// DataItem represents a single data point for performance analysis, structured for dashboard consumption.
type DataItem struct {
	// Data is a map from a metric's percentile/aggregation to its value (e.g., "Perc90" -> 23.5).
	Data map[string]float64 `json:"data"`
	// Unit is the unit of measurement for the data (e.g., "ms", "pods/s").
	Unit string `json:"unit"`
	// Labels provide dimensions for the data, such as the test name or metric type.
	Labels map[string]string `json:"labels,omitempty"`
}

// DataItems is a collection of DataItem objects, conforming to the expected format for performance dashboards.
type DataItems struct {
	// Version of the data format.
	Version   string     `json:"version"`
	// DataItems is the list of individual data points.
	DataItems []DataItem `json:"dataItems"`
}

// makeBasePod creates a basic Pod object with a standard specification, to be used as a template in benchmark tests.
func makeBasePod() *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pod-",
		},
		Spec: testutils.MakePodSpec(),
	}
	return basePod
}

// dataItems2JSONFile serializes a DataItems struct to a JSON file.
// The filename is prefixed with a given name and a timestamp.
func dataItems2JSONFile(dataItems DataItems, namePrefix string) error {
	b, err := json.Marshal(dataItems)
	if err != nil {
		return err
	}

	destFile := fmt.Sprintf("%v_%v.json", namePrefix, time.Now().Format(dateFormat))
	if *dataItemsDir != "" {
		destFile = path.Join(*dataItemsDir, destFile)
	}

	return ioutil.WriteFile(destFile, b, 0644)
}

// metricsCollectorConfig holds the configuration for which metrics to collect.
type metricsCollectorConfig struct {
	Metrics []string
}

// metricsCollector is responsible for gathering and processing histogram-based metrics
// from the default Prometheus registry.
type metricsCollector struct {
	*metricsCollectorConfig
	labels map[string]string
}

// newMetricsCollector creates a new metricsCollector.
func newMetricsCollector(config *metricsCollectorConfig, labels map[string]string) *metricsCollector {
	return &metricsCollector{
		metricsCollectorConfig: config,
		labels:                 labels,
	}
}

// run is a placeholder, as this collector operates synchronously on demand.
func (*metricsCollector) run(ctx context.Context) {
	// This collector is synchronous and does not require a background worker.
}

// collect gathers data for all configured metrics.
func (pc *metricsCollector) collect() []DataItem {
	var dataItems []DataItem
	// Block Logic: Iterate over each metric name specified in the configuration.
	for _, metric := range pc.Metrics {
		dataItem := collectHistogram(metric, pc.labels)
		if dataItem != nil {
			dataItems = append(dataItems, *dataItem)
		}
	}
	return dataItems
}

// collectHistogram fetches a single histogram metric from the Prometheus gatherer,
// calculates percentiles, and formats it as a DataItem.
func collectHistogram(metric string, labels map[string]string) *DataItem {
	// Functional Utility: Scrape the Prometheus registry for the specified metric.
	hist, err := testutil.GetHistogramFromGatherer(legacyregistry.DefaultGatherer, metric)
	if err != nil {
		klog.Error(err)
		return nil
	}
	if hist.Histogram == nil {
		klog.Errorf("metric %q is not a Histogram metric", metric)
		return nil
	}
	if err := hist.Validate(); err != nil {
		klog.Error(err)
		return nil
	}

	// Algorithm: Calculate 50th, 90th, and 99th percentiles from the histogram data.
	q50 := hist.Quantile(0.50)
	q90 := hist.Quantile(0.90)
	q99 := hist.Quantile(0.95) // Note: This is actually the 95th percentile.
	avg := hist.Average()

	// Invariant: Clear the metric after collection to ensure test isolation.
	hist.Clear()

	msFactor := float64(time.Second) / float64(time.Millisecond)

	// Copy labels and add a specific "Metric" label.
	labelMap := map[string]string{"Metric": metric}
	for k, v := range labels {
		labelMap[k] = v
	}
	// Functional Utility: Assemble the final data structure for dashboard ingestion.
	return &DataItem{
		Labels: labelMap,
		Data: map[string]float64{
			"Perc50":  q50 * msFactor,
			"Perc90":  q90 * msFactor,
			"Perc99":  q99 * msFactor,
			"Average": avg * msFactor,
		},
		Unit: "ms",
	}
}

// throughputCollector measures scheduling throughput in pods per second.
type throughputCollector struct {
	podInformer           coreinformers.PodInformer
	schedulingThroughputs []float64
	labels                map[string]string
	namespaces            []string
}

// newThroughputCollector creates a new throughputCollector.
func newThroughputCollector(podInformer coreinformers.PodInformer, labels map[string]string, namespaces []string) *throughputCollector {
	return &throughputCollector{
		podInformer: podInformer,
		labels:      labels,
		namespaces:  namespaces,
	}
}

// run periodically samples the number of scheduled pods to calculate throughput.
// It runs as a background worker until the provided context is canceled.
func (tc *throughputCollector) run(ctx context.Context) {
	podsScheduled, err := getScheduledPods(tc.podInformer, tc.namespaces...)
	if err != nil {
		klog.Fatalf("%v", err)
	}
	lastScheduledCount := len(podsScheduled)
	ticker := time.NewTicker(throughputSampleFrequency)
	defer ticker.Stop()

	// Block Logic: The main sampling loop.
	for {
		select {
		case <-ctx.Done():
			// Pre-condition: Stop sampling when the context is canceled.
			return
		case <-ticker.C:
			// Invariant: At each tick, calculate the number of newly scheduled pods.
			podsScheduled, err := getScheduledPods(tc.podInformer, tc.namespaces...)
			if err != nil {
				klog.Fatalf("%v", err)
			}

			scheduled := len(podsScheduled)
			samplingRatioSeconds := float64(throughputSampleFrequency) / float64(time.Second)
			// Algorithm: Throughput is the change in scheduled pods over the sampling interval.
			throughput := float64(scheduled-lastScheduledCount) / samplingRatioSeconds
			tc.schedulingThroughputs = append(tc.schedulingThroughputs, throughput)
			lastScheduledCount = scheduled

			klog.Infof("%d pods scheduled", lastScheduledCount)
		}
	}
}

// collect processes the raw throughput samples and calculates summary statistics.
func (tc *throughputCollector) collect() []DataItem {
	throughputSummary := DataItem{Labels: tc.labels}
	if length := len(tc.schedulingThroughputs); length > 0 {
		sort.Float64s(tc.schedulingThroughputs)
		sum := 0.0
		for i := range tc.schedulingThroughputs {
			sum += tc.schedulingThroughputs[i]
		}

		throughputSummary.Labels["Metric"] = "SchedulingThroughput"
		// Algorithm: Calculate average and percentiles from the sorted list of throughput samples.
		throughputSummary.Data = map[string]float64{
			"Average": sum / float64(length),
			"Perc50":  tc.schedulingThroughputs[int(math.Ceil(float64(length*50)/100))-1],
			"Perc90":  tc.schedulingThroughputs[int(math.Ceil(float64(length*90)/100))-1],
			"Perc99":  tc.schedulingThroughputs[int(math.Ceil(float64(length*99)/100))-1],
		}
		throughputSummary.Unit = "pods/s"
	}

	return []DataItem{throughputSummary}
}
