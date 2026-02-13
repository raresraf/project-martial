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

// Package benchmark provides utility functions and data structures for performance benchmarking
// of the Kubernetes scheduler. It includes functionalities for setting up a test environment,
// collecting and processing various performance metrics, and serializing the results for
// integration with performance dashboards.
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
	// dateFormat specifies the format for timestamps used in naming output files.
	dateFormat                = "2006-01-02T15:04:05Z"
	// testNamespace is the Kubernetes namespace used for scheduling performance tests.
	testNamespace             = "sched-test"
	// setupNamespace is the Kubernetes namespace used for setting up test resources.
	setupNamespace            = "sched-setup"
	// throughputSampleFrequency defines how often the scheduling throughput is sampled.
	throughputSampleFrequency = time.Second
)

// dataItemsDir is a command-line flag to specify the destination directory for storing
// generated performance data items, which are formatted for the perf dashboard.
var dataItemsDir = flag.String("data-items-dir", "", "destination directory for storing generated data items for perf dashboard")

// mustSetupScheduler initializes and starts the necessary components for scheduler benchmarking.
// It sets up the Kubernetes API server and the scheduler itself.
// A clientSet configured for high throughput (QPS: 5000, Burst: 5000) is returned, along with
// a podInformer for monitoring pod events and a shutdown function to gracefully terminate all started components.
func mustSetupScheduler() (util.ShutdownFunc, coreinformers.PodInformer, clientset.Interface) {
	apiURL, apiShutdown := util.StartApiserver()
	clientSet := clientset.NewForConfigOrDie(&restclient.Config{
		Host:          apiURL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}},
		QPS:           5000.0,
		Burst:         5000,
	})
	_, podInformer, schedulerShutdown := util.StartScheduler(clientSet)
	fakePVControllerShutdown := util.StartFakePVController(clientSet)

	// shutdownFunc orchestrates the graceful termination of all initiated Kubernetes components.
	shutdownFunc := func() {
		fakePVControllerShutdown()
		schedulerShutdown()
		apiShutdown()
	}

	return shutdownFunc, podInformer, clientSet
}

// getScheduledPods retrieves a list of pods that have been scheduled to a node.
// It filters pods based on whether their NodeName field is populated.
// If namespaces are provided, it further filters by those specific namespaces;
// otherwise, it returns scheduled pods from all namespaces.
func getScheduledPods(podInformer coreinformers.PodInformer, namespaces ...string) ([]*v1.Pod, error) {
	pods, err := podInformer.Lister().List(labels.Everything())
	if err != nil {
		return nil, err
	}

	s := sets.NewString(namespaces...)
	scheduled := make([]*v1.Pod, 0, len(pods))
	// Iterates through all pods and appends those that have been assigned a node (scheduled)
	// and optionally match the specified namespaces.
	for i := range pods {
		pod := pods[i]
		if len(pod.Spec.NodeName) > 0 && (len(s) == 0 || s.Has(pod.Namespace)) {
			scheduled = append(scheduled, pod)
		}
	}
	return scheduled, nil
}

// DataItem represents a single data point for the performance dashboard.
// It encapsulates the actual metric data (e.g., percentiles, average), its unit, and associated labels.
type DataItem struct {
	// Data is a map from bucket to real data point (e.g. "Perc90" -> 23.5). Notice
	// that all data items with the same label combination should have the same buckets.
	Data map[string]float64 `json:"data"`
	// Unit is the data unit. Notice that all data items with the same label combination
	// should have the same unit.
	Unit string `json:"unit"`
	// Labels is the labels of the data item.
	Labels map[string]string `json:"labels,omitempty"`
}

// DataItems is a collection of DataItem objects, structured in a format
// expected by the performance dashboard for comprehensive metric reporting.
type DataItems struct {
	// Version indicates the version of the DataItems structure.
	Version   string     `json:"version"`
	// DataItems holds an array of individual performance metric data points.
	DataItems []DataItem `json:"dataItems"`
}

// makeBasePod creates a basic Pod object with a generated name and a default
// pod specification, intended to serve as a template for test pods.
func makeBasePod() *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pod-",
		},
		Spec: testutils.MakePodSpec(),
	}
	return basePod
}

// dataItems2JSONFile serializes the provided DataItems structure into a JSON file.
// The filename is generated using a specified prefix and the current timestamp,
// and the file is saved to the directory indicated by the `dataItemsDir` flag.
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

// metricsCollectorConfig defines the configuration for which metrics to collect.
// This structure is designed to be marshaled into a YAML configuration file.
type metricsCollectorConfig struct {
	// Metrics is a list of metric names to be collected by the metricsCollector.
	Metrics []string
}

// metricsCollector is responsible for collecting histogram metrics from the
// legacyregistry.DefaultGatherer endpoint. It currently only supports Histogram type metrics.
type metricsCollector struct {
	*metricsCollectorConfig
	labels map[string]string
}

// newMetricsCollector creates and returns a new instance of metricsCollector
// with the given configuration and labels.
func newMetricsCollector(config *metricsCollectorConfig, labels map[string]string) *metricsCollector {
	return &metricsCollector{
		metricsCollectorConfig: config,
		labels:                 labels,
	}
}

// run is a no-op for metricsCollector as metric collection typically happens
// synchronously during test execution and doesn't require a separate goroutine
// to start beforehand.
func (*metricsCollector) run(ctx context.Context) {
	// metricCollector doesn't need to start before the tests, so nothing to do here.
}

// collect gathers the specified metrics and transforms them into a slice of DataItem.
// It iterates through the configured metrics, calling collectHistogram for each.
func (pc *metricsCollector) collect() []DataItem {
	var dataItems []DataItem
	// Block Logic: Iterates over each configured metric to collect its histogram data.
	for _, metric := range pc.Metrics {
		dataItem := collectHistogram(metric, pc.labels)
		if dataItem != nil {
			dataItems = append(dataItems, *dataItem)
		}
	}
	return dataItems
}

// collectHistogram retrieves a histogram metric from the default Prometheus gatherer,
// calculates key quantiles (50th, 90th, 99th percentile) and the average,
// converts the values to milliseconds, clears the metric, and returns it as a DataItem.
// This function also handles error logging if the metric is not found or is not a histogram.
func collectHistogram(metric string, labels map[string]string) *DataItem {
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

	q50 := hist.Quantile(0.50)
	q90 := hist.Quantile(0.90)
	q99 := hist.Quantile(0.95)
	avg := hist.Average()

	// Invariant: Metrics are cleared after collection to ensure subsequent tests
	// start with a clean slate, preventing data contamination between runs.
	hist.Clear()

	msFactor := float64(time.Second) / float64(time.Millisecond)

	// Copy labels and add "Metric" label for this metric.
	labelMap := map[string]string{"Metric": metric}
	// Block Logic: Merges existing labels with a new "Metric" label.
	for k, v := range labels {
		labelMap[k] = v
	}
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

// throughputCollector is responsible for periodically sampling and collecting
// scheduling throughput by monitoring the number of scheduled pods over time.
type throughputCollector struct {
	podInformer           coreinformers.PodInformer
	schedulingThroughputs []float64
	labels                map[string]string
	namespaces            []string
}

// newThroughputCollector creates and returns a new instance of throughputCollector
// initialized with a pod informer, labels, and target namespaces.
func newThroughputCollector(podInformer coreinformers.PodInformer, labels map[string]string, namespaces []string) *throughputCollector {
	return &throughputCollector{
		podInformer: podInformer,
		labels:      labels,
		namespaces:  namespaces,
	}
}

// run continuously monitors the number of scheduled pods at a fixed frequency
// and records the scheduling throughput. It runs in a goroutine and stops
// when the provided context is cancelled.
func (tc *throughputCollector) run(ctx context.Context) {
	podsScheduled, err := getScheduledPods(tc.podInformer, tc.namespaces...)
	if err != nil {
		klog.Fatalf("%v", err)
	}
	lastScheduledCount := len(podsScheduled)
	// Block Logic: Continuously samples the number of scheduled pods and calculates throughput.
	// Pre-condition: An active context for cancellation and a configured throughputSampleFrequency.
	// Invariant: `lastScheduledCount` always reflects the count of scheduled pods from the previous sample.
	for {
		select {
		case <-ctx.Done():
			return
		// TODO(#94665): use time.Ticker instead
		case <-time.After(throughputSampleFrequency):
			podsScheduled, err := getScheduledPods(tc.podInformer, tc.namespaces...)
			if err != nil {
				klog.Fatalf("%v", err)
			}

			scheduled := len(podsScheduled)
			samplingRatioSeconds := float64(throughputSampleFrequency) / float64(time.Second)
			throughput := float64(scheduled-lastScheduledCount) / samplingRatioSeconds
			tc.schedulingThroughputs = append(tc.schedulingThroughputs, throughput)
			lastScheduledCount = scheduled

			klog.Infof("%d pods scheduled", lastScheduledCount)
		}
	}
}

// collect processes the raw scheduling throughput data, calculates statistical
// summaries (average, 50th, 90th, 99th percentiles), and returns them as a DataItem.
// Pre-condition: `tc.schedulingThroughputs` contains collected throughput samples.
func (tc *throughputCollector) collect() []DataItem {
	throughputSummary := DataItem{Labels: tc.labels}
	if length := len(tc.schedulingThroughputs); length > 0 {
		sort.Float64s(tc.schedulingThroughputs)
		sum := 0.0
		// Block Logic: Calculates the sum of all collected throughput samples.
		for i := range tc.schedulingThroughputs {
			sum += tc.schedulingThroughputs[i]
		}

		throughputSummary.Labels["Metric"] = "SchedulingThroughput"
		throughputSummary.Data = map[string]float64{
			"Average": sum / float64(length),
			// Inline: Calculates the 50th percentile (median) of the scheduling throughput.
			"Perc50":  tc.schedulingThroughputs[int(math.Ceil(float64(length*50)/100))-1],
			// Inline: Calculates the 90th percentile of the scheduling throughput.
			"Perc90":  tc.schedulingThroughputs[int(math.Ceil(float64(length*90)/100))-1],
			// Inline: Calculates the 99th percentile of the scheduling throughput.
			"Perc99":  tc.schedulingThroughputs[int(math.Ceil(float64(length*99)/100))-1],
		}
		throughputSummary.Unit = "pods/s"
	}

	return []DataItem{throughputSummary}
}
