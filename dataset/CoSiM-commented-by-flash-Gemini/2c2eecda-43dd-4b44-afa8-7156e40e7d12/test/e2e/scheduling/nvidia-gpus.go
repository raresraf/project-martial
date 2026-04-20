/*
Copyright 2017 The Kubernetes Authors.

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

/**
 * @2c2eecda-43dd-4b44-afa8-7156e40e7d12/test/e2e/scheduling/nvidia-gpus.go
 * @brief End-to-End (E2E) validation suite for NVIDIA GPU scheduling in Kubernetes.
 * 
 * Functional Intent: Verifies the end-to-end lifecycle of GPU-accelerated workloads 
 * on Container-Optimized OS (COS). This includes driver installation via DaemonSets, 
 * resource capacity advertising by nodes, and successful execution of CUDA-based 
 * containerized applications using both legacy and Device Plugin resource models.
 * 
 * Domain: Kubernetes Scheduling, Hardware Acceleration, NVIDIA GPU Integration.
 */

package scheduling

import (
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	testPodNamePrefix = "nvidia-gpu-"
	cosOSImage        = "Container-Optimized OS from Google"
	// Nvidia driver installation can take upwards of 5 minutes.
	driverInstallTimeout = 10 * time.Minute
)

type podCreationFuncType func() *v1.Pod

var (
	gpuResourceName v1.ResourceName
	dsYamlUrl       string
	podCreationFunc podCreationFuncType
)

/**
 * makeCudaAdditionTestPod - Constructs a pod spec for the legacy GPU resource model.
 * Logic: Defines a pod that requests a single GPU and mounts host-path NVIDIA 
 * libraries required for CUDA execution outside of the Device Plugin framework.
 */
func makeCudaAdditionTestPod() *v1.Pod {
	podName := testPodNamePrefix + string(uuid.NewUUID())
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "vector-addition",
					Image: imageutils.GetE2EImage(imageutils.CudaVectorAdd),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "nvidia-libraries",
							MountPath: "/usr/local/nvidia/lib64",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "nvidia-libraries",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/home/kubernetes/bin/nvidia/lib",
						},
					},
				},
			},
		},
	}
	return testPod
}

/**
 * makeCudaAdditionDevicePluginTestPod - Constructs a pod spec for the Device Plugin model.
 * Logic: Defines a standard pod requesting one GPU resource. Unlike the legacy 
 * model, it relies on the Device Plugin to inject necessary drivers and libraries 
 * without explicit host mounts.
 */
func makeCudaAdditionDevicePluginTestPod() *v1.Pod {
	podName := testPodNamePrefix + string(uuid.NewUUID())
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "vector-addition",
					Image: imageutils.GetE2EImage(imageutils.CudaVectorAdd),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
				},
			},
		},
	}
	return testPod
}

/**
 * areGPUsAvailableOnAllSchedulableNodes - Predicate to verify cluster readiness.
 * Logic: Iterates through all nodes in the cluster, ensuring that every 
 * schedulable node has advertised a non-zero capacity for the active GPU resource.
 */
func areGPUsAvailableOnAllSchedulableNodes(f *framework.Framework) bool {
	framework.Logf("Getting list of Nodes from API server")
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		if node.Spec.Unschedulable {
			continue
		}
		framework.Logf("gpuResourceName %s", gpuResourceName)
		if val, ok := node.Status.Capacity[gpuResourceName]; !ok || val.Value() == 0 {
			framework.Logf("Nvidia GPUs not available on Node: %q", node.Name)
			return false
		}
	}
	framework.Logf("Nvidia GPUs exist on all schedulable nodes")
	return true
}

/**
 * testNvidiaGPUsOnCOS - Orchestrates the full GPU validation cycle.
 * Logic: 
 * 1. Validates the host OS (COS required for portable CUDA host mounts).
 * 2. Deploys the driver installer DaemonSet.
 * 3. Monitors node capacity until GPUs are advertised.
 * 4. Schedules one test pod per available GPU to verify parallel execution.
 * 5. Aggregates resource usage metrics for final test summary.
 */
func testNvidiaGPUsOnCOS(f *framework.Framework) {
	// Block Logic: Initial environment validation.
	framework.Logf("Checking base image")
	if !isClusterRunningCOS(f) {
		Skip("Nvidia GPU tests are supproted only on Container Optimized OS image currently")
	}
	framework.Logf("Cluster is running on COS. Proceeding with test")

	// Block Logic: Resource model selection.
	if f.BaseName == "device-plugin-gpus" {
		dsYamlUrl = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/daemonset.yaml"
		gpuResourceName = framework.NVIDIAGPUResourceName
		podCreationFunc = makeCudaAdditionDevicePluginTestPod
	} else {
		dsYamlUrl = "https://raw.githubusercontent.com/ContainerEngine/accelerators/master/cos-nvidia-gpu-installer/daemonset.yaml"
		gpuResourceName = v1.ResourceNvidiaGPU
		podCreationFunc = makeCudaAdditionTestPod
	}

	// Block Logic: Driver deployment.
	ds, err := framework.DsFromManifest(dsYamlUrl)
	Expect(err).NotTo(HaveOccurred())
	ds.Namespace = f.Namespace.Name
	_, err = f.ClientSet.ExtensionsV1beta1().DaemonSets(f.Namespace.Name).Create(ds)
	framework.ExpectNoError(err, "failed to create daemonset")
	framework.Logf("Successfully created daemonset to install Nvidia drivers.")

	// Block Logic: Resource usage monitoring setup.
	pods, err := framework.WaitForControlledPods(f.ClientSet, ds.Namespace, ds.Name, extensionsinternal.Kind("DaemonSet"))
	framework.ExpectNoError(err, "getting pods controlled by the daemonset")
	devicepluginPods, err := framework.WaitForControlledPods(f.ClientSet, "kube-system", "nvidia-gpu-device-plugin", extensionsinternal.Kind("DaemonSet"))
	if err == nil {
		framework.Logf("Adding deviceplugin addon pod.")
		pods.Items = append(pods.Items, devicepluginPods.Items...)
	}
	framework.Logf("Starting ResourceUsageGather for the created DaemonSet pods.")
	rsgather, err := framework.NewResourceUsageGatherer(f.ClientSet, framework.ResourceGathererOptions{false, false, 2 * time.Second, 2 * time.Second, true}, pods)
	framework.ExpectNoError(err, "creating ResourceUsageGather for the daemonset pods")
	go rsgather.StartGatheringData()

	// Block Logic: Waiting for hardware readiness.
	framework.Logf("Waiting for drivers to be installed and GPUs to be available in Node Capacity...")
	Eventually(func() bool {
		return areGPUsAvailableOnAllSchedulableNodes(f)
	}, driverInstallTimeout, time.Second).Should(BeTrue())

	// Block Logic: Workload execution.
	framework.Logf("Creating as many pods as there are Nvidia GPUs and have the pods run a CUDA app")
	podList := []*v1.Pod{}
	for i := int64(0); i < getGPUsAvailable(f); i++ {
		podList = append(podList, f.PodClient().Create(podCreationFunc()))
	}
	framework.Logf("Wait for all test pods to succeed")
	for _, po := range podList {
		f.PodClient().WaitForSuccess(po.Name, 5*time.Minute)
	}

	// Block Logic: Cleanup and reporting.
	framework.Logf("Stopping ResourceUsageGather")
	constraints := make(map[string]framework.ResourceConstraint)
	summary, err := rsgather.StopAndSummarize([]int{50, 90, 100}, constraints)
	f.TestSummaries = append(f.TestSummaries, summary)
	framework.ExpectNoError(err, "getting resource usage summary")
}

var _ = SIGDescribe("[Feature:GPU]", func() {
	f := framework.NewDefaultFramework("gpus")
	It("run Nvidia GPU tests on Container Optimized OS only", func() {
		testNvidiaGPUsOnCOS(f)
	})
})

var _ = SIGDescribe("[Feature:GPUDevicePlugin]", func() {
	f := framework.NewDefaultFramework("device-plugin-gpus")
	It("run Nvidia GPU Device Plugin tests on Container Optimized OS only", func() {
		testNvidiaGPUsOnCOS(f)
	})
})
