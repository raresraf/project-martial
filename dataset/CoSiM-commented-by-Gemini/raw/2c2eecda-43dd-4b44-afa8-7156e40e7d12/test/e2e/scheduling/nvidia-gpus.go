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

// This file contains end-to-end tests for scheduling pods with NVIDIA GPU resources.
// It ensures that Kubernetes can correctly schedule pods requesting GPUs on nodes
// that have been properly configured with NVIDIA drivers.
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
	// cosOSImage identifies the Container-Optimized OS, a specific environment this test is designed for.
	cosOSImage = "Container-Optimized OS from Google"
	// driverInstallTimeout defines the maximum time to wait for the NVIDIA driver installation to complete on the nodes.
	driverInstallTimeout = 10 * time.Minute
)

// podCreationFuncType is a function signature for creating a test pod.
// This allows for different pod configurations (e.g., for legacy vs. device plugin tests).
type podCreationFuncType func() *v1.Pod

var (
	// gpuResourceName holds the name of the GPU resource to be requested by pods.
	// This varies depending on the GPU discovery mechanism (e.g., "nvidia.com/gpu" or "alpha.kubernetes.io/nvidia-gpu").
	gpuResourceName v1.ResourceName
	// dsYamlUrl is the URL for the DaemonSet manifest used to install NVIDIA drivers and device plugins.
	dsYamlUrl string
	// podCreationFunc is a function that returns a pod definition for the test.
	podCreationFunc podCreationFuncType
)

// makeCudaAdditionTestPod creates a pod that runs a simple CUDA vector addition program.
// This version is for the legacy GPU integration, which requires manually mounting NVIDIA libraries via a HostPath volume.
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
							// This pod requests one GPU.
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
						// The path to the NVIDIA libraries on the host machine.
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

// makeCudaAdditionDevicePluginTestPod creates a pod for the NVIDIA device plugin framework.
// Unlike the legacy method, it does not require manual volume mounts for libraries,
// as the device plugin is responsible for exposing the device and its drivers to the container.
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
							// This pod requests one GPU.
							gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
				},
			},
		},
	}
	return testPod
}

// isClusterRunningCOS checks if all schedulable nodes in the cluster are running Container-Optimized OS (COS).
// The test is specific to COS due to the driver installation method and library paths.
func isClusterRunningCOS(f *framework.Framework) bool {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	// Pre-condition: At least one schedulable node exists.
	// Invariant: Iterates through nodes, returns false if any node is not running COS.
	for _, node := range nodeList.Items {
		if !strings.Contains(node.Status.NodeInfo.OSImage, cosOSImage) {
			return false
		}
	}
	return true
}

// areGPUsAvailableOnAllSchedulableNodes checks if all schedulable nodes in the cluster
// have reported GPU capacity, indicating that the drivers are installed and recognized by Kubernetes.
func areGPUsAvailableOnAllSchedulableNodes(f *framework.Framework) bool {
	framework.Logf("Getting list of Nodes from API server")
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	// Pre-condition: Assumes nodes are present in the cluster.
	// Invariant: Iterates through schedulable nodes, returns false if any node lacks GPU capacity.
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

// getGPUsAvailable returns the total number of GPUs available across all nodes in the cluster.
func getGPUsAvailable(f *framework.Framework) int64 {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	var gpusAvailable int64
	for _, node := range nodeList.Items {
		if val, ok := node.Status.Capacity[gpuResourceName]; ok {
			gpusAvailable += (&val).Value()
		}
	}
	return gpusAvailable
}

// testNvidiaGPUsOnCOS is the core test logic for verifying NVIDIA GPU functionality.
// It orchestrates driver installation, waits for GPUs to become available, and runs test pods.
func testNvidiaGPUsOnCOS(f *framework.Framework) {
	// Skip the test if the base image is not COS, as the test is tailored to its environment.
	framework.Logf("Checking base image")
	if !isClusterRunningCOS(f) {
		Skip("Nvidia GPU tests are supproted only on Container Optimized OS image currently")
	}
	framework.Logf("Cluster is running on COS. Proceeding with test")

	// Block Logic: Configure test parameters based on whether it's a device-plugin test or a legacy test.
	// This sets the appropriate DaemonSet URL, GPU resource name, and pod creation function.
	if f.BaseName == "device-plugin-gpus" {
		dsYamlUrl = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/daemonset.yaml"
		gpuResourceName = framework.NVIDIAGPUResourceName
		podCreationFunc = makeCudaAdditionDevicePluginTestPod
	} else {
		dsYamlUrl = "https://raw.githubusercontent.com/ContainerEngine/accelerators/master/cos-nvidia-gpu-installer/daemonset.yaml"
		gpuResourceName = v1.ResourceNvidiaGPU
		podCreationFunc = makeCudaAdditionTestPod
	}

	// Creates a DaemonSet from a manifest URL to install Nvidia drivers on all nodes.
	// For device plugin tests, this DaemonSet also runs the NVIDIA device plugin.
	ds, err := framework.DsFromManifest(dsYamlUrl)
	Expect(err).NotTo(HaveOccurred())
	ds.Namespace = f.Namespace.Name
	_, err = f.ClientSet.ExtensionsV1beta1().DaemonSets(f.Namespace.Name).Create(ds)
	framework.ExpectNoError(err, "failed to create daemonset")
	framework.Logf("Successfully created daemonset to install Nvidia drivers.")

	// Wait for the DaemonSet pods to be created and running.
	pods, err := framework.WaitForControlledPods(f.ClientSet, ds.Namespace, ds.Name, extensionsinternal.Kind("DaemonSet"))
	framework.ExpectNoError(err, "getting pods controlled by the daemonset")
	// In some setups, the device plugin might run as a separate DaemonSet in the kube-system namespace.
	devicepluginPods, err := framework.WaitForControlledPods(f.ClientSet, "kube-system", "nvidia-gpu-device-plugin", extensionsinternal.Kind("DaemonSet"))
	if err == nil {
		framework.Logf("Adding deviceplugin addon pod.")
		pods.Items = append(pods.Items, devicepluginPods.Items...)
	}
	// Start gathering resource usage data from the driver installation pods.
	framework.Logf("Starting ResourceUsageGather for the created DaemonSet pods.")
	rsgather, err := framework.NewResourceUsageGatherer(f.ClientSet, framework.ResourceGathererOptions{false, false, 2 * time.Second, 2 * time.Second, true}, pods)
	framework.ExpectNoError(err, "creating ResourceUsageGather for the daemonset pods")
	go rsgather.StartGatheringData()

	// Block Logic: Wait for the driver installation to complete and for Kubernetes to recognize the GPUs.
	// This is verified by checking the node's capacity in the API server.
	framework.Logf("Waiting for drivers to be installed and GPUs to be available in Node Capacity...")
	Eventually(func() bool {
		return areGPUsAvailableOnAllSchedulableNodes(f)
	}, driverInstallTimeout, time.Second).Should(BeTrue())

	// Block Logic: Create and run one test pod for each available GPU to verify functionality.
	framework.Logf("Creating as many pods as there are Nvidia GPUs and have the pods run a CUDA app")
	podList := []*v1.Pod{}
	for i := int64(0); i < getGPUsAvailable(f); i++ {
		podList = append(podList, f.PodClient().Create(podCreationFunc()))
	}
	// Wait for all CUDA test pods to complete successfully.
	framework.Logf("Wait for all test pods to succeed")
	for _, po := range podList {
		f.PodClient().WaitForSuccess(po.Name, 5*time.Minute)
	}

	// Stop the resource usage gatherer and summarize the data.
	framework.Logf("Stopping ResourceUsageGather")
	constraints := make(map[string]framework.ResourceConstraint)
	// For now, it just gets a summary. In the future, specific resource constraints could be asserted.
	summary, err := rsgather.StopAndSummarize([]int{50, 90, 100}, constraints)
	f.TestSummaries = append(f.TestSummaries, summary)
	framework.ExpectNoError(err, "getting resource usage summary")
}

// Defines a test suite for the GPU feature.
var _ = SIGDescribe("[Feature:GPU]", func() {
	f := framework.NewDefaultFramework("gpus")
	// Defines a single test case within the suite.
	It("run Nvidia GPU tests on Container Optimized OS only", func() {
		testNvidiaGPUsOnCOS(f)
	})
})

// Defines a test suite for the GPU device plugin feature.
var _ = SIGDescribe("[Feature:GPUDevicePlugin]", func() {
	f := framework.NewDefaultFramework("device-plugin-gpus")
	It("run Nvidia GPU Device Plugin tests on Container Optimized OS only", func() {
		testNvidiaGPUsOnCOS(f)
	})
})
