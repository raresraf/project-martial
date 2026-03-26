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

// Package scheduling provides end-to-end tests for Kubernetes scheduling
// functionalities, specifically focusing on Nvidia GPU integration on
// Container-Optimized OS. These tests validate the proper allocation,
// discovery, and utilization of GPU resources within a Kubernetes cluster.
package scheduling

import (
	"os"
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
	// testPodNamePrefix defines the prefix for pod names created during GPU tests.
	testPodNamePrefix = "nvidia-gpu-"
	// cosOSImage is the expected OS image string for Container-Optimized OS.
	cosOSImage = "Container-Optimized OS from Google"
	// driverInstallTimeout specifies the maximum time allowed for Nvidia driver installation.
	// Nvidia driver installation can take upwards of 5 minutes.
	driverInstallTimeout = 10 * time.Minute
)

// podCreationFuncType defines a function signature for creating a Kubernetes Pod object.
type podCreationFuncType func() *v1.Pod

var (
	// gpuResourceName stores the Kubernetes resource name for GPUs, either
	// v1.ResourceNvidiaGPU or framework.NVIDIAGPUResourceName based on the test feature.
	gpuResourceName v1.ResourceName
	// dsYamlUrl stores the URL for the DaemonSet YAML file used to install Nvidia drivers.
	dsYamlUrl string
	// podCreationFunc stores the function used to create a test pod,
	// which varies based on whether device plugin tests are enabled.
	podCreationFunc podCreationFuncType
)

// makeCudaAdditionTestPod creates a v1.Pod object configured to run a CUDA vector addition application.
// This pod is designed for scenarios where host mounts are required for Nvidia libraries.
// It limits GPU resource usage to 1.
func makeCudaAdditionTestPod() *v1.Pod {
	// Generate a unique pod name using a prefix and a UUID.
	podName := testPodNamePrefix + string(uuid.NewUUID())
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever, // Pod should not restart on failure
			Containers: []v1.Container{
				{
					Name:  "vector-addition",
					Image: imageutils.GetE2EImage(imageutils.CudaVectorAdd), // Use a predefined CUDA vector addition image
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							// Request 1 GPU resource for this container.
							gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "nvidia-libraries",
							MountPath: "/usr/local/nvidia/lib64", // Mount Nvidia libraries from host
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "nvidia-libraries",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/home/kubernetes/bin/nvidia/lib", // Host path for Nvidia libraries
						},
					},
				},
			},
		},
	}
	return testPod
}

// makeCudaAdditionDevicePluginTestPod creates a v1.Pod object for running a CUDA vector addition application,
// specifically for device plugin tests. This pod does not require host mounts, as the device plugin
// handles GPU resource allocation. It limits GPU resource usage to 1.
func makeCudaAdditionDevicePluginTestPod() *v1.Pod {
	// Generate a unique pod name using a prefix and a UUID.
	podName := testPodNamePrefix + string(uuid.NewUUID())
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever, // Pod should not restart on failure
			Containers: []v1.Container{
				{
					Name:  "vector-addition",
					Image: imageutils.GetE2EImage(imageutils.CudaVectorAdd), // Use a predefined CUDA vector addition image
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							// Request 1 GPU resource for this container via the device plugin.
							gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
				},
			},
		},
	}
	return testPod
}

// isClusterRunningCOS checks if all schedulable nodes in the Kubernetes cluster are running
// Container-Optimized OS (COS). This is a prerequisite for certain Nvidia GPU tests.
func isClusterRunningCOS(f *framework.Framework) bool {
	// Retrieve the list of all nodes in the cluster.
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list") // Assert no error during node list retrieval.
	// Iterate through each node to check its OS image.
	for _, node := range nodeList.Items {
		// If any node's OS image does not contain the COS identifier, return false.
		if !strings.Contains(node.Status.NodeInfo.OSImage, cosOSImage) {
			return false
		}
	}
	return true // All nodes are running COS.
}

// areGPUsAvailableOnAllSchedulableNodes checks if Nvidia GPUs are reported as available
// in the capacity of all schedulable nodes within the cluster.
func areGPUsAvailableOnAllSchedulableNodes(f *framework.Framework) bool {
	framework.Logf("Getting list of Nodes from API server")
	// Retrieve the list of all nodes in the cluster.
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list") // Assert no error during node list retrieval.
	// Iterate through each node.
	for _, node := range nodeList.Items {
		// Skip unschedulable nodes as they won't be considered for GPU allocation.
		if node.Spec.Unschedulable {
			continue
		}
		framework.Logf("gpuResourceName %s", gpuResourceName)
		// Check if the GPU resource is present in the node's capacity and if its value is greater than 0.
		if val, ok := node.Status.Capacity[gpuResourceName]; !ok || val.Value() == 0 {
			framework.Logf("Nvidia GPUs not available on Node: %q", node.Name)
			return false // GPUs are not available on this schedulable node.
		}
	}
	framework.Logf("Nvidia GPUs exist on all schedulable nodes")
	return true // GPUs are available on all schedulable nodes.
}

// getGPUsAvailable calculates the total number of Nvidia GPUs available across all nodes in the cluster.
func getGPUsAvailable(f *framework.Framework) int64 {
	// Retrieve the list of all nodes in the cluster.
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list") // Assert no error during node list retrieval.
	var gpusAvailable int64
	// Iterate through each node.
	for _, node := range nodeList.Items {
		// If the GPU resource is found in the node's capacity, add its value to the total.
		if val, ok := node.Status.Capacity[gpuResourceName]; ok {
			gpusAvailable += (&val).Value()
		}
	}
	return gpusAvailable // Return the total count of available GPUs.
}

// testNvidiaGPUsOnCOS orchestrates the end-to-end test for Nvidia GPUs on Container-Optimized OS.
// It performs driver installation, verifies GPU availability, and runs CUDA applications.
func testNvidiaGPUsOnCOS(f *framework.Framework) {
	// Skip the test if the base image is not COS.
	// TODO: Add support for other base images.
	// CUDA apps require host mounts which is not portable across base images (yet).
	framework.Logf("Checking base image")
	// Perform the check for Container-Optimized OS.
	if !isClusterRunningCOS(f) {
		Skip("Nvidia GPU tests are supproted only on Container Optimized OS image currently")
	}
	framework.Logf("Cluster is running on COS. Proceeding with test")

	// Configure DaemonSet YAML URL and pod creation function based on the test feature.
	if f.BaseName == "device-plugin-gpus" {
		// For device plugin GPU tests, get the DaemonSet URL from environment variable
		// or use a default if not set.
		dsYamlUrlFromEnv := os.Getenv("NVIDIA_DRIVER_INSTALLER_DAEMONSET")
		if dsYamlUrlFromEnv != "" {
			dsYamlUrl = dsYamlUrlFromEnv
		} else {
			dsYamlUrl = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/daemonset.yaml"
		}
		// Use the device plugin specific GPU resource name and pod creation function.
		gpuResourceName = framework.NVIDIAGPUResourceName
		podCreationFunc = makeCudaAdditionDevicePluginTestPod
	} else {
		// For other GPU tests, use a different DaemonSet URL and default GPU resource name/pod creation.
		dsYamlUrl = "https://raw.githubusercontent.com/ContainerEngine/accelerators/master/cos-nvidia-gpu-installer/daemonset.yaml"
		gpuResourceName = v1.ResourceNvidiaGPU
		podCreationFunc = makeCudaAdditionTestPod
	}

	framework.Logf("Using %v", dsYamlUrl)
	// Create the DaemonSet that installs Nvidia Drivers.
	// The DaemonSet also runs nvidia device plugin for device plugin test.
	ds, err := framework.DsFromManifest(dsYamlUrl)
	Expect(err).NotTo(HaveOccurred())
	ds.Namespace = f.Namespace.Name
	// Create the DaemonSet in the specified namespace.
	_, err = f.ClientSet.ExtensionsV1beta1().DaemonSets(f.Namespace.Name).Create(ds)
	framework.ExpectNoError(err, "failed to create daemonset")
	framework.Logf("Successfully created daemonset to install Nvidia drivers.")

	// Wait for the pods controlled by the DaemonSet to be ready.
	pods, err := framework.WaitForControlledPods(f.ClientSet, ds.Namespace, ds.Name, extensionsinternal.Kind("DaemonSet"))
	framework.ExpectNoError(err, "getting pods controlled by the daemonset")
	// Check and add device plugin pods if they exist.
	devicepluginPods, err := framework.WaitForControlledPods(f.ClientSet, "kube-system", "nvidia-gpu-device-plugin", extensionsinternal.Kind("DaemonSet"))
	if err == nil {
		framework.Logf("Adding deviceplugin addon pod.")
		pods.Items = append(pods.Items, devicepluginPods.Items...)
	}
	framework.Logf("Starting ResourceUsageGather for the created DaemonSet pods.")
	// Initialize and start resource usage gathering for the DaemonSet pods.
	rsgather, err := framework.NewResourceUsageGatherer(f.ClientSet, framework.ResourceGathererOptions{false, false, 2 * time.Second, 2 * time.Second, true}, pods)
	framework.ExpectNoError(err, "creating ResourceUsageGather for the daemonset pods")
	go rsgather.StartGatheringData() // Run data gathering in a goroutine.

	// Wait for Nvidia GPUs to be available on nodes.
	framework.Logf("Waiting for drivers to be installed and GPUs to be available in Node Capacity...")
	// Continuously check for GPU availability until timeout.
	Eventually(func() bool {
		return areGPUsAvailableOnAllSchedulableNodes(f)
	}, driverInstallTimeout, time.Second).Should(BeTrue())

	framework.Logf("Creating as many pods as there are Nvidia GPUs and have the pods run a CUDA app")
	podList := []*v1.Pod{}
	// Create a CUDA application pod for each available GPU.
	for i := int64(0); i < getGPUsAvailable(f); i++ {
		podList = append(podList, f.PodClient().Create(podCreationFunc()))
	}
	framework.Logf("Wait for all test pods to succeed")
	// Wait for all created pods to complete successfully.
	for _, po := range podList {
		f.PodClient().WaitForSuccess(po.Name, 5*time.Minute)
	}

	framework.Logf("Stopping ResourceUsageGather")
	constraints := make(map[string]framework.ResourceConstraint)
	// For now, just gets summary. Can pass valid constraints in the future.
	// Stop resource gathering and summarize the results.
	summary, err := rsgather.StopAndSummarize([]int{50, 90, 100}, constraints)
	f.TestSummaries = append(f.TestSummaries, summary)
	framework.ExpectNoError(err, "getting resource usage summary")
}

// SIGDescribe block for [Feature:GPU] tests.
var _ = SIGDescribe("[Feature:GPU]", func() {
	f := framework.NewDefaultFramework("gpus")
	// It block for running Nvidia GPU tests on Container Optimized OS.
	It("run Nvidia GPU tests on Container Optimized OS only", func() {
		testNvidiaGPUsOnCOS(f)
	})
})

// SIGDescribe block for [Feature:GPUDevicePlugin] tests.
var _ = SIGDescribe("[Feature:GPUDevicePlugin]", func() {
	f := framework.NewDefaultFramework("device-plugin-gpus")
	// It block for running Nvidia GPU Device Plugin tests on Container Optimized OS.
	It("run Nvidia GPU Device Plugin tests on Container Optimized OS only", func() {
		testNvidiaGPUsOnCOS(f)
	})
})
