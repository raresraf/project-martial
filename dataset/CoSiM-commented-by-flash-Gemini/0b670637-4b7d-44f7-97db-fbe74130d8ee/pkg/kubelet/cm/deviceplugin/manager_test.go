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

// @file manager_test.go
// @brief Test suite for the Kubernetes device plugin manager.
// Functional Utility: This file contains unit tests for the `ManagerImpl`
// to ensure correct behavior of device plugin registration, resource allocation,
// capacity updates, and checkpointing mechanisms within the Kubelet's device management.
package deviceplugin

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	utilstore "k8s.io/kubernetes/pkg/kubelet/util/store"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

const (
	socketName       = "/tmp/device_plugin/server.sock"
	pluginSocketName = "/tmp/device_plugin/device-plugin.sock"
	testResourceName = "fake-domain/resource"
)

// @func TestNewManagerImpl
// @brief Tests the successful creation of a new ManagerImpl instance.
// @scenario Verifies that `newManagerImpl` function returns a valid manager
//          without any errors.
// @expected No error is returned during manager initialization.
func TestNewManagerImpl(t *testing.T) {
	_, err := newManagerImpl(socketName)
	require.NoError(t, err)
}

// @func TestNewManagerImplStart
// @brief Tests the basic startup and shutdown of the Device Plugin Manager.
// @scenario Initializes the manager and a stub device plugin, starts them,
//          and then gracefully shuts them down. This verifies the fundamental
//          lifecycle operations without complex interactions.
// @expected No errors during setup, start, or cleanup.
func TestNewManagerImplStart(t *testing.T) {
	m, p := setup(t, []*pluginapi.Device{}, func(n string, a, u, r []pluginapi.Device) {})
	cleanup(t, m, p)
}

// @func TestDevicePluginReRegistration
// @brief Tests the device plugin manager's handling of device registration and re-registration.
// @scenario 1. A device plugin registers with two devices.
//           2. The same device plugin re-registers (simulating a restart) with the same two devices.
//           3. A *new* device plugin registers for the same resource, providing *different* devices.
// @expected - After initial registration, the manager should list two devices.
//           - After re-registration of the *same* plugin, devices should remain unchanged (not removed).
//           - After a *different* plugin registers, previously registered devices from the older plugin should be removed and replaced by the new plugin's devices.
//           - No orphaned devices should remain.
func TestDevicePluginReRegistration(t *testing.T) {
	devs := []*pluginapi.Device{
		{ID: "Dev1", Health: pluginapi.Healthy},
		{ID: "Dev2", Health: pluginapi.Healthy},
	}
	devsForRegistration := []*pluginapi.Device{
		{ID: "Dev3", Health: pluginapi.Healthy},
	}

	expCallbackCount := int32(0)
	callbackCount := int32(0)
	callbackChan := make(chan int32)
	// Functional Utility: Defines a callback function to monitor registration events.
	callback := func(n string, a, u, r []pluginapi.Device) {
		atomic.AddInt32(&callbackCount, 1)
		// Functional Utility: Ensures the callback is not invoked more times than expected.
		if callbackCount > atomic.LoadInt32(&expCallbackCount) {
			t.FailNow()
		}
		callbackChan <- callbackCount
	}
	// Block Logic: Setup manager and first stub plugin.
	m, p1 := setup(t, devs, callback)
	atomic.StoreInt32(&expCallbackCount, 1)
	// Functional Utility: Register the first plugin.
	p1.Register(socketName, testResourceName)
	// Block Logic: Wait for the initial registration callback.
	<-callbackChan
	// Functional Utility: Verify that the manager's devices are updated correctly.
	devices := m.Devices()
	require.Equal(t, 2, len(devices[testResourceName]), "Devices are not updated.")

	// Block Logic: Setup and register a second plugin (simulating a re-registration from a new instance).
	p2 := NewDevicePluginStub(devs, pluginSocketName+".new")
	err := p2.Start()
	require.NoError(t, err)
	atomic.StoreInt32(&expCallbackCount, 2)
	p2.Register(socketName, testResourceName)
	// Block Logic: Wait for the re-registration callback.
	<-callbackChan

	// Functional Utility: Verify that devices remain unchanged after re-registration by the same logical plugin.
	devices2 := m.Devices()
	require.Equal(t, 2, len(devices2[testResourceName]), "Devices shouldn't change.")

	// Block Logic: Setup and register a third plugin with different devices.
	p3 := NewDevicePluginStub(devsForRegistration, pluginSocketName+".third")
	err = p3.Start()
	require.NoError(t, err)
	atomic.StoreInt32(&expCallbackCount, 3)
	p3.Register(socketName, testResourceName)
	// Block Logic: Wait for the third registration callback.
	<-callbackChan

	// Functional Utility: Verify that devices from the previously registered plugin are removed and replaced.
	devices3 := m.Devices()
	require.Equal(t, 1, len(devices3[testResourceName]), "Devices of plugin previously registered should be removed.")
	// Functional Utility: Stop all stub plugins and clean up the manager.
	p2.Stop()
	p3.Stop()
	cleanup(t, m, p1)
	close(callbackChan)
}

// @func setup
// @brief Sets up a test environment for the Device Plugin Manager.
// @param t The testing.T instance for reporting errors.
// @param devs A slice of `pluginapi.Device` representing the devices to be served by the stub plugin.
// @param callback A `monitorCallback` function to be used by the manager for device updates.
// @return A `Manager` interface and a `Stub` device plugin.
// Functional Utility: Initializes a new `ManagerImpl`, starts it, and creates/starts a `DevicePluginStub`
// for testing device registration and lifecycle.
func setup(t *testing.T, devs []*pluginapi.Device, callback monitorCallback) (Manager, *Stub) {
	// Functional Utility: Create a new instance of the device plugin manager.
	m, err := newManagerImpl(socketName)
	require.NoError(t, err)

	// Functional Utility: Assign the provided callback function to the manager for testing purposes.
	m.callback = callback

	// Functional Utility: Define a stub for `activePods` function, returning an empty list for this setup.
	activePods := func() []*v1.Pod {
		return []*v1.Pod{}
	}
	// Functional Utility: Start the device plugin manager.
	err = m.Start(activePods, &sourcesReadyStub{})
	require.NoError(t, err)

	// Functional Utility: Create and start a new device plugin stub.
	p := NewDevicePluginStub(devs, pluginSocketName)
	err = p.Start()
	require.NoError(t, err)

	return m, p
}

// @func cleanup
// @brief Tears down the test environment by stopping the manager and the device plugin stub.
// @param t The testing.T instance.
// @param m The `Manager` interface to stop.
// @param p The `Stub` device plugin to stop.
// Functional Utility: Ensures that all background goroutines and resources are properly
// terminated and released after a test.
func cleanup(t *testing.T, m Manager, p *Stub) {
	p.Stop()
	m.Stop()
}

// @func TestUpdateCapacity
// @brief Tests the `UpdateCapacity` functionality of the Device Plugin Manager.
// @scenario - Initializes a manager with various device states (healthy, unhealthy).
//           - Simulates adding, updating (health status), and deleting devices.
//           - Simulates adding and removing entire resource endpoints.
// @expected The reported capacity for each resource should accurately reflect the
//          number of healthy devices at each step, and removed resources should
//          be correctly identified.
func TestUpdateCapacity(t *testing.T) {
	testManager, err := newManagerImpl(socketName)
	as := assert.New(t)
	as.NotNil(testManager)
	as.Nil(err)

	devs := []pluginapi.Device{
		{ID: "Device1", Health: pluginapi.Healthy},
		{ID: "Device2", Health: pluginapi.Healthy},
		{ID: "Device3", Health: pluginapi.Unhealthy},
	}
	// Functional Utility: Use the internal callback for device updates.
	callback := testManager.genericDeviceUpdateCallback

	// Block Logic: Adds three devices for resource1 (two healthy, one unhealthy). Expects capacity for resource1 to be 2.
	resourceName1 := "domain1.com/resource1"
	testManager.endpoints[resourceName1] = &endpointImpl{devices: make(map[string]pluginapi.Device)}
	callback(resourceName1, devs, []pluginapi.Device{}, []pluginapi.Device{})
	capacity, removedResources := testManager.GetCapacity()
	resource1Capacity, ok := capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(2), resource1Capacity.Value())
	as.Equal(0, len(removedResources))

	// Block Logic: Deleting an unhealthy device should not change capacity.
	callback(resourceName1, []pluginapi.Device{}, []pluginapi.Device{}, []pluginapi.Device{devs[2]})
	capacity, removedResources = testManager.GetCapacity()
	resource1Capacity, ok = capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(2), resource1Capacity.Value())
	as.Equal(0, len(removedResources))

	// Block Logic: Updating a healthy device to unhealthy should reduce capacity by 1.
	dev2 := devs[1]
	dev2.Health = pluginapi.Unhealthy
	callback(resourceName1, []pluginapi.Device{}, []pluginapi.Device{dev2}, []pluginapi.Device{})
	capacity, removedResources = testManager.GetCapacity()
	resource1Capacity, ok = capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(1), resource1Capacity.Value())
	as.Equal(0, len(removedResources))

	// Block Logic: Deleting a healthy device should reduce capacity by 1.
	callback(resourceName1, []pluginapi.Device{}, []pluginapi.Device{}, []pluginapi.Device{devs[0]})
	capacity, removedResources = testManager.GetCapacity()
	resource1Capacity, ok = capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(0), resource1Capacity.Value())
	as.Equal(0, len(removedResources))

	// Block Logic: Tests adding another resource.
	resourceName2 := "resource2"
	testManager.endpoints[resourceName2] = &endpointImpl{devices: make(map[string]pluginapi.Device)}
	callback(resourceName2, devs, []pluginapi.Device{}, []pluginapi.Device{})
	capacity, removedResources = testManager.GetCapacity()
	as.Equal(2, len(capacity))
	resource2Capacity, ok := capacity[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(2), resource2Capacity.Value())
	as.Equal(0, len(removedResources))

	// Block Logic: Removes resourceName1 endpoint. Verifies GetCapacity() reports its removal and non-existence.
	delete(testManager.endpoints, resourceName1)
	capacity, removed := testManager.GetCapacity()
	as.Equal([]string{resourceName1}, removed)
	_, ok = capacity[v1.ResourceName(resourceName1)]
	as.False(ok)
	val, ok := capacity[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(2), val.Value())
	_, ok = testManager.allDevices[resourceName1]
	as.False(ok)
}

// @type stringPairType
// @brief A simple struct to hold two strings.
// Functional Utility: Used as a key in maps for testing purposes, allowing for
// composite keys when needing to associate data with two string identifiers.
type stringPairType struct {
	value1 string
	value2 string
}

// @func constructDevices
// @brief Converts a slice of device IDs into a `sets.String`.
// @param devices A slice of strings representing device IDs.
// @return A `sets.String` containing the unique device IDs.
// Functional Utility: A helper function to easily create `sets.String` objects
// from string slices, which are commonly used in device management for quick lookups.
func constructDevices(devices []string) sets.String {
	ret := sets.NewString()
	for _, dev := range devices {
		ret.Insert(dev)
	}
	return ret
}

// @func constructAllocResp
// @brief Constructs a `pluginapi.AllocateResponse` from given device, mount, and environment maps.
// @param devices A map of host paths to container paths for devices.
// @param mounts A map of container paths to host paths for mounts.
// @param envs A map of environment variable names to values.
// @return A pointer to a `pluginapi.AllocateResponse` populated with the provided data.
// Functional Utility: A test helper to easily create allocation responses that simulate
// what a device plugin would return, simplifying the setup of test scenarios.
func constructAllocResp(devices, mounts, envs map[string]string) *pluginapi.AllocateResponse {
	resp := &pluginapi.AllocateResponse{}
	for k, v := range devices {
		resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
			HostPath:      k,
			ContainerPath: v,
			Permissions:   "mrw",
		})
	}
	for k, v := range mounts {
		resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
			ContainerPath: k,
			HostPath:      v,
			ReadOnly:      true,
		})
	}
	resp.Envs = make(map[string]string)
	for k, v := range envs {
		resp.Envs[k] = v
	}
	return resp
}

// @func TestCheckpoint
// @brief Tests the checkpointing mechanism of the Device Plugin Manager.
// @scenario - A `ManagerImpl` is initialized and populated with mock pod and device data.
//           - The manager's state is written to a checkpoint file.
//           - The manager's in-memory state is cleared.
//           - The manager's state is re-read from the checkpoint file.
// @expected The state recovered from the checkpoint should be identical to the original state.
// Functional Utility: Ensures that the device plugin manager can persistently store and
// recover its internal state, which is critical for Kubelet restarts and resilience.
func TestCheckpoint(t *testing.T) {
	resourceName1 := "domain1.com/resource1"
	resourceName2 := "domain2.com/resource2"

	as := assert.New(t)
	// Functional Utility: Create a temporary directory for checkpoint files.
	tmpDir, err := ioutil.TempDir("", "checkpoint")
	as.Nil(err)
	defer os.RemoveAll(tmpDir) // Functional Utility: Ensures cleanup of the temporary directory.

	// Functional Utility: Initialize a test manager with mock data structures.
	testManager := &ManagerImpl{
		socketdir:        tmpDir,
		allDevices:       make(map[string]sets.String),
		allocatedDevices: make(map[string]sets.String),
		podDevices:       make(podDevices),
	}
	// Functional Utility: Initialize a file store for checkpointing.
	testManager.store, _ = utilstore.NewFileStore("/tmp/", utilfs.DefaultFs{})

	// Block Logic: Insert mock pod and container device allocation data into the manager.
	testManager.podDevices.insert("pod1", "con1", resourceName1,
		constructDevices([]string{"dev1", "dev2"}),
		constructAllocResp(map[string]string{"/dev/r1dev1": "/dev/r1dev1", "/dev/r1dev2": "/dev/r1dev2"},
			map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))
	testManager.podDevices.insert("pod1", "con1", resourceName2,
		constructDevices([]string{"dev1", "dev2"}),
		constructAllocResp(map[string]string{"/dev/r2dev1": "/dev/r2dev1", "/dev/r2dev2": "/dev/r2dev2"},
			map[string]string{"/home/r2lib1": "/usr/r2lib1"},
			map[string]string{"r2devices": "dev1 dev2"}))
	testManager.podDevices.insert("pod1", "con2", resourceName1,
		constructDevices([]string{"dev3"}),
		constructAllocResp(map[string]string{"/dev/r1dev3": "/dev/r1dev3"},
			map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))
	testManager.podDevices.insert("pod2", "con1", resourceName1,
		constructDevices([]string{"dev4"}),
		constructAllocResp(map[string]string{"/dev/r1dev4": "/dev/r1dev4"},
			map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))

	// Block Logic: Populate allDevices map with mock data.
	testManager.allDevices[resourceName1] = sets.NewString()
	testManager.allDevices[resourceName1].Insert("dev1")
	testManager.allDevices[resourceName1].Insert("dev2")
	testManager.allDevices[resourceName1].Insert("dev3")
	testManager.allDevices[resourceName1].Insert("dev4")
	testManager.allDevices[resourceName1].Insert("dev5")
	testManager.allDevices[resourceName2] = sets.NewString()
	testManager.allDevices[resourceName2].Insert("dev1")
	testManager.allDevices[resourceName2].Insert("dev2")

	// Functional Utility: Store the expected state before checkpointing.
	expectedPodDevices := testManager.podDevices
	expectedAllocatedDevices := testManager.podDevices.devices()
	expectedAllDevices := testManager.allDevices

	// Block Logic: Write the current manager state to a checkpoint file.
	err = testManager.writeCheckpoint()

	as.Nil(err)
	// Block Logic: Clear the in-memory state of podDevices and read from checkpoint.
	testManager.podDevices = make(podDevices)
	err = testManager.readCheckpoint()
	as.Nil(err)

	// Block Logic: Verify that the recovered podDevices match the expected state.
	as.Equal(len(expectedPodDevices), len(testManager.podDevices))
	for podUID, containerDevices := range expectedPodDevices {
		for conName, resources := range containerDevices {
			for resource := range resources {
				as.True(reflect.DeepEqual(
					expectedPodDevices.containerDevices(podUID, conName, resource),
					testManager.podDevices.containerDevices(podUID, conName, resource)))
				opts1 := expectedPodDevices.deviceRunContainerOptions(podUID, conName)
				opts2 := testManager.podDevices.deviceRunContainerOptions(podUID, conName)
				as.Equal(len(opts1.Envs), len(opts2.Envs))
				as.Equal(len(opts1.Mounts), len(opts2.Mounts))
				as.Equal(len(opts1.Devices), len(opts2.Devices))
			}
		}
	}
	// Functional Utility: Verify that allocatedDevices and allDevices also match the expected state.
	as.True(reflect.DeepEqual(expectedAllocatedDevices, testManager.allocatedDevices))
	as.True(reflect.DeepEqual(expectedAllDevices, testManager.allDevices))
}

// @type activePodsStub
// @brief A stub implementation for `ActivePodsFunc`.
// Functional Utility: Allows tests to simulate changes in the list of active pods
// without relying on a real Kubelet state.
type activePodsStub struct {
	activePods []*v1.Pod
}

// @func (*activePodsStub) getActivePods
// @brief Returns the currently active pods managed by the stub.
// @return A slice of `*v1.Pod`.
func (a *activePodsStub) getActivePods() []*v1.Pod {
	return a.activePods
}

// @func (*activePodsStub) updateActivePods
// @brief Updates the list of active pods in the stub.
// @param newPods A slice of `*v1.Pod` to set as active.
func (a *activePodsStub) updateActivePods(newPods []*v1.Pod) {
	a.activePods = newPods
}

// @type MockEndpoint
// @brief A mock implementation of the `endpoint` interface.
// Functional Utility: Used in tests to simulate the behavior of a device plugin
// endpoint, particularly for controlling the `allocate` method's response.
type MockEndpoint struct {
	allocateFunc func(devs []string) (*pluginapi.AllocateResponse, error)
}

// @func (*MockEndpoint) stop
// @brief Implements the `stop` method of the `endpoint` interface for the mock.
// Functional Utility: No-op for the mock, as it doesn't manage external resources.
func (m *MockEndpoint) stop() {}

// @func (*MockEndpoint) run
// @brief Implements the `run` method of the `endpoint` interface for the mock.
// Functional Utility: No-op for the mock.
func (m *MockEndpoint) run() {}

// @func (*MockEndpoint) getDevices
// @brief Implements the `getDevices` method of the `endpoint` interface for the mock.
// @return An empty slice of `pluginapi.Device`.
// Functional Utility: Returns no devices, as this mock focuses on allocation behavior.
func (m *MockEndpoint) getDevices() []pluginapi.Device {
	return []pluginapi.Device{}
}

// @func (*MockEndpoint) callback
// @brief Implements the `callback` method of the `endpoint` interface for the mock.
// Functional Utility: No-op for the mock.
func (m *MockEndpoint) callback(resourceName string, added, updated, deleted []pluginapi.Device) {}

// @func (*MockEndpoint) allocate
// @brief Implements the `allocate` method of the `endpoint` interface for the mock.
// @param devs A slice of device IDs to allocate.
// @return A `*pluginapi.AllocateResponse` and an error.
// Functional Utility: Delegates to an optional `allocateFunc` if set, otherwise returns nil.
// This allows tests to define custom allocation logic for specific scenarios.
func (m *MockEndpoint) allocate(devs []string) (*pluginapi.AllocateResponse, error) {
	if m.allocateFunc != nil {
		return m.allocateFunc(devs)
	}
	return nil, nil
}

// @func makePod
// @brief Creates a new `*v1.Pod` object with a generated UID and specified resource requests.
// @param requests A `v1.ResourceList` defining the resource requests for the pod's container.
// @return A new `*v1.Pod` instance.
// Functional Utility: A test helper for easily constructing pod objects used in allocation tests.
func makePod(requests v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: requests,
					},
				},
			},
		},
	}
}

// @func getTestManager
// @brief Creates and initializes a `ManagerImpl` for testing purposes.
// @param tmpDir The temporary directory for checkpoint files.
// @param activePods A function that returns a list of active pods.
// @param testRes A slice of `TestResource` defining available resources and devices.
// @return An initialized `*ManagerImpl`.
// Functional Utility: Centralized helper to set up a test manager with mock endpoints
// and devices based on test resource definitions.
func getTestManager(tmpDir string, activePods ActivePodsFunc, testRes []TestResource) *ManagerImpl {
	// Functional Utility: No-op callback for this test manager instance.
	monitorCallback := func(resourceName string, added, updated, deleted []pluginapi.Device) {}
	testManager := &ManagerImpl{
		socketdir:        tmpDir,
		callback:         monitorCallback,
		allDevices:       make(map[string]sets.String),
		allocatedDevices: make(map[string]sets.String),
		endpoints:        make(map[string]endpoint),
		podDevices:       make(podDevices),
		activePods:       activePods,
		sourcesReady:     &sourcesReadyStub{},
	}
	// Functional Utility: Initialize file store for checkpointing.
	testManager.store, _ = utilstore.NewFileStore("/tmp/", utilfs.DefaultFs{})
	// Block Logic: Populate `allDevices` and mock `endpoints` based on provided `testRes`.
	for _, res := range testRes {
		testManager.allDevices[res.resourceName] = sets.NewString()
		for _, dev := range res.devs {
			testManager.allDevices[res.resourceName].Insert(dev)
		}
		// Block Logic: Define specific mock allocation behavior for "domain1.com/resource1".
		if res.resourceName == "domain1.com/resource1" {
			testManager.endpoints[res.resourceName] = &MockEndpoint{
				allocateFunc: func(devs []string) (*pluginapi.AllocateResponse, error) {
					resp := new(pluginapi.AllocateResponse)
					resp.Envs = make(map[string]string)
					for _, dev := range devs {
						switch dev {
						case "dev1":
							resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
								ContainerPath: "/dev/aaa",
								HostPath:      "/dev/aaa",
								Permissions:   "mrw",
							})

							resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
								ContainerPath: "/dev/bbb",
								HostPath:      "/dev/bbb",
								Permissions:   "mrw",
							})

							resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
								ContainerPath: "/container_dir1/file1",
								HostPath:      "host_dir1/file1",
								ReadOnly:      true,
							})

						case "dev2":
							resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
								ContainerPath: "/dev/ccc",
								HostPath:      "/dev/ccc",
								Permissions:   "mrw",
							})

							resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
								ContainerPath: "/container_dir1/file2",
								HostPath:      "host_dir1/file2",
								ReadOnly:      true,
							})

							resp.Envs["key1"] = "val1"
						}
					}
					return resp, nil
				},
			}
		}
		// Block Logic: Define specific mock allocation behavior for "domain2.com/resource2".
		if res.resourceName == "domain2.com/resource2" {
			testManager.endpoints[res.resourceName] = &MockEndpoint{
				allocateFunc: func(devs []string) (*pluginapi.AllocateResponse, error) {
					resp := new(pluginapi.AllocateResponse)
					resp.Envs = make(map[string]string)
					for _, dev := range devs {
						switch dev {
						case "dev3":
							resp.Envs["key2"] = "val2"

						case "dev4":
							resp.Envs["key2"] = "val3"
						}
					}
					return resp, nil
				},
			}
		}
	}
	return testManager
}

// @func getTestNodeInfo
// @brief Creates a `*schedulercache.NodeInfo` with specified allocatable resources.
// @param allocatable A `v1.ResourceList` representing the node's allocatable resources.
// @return A `*schedulercache.NodeInfo`.
// Functional Utility: A test helper to construct a node information object used in allocation tests.
func getTestNodeInfo(allocatable v1.ResourceList) *schedulercache.NodeInfo {
	cachedNode := &v1.Node{
		Status: v1.NodeStatus{
			Allocatable: allocatable,
		},
	}
	nodeInfo := &schedulercache.NodeInfo{}
	nodeInfo.SetNode(cachedNode)
	return nodeInfo
}

// @type TestResource
// @brief Represents a test resource with its name, quantity, and associated devices.
// Functional Utility: A convenience struct for defining test data for device plugin resources.
type TestResource struct {
	resourceName     string
	resourceQuantity resource.Quantity
	devs             []string
}

// @func TestPodContainerDeviceAllocation
// @brief Tests the allocation of devices to pod containers.
// @scenario - Configures a test manager with mock resources and devices.
//           - Simulates various pod allocation requests, including successful allocations
//             and requests exceeding available resources.
// @expected - Successful allocations should result in correct `DeviceRunContainerOptions`
//             and updated `allocatedDevices` counts.
//           - Requests for unavailable resources should return a specific error.
func TestPodContainerDeviceAllocation(t *testing.T) {
	// Functional Utility: Set up logging for detailed output during the test.
	flag.Set("alsologtostderr", fmt.Sprintf("%t", true))
	var logLevel string
	flag.StringVar(&logLevel, "logLevel", "4", "test")
	flag.Lookup("v").Value.Set(logLevel)
	// Functional Utility: Define test resources with names, quantities, and associated devices.
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(2), resource.DecimalSI),
		devs:             []string{"dev1", "dev2"},
	}
	res2 := TestResource{
		resourceName:     "domain2.com/resource2",
		resourceQuantity: *resource.NewQuantity(int64(1), resource.DecimalSI),
		devs:             []string{"dev3", "dev4"},
	}
	testResources := make([]TestResource, 2)
	testResources = append(testResources, res1)
	testResources = append(testResources, res2)
	as := require.New(t)
	// Functional Utility: Create a stub for active pods.
	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	// Functional Utility: Create a temporary directory for checkpoint files.
	tmpDir, err := ioutil.TempDir("", "checkpoint")
	as.Nil(err)
	defer os.RemoveAll(tmpDir) // Functional Utility: Ensures cleanup of the temporary directory.
	// Functional Utility: Get a test node info object with empty allocatable resources.
	nodeInfo := getTestNodeInfo(v1.ResourceList{})
	// Functional Utility: Initialize the test manager with the defined resources.
	testManager := getTestManager(tmpDir, podsStub.getActivePods, testResources)

	// Functional Utility: Define test pods with various resource requests.
	testPods := []*v1.Pod{
		makePod(v1.ResourceList{
			v1.ResourceName(res1.resourceName): res1.resourceQuantity,
			v1.ResourceName("cpu"):             res1.resourceQuantity,
			v1.ResourceName(res2.resourceName): res2.resourceQuantity}),
		makePod(v1.ResourceList{
			v1.ResourceName(res1.resourceName): res2.resourceQuantity}),
		makePod(v1.ResourceList{
			v1.ResourceName(res2.resourceName): res2.resourceQuantity}),
	}
	// Functional Utility: Define test cases with descriptions, expected outcomes, and errors.
	testCases := []struct {
		description               string
		testPod                   *v1.Pod
		expectedContainerOptsLen  []int
		expectedAllocatedResName1 int
		expectedAllocatedResName2 int
		expErr                    error
	}{
		{
			description:               "Successfull allocation of two Res1 resources and one Res2 resource",
			testPod:                   testPods[0],
			expectedContainerOptsLen:  []int{3, 2, 2}, // Devices, Mounts, Envs
			expectedAllocatedResName1: 2,
			expectedAllocatedResName2: 1,
			expErr: nil,
		},
		{
			description:               "Requesting to create a pod without enough resources should fail",
			testPod:                   testPods[1],
			expectedContainerOptsLen:  nil,
			expectedAllocatedResName1: 2,
			expectedAllocatedResName2: 1,
			expErr: fmt.Errorf("requested number of devices unavailable for domain1.com/resource1. Requested: 1, Available: 0"),
		},
		{
			description:               "Successfull allocation of all available Res1 resources and Res2 resources",
			testPod:                   testPods[2],
			expectedContainerOptsLen:  []int{0, 0, 1},
			expectedAllocatedResName1: 2,
			expectedAllocatedResName2: 2,
			expErr: nil,
		},
	}
	activePods := []*v1.Pod{}
	/**
	 * Block Logic: Iterates through each defined test case to perform allocation and verification.
	 * Invariant: The manager's state (allocated devices, container options) is checked against expected values after each allocation attempt.
	 */
	for _, testCase := range testCases {
		pod := testCase.testPod
		activePods = append(activePods, pod)
		podsStub.updateActivePods(activePods)
		// Functional Utility: Attempt to allocate devices for the current pod.
		err := testManager.Allocate(nodeInfo, &lifecycle.PodAdmitAttributes{Pod: pod})
		// Functional Utility: Verify that the returned error matches the expected error.
		if !reflect.DeepEqual(err, testCase.expErr) {
			t.Errorf("DevicePluginManager error (%v). expected error: %v but got: %v",
				testCase.description, testCase.expErr, err)
		}
		// Functional Utility: Retrieve the device run container options.
		runContainerOpts := testManager.GetDeviceRunContainerOptions(pod, &pod.Spec.Containers[0])
		// Block Logic: Verify the length of container options (devices, mounts, envs) if an allocation was expected.
		if testCase.expectedContainerOptsLen == nil {
			as.Nil(runContainerOpts)
		} else {
			as.Equal(len(runContainerOpts.Devices), testCase.expectedContainerOptsLen[0])
			as.Equal(len(runContainerOpts.Mounts), testCase.expectedContainerOptsLen[1])
			as.Equal(len(runContainerOpts.Envs), testCase.expectedContainerOptsLen[2])
		}
		// Functional Utility: Verify the total number of allocated devices for each resource type.
		as.Equal(testCase.expectedAllocatedResName1, testManager.allocatedDevices[res1.resourceName].Len())
		as.Equal(testCase.expectedAllocatedResName2, testManager.allocatedDevices[res2.resourceName].Len())
	}

}

// @func TestInitContainerDeviceAllocation
// @brief Tests device allocation behavior for pods with init containers.
// @scenario A pod requests device resources in both init containers and regular containers.
// @expected Devices allocated to init containers should be successfully reallocated
//          to normal containers without conflict, ensuring that devices are
//          available across container phases.
func TestInitContainerDeviceAllocation(t *testing.T) {
	// Functional Utility: Define test resources similar to how a device plugin would report them.
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(2), resource.DecimalSI),
		devs:             []string{"dev1", "dev2"},
	}
	res2 := TestResource{
		resourceName:     "domain2.com/resource2",
		resourceQuantity: *resource.NewQuantity(int64(1), resource.DecimalSI),
		devs:             []string{"dev3", "dev4"},
	}
	testResources := make([]TestResource, 2)
	testResources = append(testResources, res1)
	testResources = append(testResources, res2)
	as := require.New(t)
	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	nodeInfo := getTestNodeInfo(v1.ResourceList{})
	// Functional Utility: Create a temporary directory for checkpoint files.
	tmpDir, err := ioutil.TempDir("", "checkpoint")
	as.Nil(err)
	defer os.RemoveAll(tmpDir) // Functional Utility: Ensures cleanup of the temporary directory.
	// Functional Utility: Initialize the test manager with the defined resources.
	testManager := getTestManager(tmpDir, podsStub.getActivePods, testResources)

	// Functional Utility: Create a pod that requests resources in both init and normal containers.
	podWithPluginResourcesInInitContainers := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res2.resourceQuantity, // Requesting 1 of resource1
						},
					},
				},
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res1.resourceQuantity, // Requesting 2 of resource1
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res2.resourceQuantity, // Requesting 1 of resource1
							v1.ResourceName(res2.resourceName): res2.resourceQuantity,
						},
					},
				},
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res2.resourceQuantity, // Requesting 1 of resource1
							v1.ResourceName(res2.resourceName): res2.resourceQuantity,
						},
					},
				},
			},
		},
	}
	podsStub.updateActivePods([]*v1.Pod{podWithPluginResourcesInInitContainers})
	// Functional Utility: Attempt to allocate devices for the pod.
	err = testManager.Allocate(nodeInfo, &lifecycle.PodAdmitAttributes{Pod: podWithPluginResourcesInInitContainers})
	as.Nil(err) // Expected to succeed.

	// Functional Utility: Extract UIDs and names for verification.
	podUID := string(podWithPluginResourcesInInitContainers.UID)
	initCont1 := podWithPluginResourcesInInitContainers.Spec.InitContainers[0].Name
	initCont2 := podWithPluginResourcesInInitContainers.Spec.InitContainers[1].Name
	normalCont1 := podWithPluginResourcesInInitContainers.Spec.Containers[0].Name
	normalCont2 := podWithPluginResourcesInInitContainers.Spec.Containers[1].Name

	// Functional Utility: Get device sets allocated to each container.
	initCont1Devices := testManager.podDevices.containerDevices(podUID, initCont1, res1.resourceName)
	initCont2Devices := testManager.podDevices.containerDevices(podUID, initCont2, res1.resourceName)
	normalCont1Devices := testManager.podDevices.containerDevices(podUID, normalCont1, res1.resourceName)
	normalCont2Devices := testManager.podDevices.containerDevices(podUID, normalCont2, res1.resourceName)

	// Block Logic: Verify that devices allocated to earlier containers are subsets of later containers' allocations,
	// and that normal containers don't share devices.
	// Invariant: InitContainer2's devices include those from InitContainer1.
	as.True(initCont2Devices.IsSuperset(initCont1Devices))
	// Invariant: InitContainer2's devices include those from NormalContainer1.
	as.True(initCont2Devices.IsSuperset(normalCont1Devices))
	// Invariant: InitContainer2's devices include those from NormalContainer2.
	as.True(initCont2Devices.IsSuperset(normalCont2Devices))
	// Invariant: NormalContainer1 and NormalContainer2 do not share devices.
	as.Equal(0, normalCont1Devices.Intersection(normalCont2Devices).Len())
}

// @func TestSanitizeNodeAllocatable
// @brief Tests the `sanitizeNodeAllocatable` method, which adjusts node allocatable resources based on allocated devices.
// @scenario - Initializes a test manager with mock allocated devices for two resources.
//           - Creates a `NodeInfo` with allocatable resources that are inconsistent with the allocated devices
//             (e.g., missing an allocated resource, or having more than allocated).
//           - Calls `sanitizeNodeAllocatable` to correct the `NodeInfo`.
// @expected - If a resource is allocated but missing from `NodeInfo.Allocatable`, it should be added with a quantity equal to its allocated amount.
//           - If a resource is allocated and present in `NodeInfo.Allocatable` with a higher quantity, it should remain unchanged.
func TestSanitizeNodeAllocatable(t *testing.T) {
	resourceName1 := "domain1.com/resource1"
	devID1 := "dev1"

	resourceName2 := "domain2.com/resource2"
	devID2 := "dev2"

	as := assert.New(t)
	// Functional Utility: No-op callback for this test manager instance.
	monitorCallback := func(resourceName string, added, updated, deleted []pluginapi.Device) {}

	// Functional Utility: Initialize a test manager with mock data structures.
	testManager := &ManagerImpl{
		callback:         monitorCallback,
		allDevices:       make(map[string]sets.String),
		allocatedDevices: make(map[string]sets.String),
		podDevices:       make(podDevices),
	}
	// Functional Utility: Initialize a file store for checkpointing.
	testManager.store, _ = utilstore.NewFileStore("/tmp/", utilfs.DefaultFs{})
	// Block Logic: Manually set `allocatedDevices` to simulate devices already allocated.
	testManager.allocatedDevices[resourceName1] = sets.NewString()
	testManager.allocatedDevices[resourceName1].Insert(devID1)
	testManager.allocatedDevices[resourceName2] = sets.NewString()
	testManager.allocatedDevices[resourceName2].Insert(devID2)

	// Functional Utility: Create a cached node with an inconsistent allocatable resource list.
	cachedNode := &v1.Node{
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				// has no resource1 and two of resource2
				v1.ResourceName(resourceName2): *resource.NewQuantity(int64(2), resource.DecimalSI),
			},
		},
	}
	nodeInfo := &schedulercache.NodeInfo{}
	nodeInfo.SetNode(cachedNode)

	// Functional Utility: Call the method under test.
	testManager.sanitizeNodeAllocatable(nodeInfo)

	// Functional Utility: Retrieve the sanitized allocatable scalar resources.
	allocatableScalarResources := nodeInfo.AllocatableResource().ScalarResources
	// Block Logic: Verify that resource1 is added to allocatable with quantity 1 (matching allocated).
	as.Equal(1, int(allocatableScalarResources[v1.ResourceName(resourceName1)]))
	// Block Logic: Verify that resource2's allocatable quantity remains 2 (as it was greater than allocated).
	as.Equal(2, int(allocatableScalarResources[v1.ResourceName(resourceName2)]))
}
