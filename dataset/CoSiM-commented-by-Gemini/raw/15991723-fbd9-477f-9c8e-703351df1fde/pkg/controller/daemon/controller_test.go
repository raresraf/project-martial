// +build !integration

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// This file contains the unit tests for the DaemonSet controller. It tests
// the controller's logic for creating, updating, and deleting pods based on
// DaemonSet specifications, node selectors, resource constraints, and various
// scheduling conditions.
package daemon

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/securitycontext"
)

var (
	// simpleDaemonSetLabel and simpleDaemonSetLabel2 are labels for pods created by the daemon set.
	simpleDaemonSetLabel  = map[string]string{"name": "simple-daemon", "type": "production"}
	simpleDaemonSetLabel2 = map[string]string{"name": "simple-daemon", "type": "test"}
	// simpleNodeLabel and simpleNodeLabel2 are labels for nodes.
	simpleNodeLabel       = map[string]string{"color": "blue", "speed": "fast"}
	simpleNodeLabel2      = map[string]string{"color": "red", "speed": "fast"}
	// alwaysReady is a function that returns true, used for mocking store sync checks.
	alwaysReady           = func() bool { return true }
)

// getKey is a helper function to get the key for a DaemonSet.
func getKey(ds *extensions.DaemonSet, t *testing.T) string {
	if key, err := controller.KeyFunc(ds); err != nil {
		t.Errorf("Unexpected error getting key for ds %v: %v", ds.Name, err)
		return ""
	} else {
		return key
	}
}

// newDaemonSet is a factory function for creating a new DaemonSet for testing.
func newDaemonSet(name string) *extensions.DaemonSet {
	return &extensions.DaemonSet{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Extensions.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.DaemonSetSpec{
			Selector: &unversioned.LabelSelector{MatchLabels: simpleDaemonSetLabel},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: simpleDaemonSetLabel,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo/bar",
							TerminationMessagePath: api.TerminationMessagePathDefault,
							ImagePullPolicy:        api.PullIfNotPresent,
							SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
						},
					},
					DNSPolicy: api.DNSDefault,
				},
			},
		},
	}
}

// newNode is a factory function for creating a new Node for testing.
func newNode(name string, label map[string]string) *api.Node {
	return &api.Node{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Labels:    label,
			Namespace: api.NamespaceDefault,
		},
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{Type: api.NodeReady, Status: api.ConditionTrue},
			},
		},
	}
}

// addNodes is a helper function to add a number of nodes to the node store.
func addNodes(nodeStore cache.Store, startIndex, numNodes int, label map[string]string) {
	for i := startIndex; i < startIndex+numNodes; i++ {
		nodeStore.Add(newNode(fmt.Sprintf("node-%d", i), label))
	}
}

// newPod is a factory function for creating a new Pod for testing.
func newPod(podName string, nodeName string, label map[string]string) *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			GenerateName: podName,
			Labels:       label,
			Namespace:    api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			NodeName: nodeName,
			Containers: []api.Container{
				{
					Image: "foo/bar",
					TerminationMessagePath: api.TerminationMessagePathDefault,
					ImagePullPolicy:        api.PullIfNotPresent,
					SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
				},
			},
			DNSPolicy: api.DNSDefault,
		},
	}
	api.GenerateName(api.SimpleNameGenerator, &pod.ObjectMeta)
	return pod
}

// addPods is a helper function to add a number of pods to the pod store.
func addPods(podStore cache.Store, nodeName string, label map[string]string, number int) {
	for i := 0; i < number; i++ {
		podStore.Add(newPod(fmt.Sprintf("%s-", nodeName), nodeName, label))
	}
}

// newTestController creates a new DaemonSetsController for testing and a fake pod control.
func newTestController() (*DaemonSetsController, *controller.FakePodControl) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewDaemonSetsControllerFromClient(clientset, controller.NoResyncPeriodFunc, 0)
	manager.podStoreSynced = alwaysReady
	podControl := &controller.FakePodControl{}
	manager.podControl = podControl
	return manager, podControl
}

// validateSyncDaemonSets is a helper function to validate the number of creates and deletes from a sync.
func validateSyncDaemonSets(t *testing.T, fakePodControl *controller.FakePodControl, expectedCreates, expectedDeletes int) {
	if len(fakePodControl.Templates) != expectedCreates {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", expectedCreates, len(fakePodControl.Templates))
	}
	if len(fakePodControl.DeletePodName) != expectedDeletes {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", expectedDeletes, len(fakePodControl.DeletePodName))
	}
}

// syncAndValidateDaemonSets is a helper function that syncs a DaemonSet and validates the outcome.
func syncAndValidateDaemonSets(t *testing.T, manager *DaemonSetsController, ds *extensions.DaemonSet, podControl *controller.FakePodControl, expectedCreates, expectedDeletes int) {
	key, err := controller.KeyFunc(ds)
	if err != nil {
		t.Errorf("Could not get key for daemon.")
	}
	manager.syncHandler(key)
	validateSyncDaemonSets(t, podControl, expectedCreates, expectedDeletes)
}

// TestSimpleDaemonSetLaunchesPods tests that a DaemonSet with no node selector launches pods on all nodes.
func TestSimpleDaemonSetLaunchesPods(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 5, 0)
}

// TestNoNodesDoesNothing tests that a DaemonSet does nothing when there are no nodes.
func TestNoNodesDoesNothing(t *testing.T) {
	manager, podControl := newTestController()
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// TestOneNodeDaemonLaunchesPod tests that a DaemonSet with no node selector launches a pod on a single-node cluster.
func TestOneNodeDaemonLaunchesPod(t *testing.T) {
	manager, podControl := newTestController()
	manager.nodeStore.Add(newNode("only-node", nil))
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// TestNotReadNodeDaemonDoesNotLaunchPod tests that a DaemonSet *does* launch a pod on a NotReady node.
// DaemonSet pods are expected to run on all nodes that are schedulable, regardless of their readiness state.
func TestNotReadNodeDaemonDoesNotLaunchPod(t *testing.T) {
	manager, podControl := newTestController()
	node := newNode("not-ready", nil)
	node.Status = api.NodeStatus{
		Conditions: []api.NodeCondition{
			{Type: api.NodeReady, Status: api.ConditionFalse},
		},
	}
	manager.nodeStore.Add(node)
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// TestOutOfDiskNodeDaemonDoesNotLaunchPod tests that a DaemonSet does not launch a pod on a node that is out of disk.
func TestOutOfDiskNodeDaemonDoesNotLaunchPod(t *testing.T) {
	manager, podControl := newTestController()
	node := newNode("not-enough-disk", nil)
	node.Status.Conditions = []api.NodeCondition{{Type: api.NodeOutOfDisk, Status: api.ConditionTrue}}
	manager.nodeStore.Add(node)
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// TestInsufficentCapacityNodeDaemonDoesNotLaunchPod tests that a DaemonSet does not launch a pod on a node with insufficient resources.
func TestInsufficentCapacityNodeDaemonDoesNotLaunchPod(t *testing.T) {
	podSpec := api.PodSpec{
		NodeName: "too-much-mem",
		Containers: []api.Container{{
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceMemory: resource.MustParse("75M"),
					api.ResourceCPU:    resource.MustParse("75m"),
				},
			},
		}},
	}
	manager, podControl := newTestController()
	node := newNode("too-much-mem", nil)
	node.Status.Allocatable = api.ResourceList{
		api.ResourceMemory: resource.MustParse("100M"),
		api.ResourceCPU:    resource.MustParse("200m"),
	}
	manager.nodeStore.Add(node)
	manager.podStore.Add(&api.Pod{
		Spec: podSpec,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// TestSufficentCapacityWithTerminatedPodsDaemonLaunchesPod tests that a DaemonSet launches a pod on a node with sufficient resources,
// ignoring terminated pods when calculating resource usage.
func TestSufficentCapacityWithTerminatedPodsDaemonLaunchesPod(t *testing.T) {
	podSpec := api.PodSpec{
		NodeName: "too-much-mem",
		Containers: []api.Container{{
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceMemory: resource.MustParse("75M"),
					api.ResourceCPU:    resource.MustParse("75m"),
				},
			},
		}},
	}
	manager, podControl := newTestController()
	node := newNode("too-much-mem", nil)
	node.Status.Allocatable = api.ResourceList{
		api.ResourceMemory: resource.MustParse("100M"),
		api.ResourceCPU:    resource.MustParse("200m"),
	}
	manager.nodeStore.Add(node)
	manager.podStore.Add(&api.Pod{
		Spec:   podSpec,
		Status: api.PodStatus{Phase: api.PodSucceeded},
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// TestSufficentCapacityNodeDaemonLaunchesPod tests that a DaemonSet launches a pod on a node with sufficient resources.
func TestSufficentCapacityNodeDaemonLaunchesPod(t *testing.T) {
	podSpec := api.PodSpec{
		NodeName: "not-too-much-mem",
		Containers: []api.Container{{
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceMemory: resource.MustParse("75M"),
					api.ResourceCPU:    resource.MustParse("75m"),
				},
			},
		}},
	}
	manager, podControl := newTestController()
	node := newNode("not-too-much-mem", nil)
	node.Status.Allocatable = api.ResourceList{
		api.ResourceMemory: resource.MustParse("200M"),
		api.ResourceCPU:    resource.MustParse("200m"),
	}
	manager.nodeStore.Add(node)
	manager.podStore.Add(&api.Pod{
		Spec: podSpec,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// TestPortConflictNodeDaemonDoesNotLaunchPod tests that a DaemonSet does not launch a pod on a node if there is a host port conflict.
func TestPortConflictNodeDaemonDoesNotLaunchPod(t *testing.T) {
	podSpec := api.PodSpec{
		NodeName: "port-conflict",
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 666,
			}},
		}},
	}
	manager, podControl := newTestController()
	node := newNode("port-conflict", nil)
	manager.nodeStore.Add(node)
	manager.podStore.Add(&api.Pod{
		Spec: podSpec,
	})

	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// TestPortConflictWithSameDaemonPodDoesNotDeletePod tests that if a host port conflict is with a pod from the same DaemonSet,
// the controller does not delete the existing pod. This prevents churn.
//
// Issue: https://github.com/kubernetes/kubernetes/issues/22309
func TestPortConflictWithSameDaemonPodDoesNotDeletePod(t *testing.T) {
	podSpec := api.PodSpec{
		NodeName: "port-conflict",
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 666,
			}},
		}},
	}
	manager, podControl := newTestController()
	node := newNode("port-conflict", nil)
	manager.nodeStore.Add(node)
	manager.podStore.Add(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels:    simpleDaemonSetLabel,
			Namespace: api.NamespaceDefault,
		},
		Spec: podSpec,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// TestNoPortConflictNodeDaemonLaunchesPod tests that a DaemonSet launches a pod on a node if there is no host port conflict.
func TestNoPortConflictNodeDaemonLaunchesPod(t *testing.T) {
	podSpec1 := api.PodSpec{
		NodeName: "no-port-conflict",
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 6661,
			}},
		}},
	}
	podSpec2 := api.PodSpec{
		NodeName: "no-port-conflict",
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 6662,
			}},
		}},
	}
	manager, podControl := newTestController()
	node := newNode("no-port-conflict", nil)
	manager.nodeStore.Add(node)
	manager.podStore.Add(&api.Pod{
		Spec: podSpec1,
	})
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec = podSpec2
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// TestPodIsNotDeletedByDaemonsetWithEmptyLabelSelector tests that a DaemonSet with an empty pod selector does not sync.
// This is a safety measure to prevent a misconfigured DaemonSet from matching and deleting all pods in a namespace.
//
// issue https://github.com/kubernetes/kubernetes/pull/23223
func TestPodIsNotDeletedByDaemonsetWithEmptyLabelSelector(t *testing.T) {
	manager, podControl := newTestController()
	manager.nodeStore.Store.Add(newNode("node1", nil))
	// Create pod not controlled by a daemonset.
	manager.podStore.Add(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels:    map[string]string{"bang": "boom"},
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			NodeName: "node1",
		},
	})

	// Create a misconfigured DaemonSet with an empty pod selector.
	ds := newDaemonSet("foo")
	ls := unversioned.LabelSelector{}
	ds.Spec.Selector = &ls
	ds.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
	manager.dsStore.Add(ds)

	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// TestDealsWithExistingPods tests the controller's ability to reconcile the state of pods on nodes.
// It should not create pods on nodes that already have a daemon pod, and it should remove excess pods
// from nodes that have more than one.
func TestDealsWithExistingPods(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	addPods(manager.podStore.Store, "node-1", simpleDaemonSetLabel, 1)
	addPods(manager.podStore.Store, "node-2", simpleDaemonSetLabel, 2)
	addPods(manager.podStore.Store, "node-3", simpleDaemonSetLabel, 5)
	addPods(manager.podStore.Store, "node-4", simpleDaemonSetLabel2, 2)
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 2, 5)
}

// TestSelectorDaemonLaunchesPods tests that a DaemonSet with a node selector launches pods only on nodes matching the selector.
func TestSelectorDaemonLaunchesPods(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	daemon := newDaemonSet("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager.dsStore.Add(daemon)
	syncAndValidateDaemonSets(t, manager, daemon, podControl, 3, 0)
}

// TestSelectorDaemonDeletesUnselectedPods tests that a DaemonSet with a node selector deletes pods from nodes that do not match the selector.
func TestSelectorDaemonDeletesUnselectedPods(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	addNodes(manager.nodeStore.Store, 5, 5, simpleNodeLabel)
	addPods(manager.podStore.Store, "node-0", simpleDaemonSetLabel2, 2)
	addPods(manager.podStore.Store, "node-1", simpleDaemonSetLabel, 3)
	addPods(manager.podStore.Store, "node-1", simpleDaemonSetLabel2, 1)
	addPods(manager.podStore.Store, "node-4", simpleDaemonSetLabel, 1)
	daemon := newDaemonSet("foo")
	daemon.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager.dsStore.Add(daemon)
	syncAndValidateDaemonSets(t, manager, daemon, podControl, 5, 4)
}

// TestSelectorDaemonDealsWithExistingPods tests that a DaemonSet with a node selector correctly reconciles pods on both matching and non-matching nodes.
func TestSelectorDaemonDealsWithExistingPods(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	addNodes(manager.nodeStore.Store, 5, 5, simpleNodeLabel)
	addPods(manager.podStore.Store, "node-0", simpleDaemonSetLabel, 1)
	addPods(manager.podStore.Store, "node-1", simpleDaemonSetLabel, 3)
	addPods(manager.podStore.Store, "node-1", simpleDaemonSetLabel2, 2)
	addPods(manager.podStore.Store, "node-2", simpleDaemonSetLabel, 4)
	addPods(manager.podStore.Store, "node-6", simpleDaemonSetLabel, 13)
	addPods(manager.podStore.Store, "node-7", simpleDaemonSetLabel2, 4)
	addPods(manager.podStore.Store, "node-9", simpleDaemonSetLabel, 1)
	addPods(manager.podStore.Store, "node-9", simpleDaemonSetLabel2, 1)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 3, 20)
}

// TestBadSelectorDaemonDoesNothing tests that a DaemonSet with a node selector that matches no nodes does not launch any pods.
func TestBadSelectorDaemonDoesNothing(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel2
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// TestNameDaemonSetLaunchesPods tests that a DaemonSet with a node name launches a pod only on the specified node.
func TestNameDaemonSetLaunchesPods(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeName = "node-0"
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// TestBadNameDaemonSetDoesNothing tests that a DaemonSet with a non-existent node name does not launch any pods.
func TestBadNameDaemonSetDoesNothing(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 5, nil)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeName = "node-10"
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// TestNameAndSelectorDaemonSetLaunchesPods tests that a DaemonSet with both a node name and a node selector launches a pod only if both match the same node.
func TestNameAndSelectorDaemonSetLaunchesPods(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	ds.Spec.Template.Spec.NodeName = "node-6"
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}

// TestInconsistentNameSelectorDaemonSetDoesNothing tests that a DaemonSet with a node name and a node selector that do not match the same node does not launch any pods.
func TestInconsistentNameSelectorDaemonSetDoesNothing(t *testing.T) {
	manager, podControl := newTestController()
	addNodes(manager.nodeStore.Store, 0, 4, nil)
	addNodes(manager.nodeStore.Store, 4, 3, simpleNodeLabel)
	ds := newDaemonSet("foo")
	ds.Spec.Template.Spec.NodeSelector = simpleNodeLabel
	ds.Spec.Template.Spec.NodeName = "node-0"
	manager.dsStore.Add(ds)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
}

// TestDSManagerNotReady tests that the controller requeues a DaemonSet if the pod store is not yet synced.
// This prevents the controller from making incorrect decisions based on incomplete information during startup.
func TestDSManagerNotReady(t *testing.T) {
	manager, podControl := newTestController()
	manager.podStoreSynced = func() bool { return false }
	addNodes(manager.nodeStore.Store, 0, 1, nil)

	// Simulates the ds reflector running before the pod reflector. We don't
	// want to end up creating daemon pods in this case until the pod reflector
	// has synced, so the ds manager should just requeue the ds.
	ds := newDaemonSet("foo")
	manager.dsStore.Add(ds)

	dsKey := getKey(ds, t)
	syncAndValidateDaemonSets(t, manager, ds, podControl, 0, 0)
	queueDS, _ := manager.queue.Get()
	if queueDS != dsKey {
		t.Fatalf("Expected to find key %v in queue, found %v", dsKey, queueDS)
	}

	manager.podStoreSynced = alwaysReady
	syncAndValidateDaemonSets(t, manager, ds, podControl, 1, 0)
}
