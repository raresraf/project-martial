/*
Copyright 2016 The Kubernetes Authors.

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

// This file contains integration tests for the attach/detach controller.
// It uses a fake cloud provider and fake volume plugins to simulate volume
// operations. The tests are designed to verify the controller's behavior
// in response to pod and node events, especially in scenarios where events
// might be missed. The Desired State of World Populator (DSWP) is a key
// component tested here, ensuring that the system eventually converges to
// the correct state even if some events are not processed.

package volume

import (
	"net/http/httptest"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	volumecache "k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
	"k8s.io/kubernetes/test/integration/framework"
)

// fakePodWithVol creates a simple pod object with a single volume for testing purposes.
// The pod is scheduled on a predefined node "node-sandbox".
func fakePodWithVol(namespace string) *v1.Pod {
	fakePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      "fakepod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-container",
					Image: "nginx",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "fake-mount",
							MountPath: "/var/www/html",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "fake-mount",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/www/html",
						},
					},
				},
			},
			NodeName: "node-sandbox",
		},
	}
	return fakePod
}

// TestPodDeletionWithDswp verifies that the attach-detach controller correctly
// handles pod deletion even if the deletion event is missed. It simulates a
// scenario where a pod is removed from the informer's cache without a corresponding
// event being sent. The test ensures that the Desired State of World Populator (DSWP)
// eventually reconciles the state and removes the pod from the list of pods to
// attach volumes to, preventing orphaned volumes.
func TestPodDeletionWithDswp(t *testing.T) {
	_, server, closeFn := framework.RunAMaster(nil)
	defer closeFn()
	namespaceName := "test-pod-deletion"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				volumehelper.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	testClient, ctrl, informers := createAdClients(ns, t, server, defaultSyncPeriod)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.Core().Nodes().Create(node); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	stopCh := make(chan struct{})
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(stopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(stopCh)
	go ctrl.Run(stopCh)

	waitToObservePods(t, podInformer, 1)
	podKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		t.Fatalf("MetaNamespaceKeyFunc failed with : %v", err)
	}

	podInformerObj, _, err := podInformer.GetStore().GetByKey(podKey)

	if err != nil {
		t.Fatalf("Pod not found in Pod Informer cache : %v", err)
	}

	waitForPodsInDSWP(t, ctrl.GetDesiredStateOfWorld())

	// let's stop pod events from getting triggered
	close(podStopCh)
	err = podInformer.GetStore().Delete(podInformerObj)
	if err != nil {
		t.Fatalf("Error deleting pod : %v", err)
	}

	waitToObservePods(t, podInformer, 0)
	// the populator loop turns every 1 minute
	time.Sleep(80 * time.Second)
	podsToAdd := ctrl.GetDesiredStateOfWorld().GetPodToAdd()
	if len(podsToAdd) != 0 {
		t.Fatalf("All pods should have been removed")
	}

	close(stopCh)
}

// TestPodUpdateWithWithADC verifies that when a pod's status is updated to
// a terminal phase (e.g., Succeeded), the Attach-Detach Controller (ADC)
// recognizes this change and removes the pod from the desired state of world.
// This ensures that volumes are detached from completed pods.
func TestPodUpdateWithWithADC(t *testing.T) {
	_, server, closeFn := framework.RunAMaster(nil)
	defer closeFn()
	namespaceName := "test-pod-update"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				volumehelper.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	testClient, ctrl, informers := createAdClients(ns, t, server, defaultSyncPeriod)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.Core().Nodes().Create(node); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	stopCh := make(chan struct{})
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(stopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(stopCh)
	go ctrl.Run(stopCh)

	waitToObservePods(t, podInformer, 1)
	podKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		t.Fatalf("MetaNamespaceKeyFunc failed with : %v", err)
	}

	_, _, err = podInformer.GetStore().GetByKey(podKey)

	if err != nil {
		t.Fatalf("Pod not found in Pod Informer cache : %v", err)
	}

	waitForPodsInDSWP(t, ctrl.GetDesiredStateOfWorld())

	pod.Status.Phase = v1.PodSucceeded

	if _, err := testClient.Core().Pods(ns.Name).UpdateStatus(pod); err != nil {
		t.Errorf("Failed to update pod : %v", err)
	}

	time.Sleep(20 * time.Second)
	podsToAdd := ctrl.GetDesiredStateOfWorld().GetPodToAdd()
	if len(podsToAdd) != 0 {
		t.Fatalf("All pods should have been removed")
	}

	close(podStopCh)
	close(stopCh)
}

// TestPodUpdateWithKeepTerminatedPodVolumes verifies that the attach-detach controller
// respects the `KeepTerminatedPodVolumesAnnotation`. When this annotation is present on a node,
// volumes associated with terminated pods (Succeeded or Failed) should not be detached.
// This is useful for debugging and data recovery.
func TestPodUpdateWithKeepTerminatedPodVolumes(t *testing.T) {
	_, server, closeFn := framework.RunAMaster(nil)
	defer closeFn()
	namespaceName := "test-pod-update"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				volumehelper.ControllerManagedAttachAnnotation:  "true",
				volumehelper.KeepTerminatedPodVolumesAnnotation: "true",
			},
		},
	}

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	testClient, ctrl, informers := createAdClients(ns, t, server, defaultSyncPeriod)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.Core().Nodes().Create(node); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	stopCh := make(chan struct{})
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(stopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(stopCh)
	go ctrl.Run(stopCh)

	waitToObservePods(t, podInformer, 1)
	podKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		t.Fatalf("MetaNamespaceKeyFunc failed with : %v", err)
	}

	_, _, err = podInformer.GetStore().GetByKey(podKey)

	if err != nil {
		t.Fatalf("Pod not found in Pod Informer cache : %v", err)
	}

	waitForPodsInDSWP(t, ctrl.GetDesiredStateOfWorld())

	pod.Status.Phase = v1.PodSucceeded

	if _, err := testClient.Core().Pods(ns.Name).UpdateStatus(pod); err != nil {
		t.Errorf("Failed to update pod : %v", err)
	}

	time.Sleep(20 * time.Second)
	podsToAdd := ctrl.GetDesiredStateOfWorld().GetPodToAdd()
	if len(podsToAdd) == 0 {
		t.Fatalf("The pod should not be removed if KeepTerminatedPodVolumesAnnotation is set")
	}

	close(podStopCh)
	close(stopCh)
}

// waitToObservePods is a helper function that blocks until the pod informer's
// cache contains the expected number of pods. This is crucial for ensuring that
// the test environment is stable before proceeding with actions that depend on
// the informer's state.
func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int) {
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) == podNum {
			return true, nil
		} else {
			return false, nil
		}
	}); err != nil {
		t.Fatal(err)
	}
}

// waitForPodsInDSWP is a helper function that blocks until at least one pod
// is present in the desired state of world cache. This indicates that the
// attach-detach controller has started processing pods.
func waitForPodsInDSWP(t *testing.T, dswp volumecache.DesiredStateOfWorld) {
	if err := wait.Poll(time.Millisecond*500, wait.ForeverTestTimeout, func() (bool, error) {
		pods := dswp.GetPodToAdd()
		if len(pods) > 0 {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatalf("Pod not added to desired state of world : %v", err)
	}
}

// createAdClients sets up the necessary clients, informers, and the
// attach-detach controller for integration testing. It uses a fake cloud
// provider and fake volume plugins to isolate the controller's logic.
func createAdClients(ns *v1.Namespace, t *testing.T, server *httptest.Server, syncPeriod time.Duration) (*clientset.Clientset, attachdetach.AttachDetachController, informers.SharedInformerFactory) {
	config := restclient.Config{
		Host:          server.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion},
		QPS:           1000000,
		Burst:         1000000,
	}
	resyncPeriod := 12 * time.Hour
	testClient := clientset.NewForConfigOrDie(&config)

	host := volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil)
	plugin := &volumetest.FakeVolumePlugin{
		PluginName:             provisionerPluginName,
		Host:                   host,
		Config:                 volume.VolumeConfig{},
		LastProvisionerOptions: volume.VolumeOptions{},
		NewAttacherCallCount:   0,
		NewDetacherCallCount:   0,
		Mounters:               nil,
		Unmounters:             nil,
		Attachers:              nil,
		Detachers:              nil,
	}
	plugins := []volume.VolumePlugin{plugin}
	cloud := &fakecloud.FakeCloud{}
	informers := informers.NewSharedInformerFactory(testClient, resyncPeriod)
	ctrl, err := attachdetach.NewAttachDetachController(
		testClient,
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		informers.Core().V1().PersistentVolumeClaims(),
		informers.Core().V1().PersistentVolumes(),
		cloud,
		plugins,
		false,
		time.Second*5)

	if err != nil {
		t.Fatalf("Error creating AttachDetach : %v", err)
	}
	return testClient, ctrl, informers
}

// TestPodAddedByDswp verifies that if a pod creation event is missed by the
// attach-detach controller, the Desired State of World Populator (DSWP) will
// eventually discover the pod and add it to the desired state. This ensures
// that volumes for the pod are correctly attached even in the face of missed events.
func TestPodAddedByDswp(t *testing.T) {
	_, server, closeFn := framework.RunAMaster(nil)
	defer closeFn()
	namespaceName := "test-pod-deletion"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				volumehelper.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	testClient, ctrl, informers := createAdClients(ns, t, server, defaultSyncPeriod)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.Core().Nodes().Create(node); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	stopCh := make(chan struct{})
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(stopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(stopCh)
	go ctrl.Run(stopCh)

	waitToObservePods(t, podInformer, 1)
	podKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		t.Fatalf("MetaNamespaceKeyFunc failed with : %v", err)
	}

	_, _, err = podInformer.GetStore().GetByKey(podKey)

	if err != nil {
		t.Fatalf("Pod not found in Pod Informer cache : %v", err)
	}

	waitForPodsInDSWP(t, ctrl.GetDesiredStateOfWorld())

	// let's stop pod events from getting triggered
	close(podStopCh)
	podObj, err := api.Scheme.DeepCopy(pod)
	if err != nil {
		t.Fatalf("Error copying pod : %v", err)
	}
	podNew, ok := podObj.(*v1.Pod)
	if !ok {
		t.Fatalf("Error converting pod : %v", err)
	}
	newPodName := "newFakepod"
	podNew.SetName(newPodName)
	err = podInformer.GetStore().Add(podNew)
	if err != nil {
		t.Fatalf("Error adding pod : %v", err)
	}

	waitToObservePods(t, podInformer, 2)
	// the findAndAddActivePods loop turns every 3 minute
	time.Sleep(200 * time.Second)
	podsToAdd := ctrl.GetDesiredStateOfWorld().GetPodToAdd()
	if len(podsToAdd) != 2 {
		t.Fatalf("DSW should have two pods")
	}

	close(stopCh)
}
