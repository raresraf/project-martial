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

/**
 * @file attach_detach_test.go
 * @brief Integration tests for the Attach/Detach controller's synchronization logic.
 * 
 * Functional Intent: Validates that the Attach/Detach controller maintains a 
 * consistent state between the API server and the physical nodes, even in the 
 * presence of missed watch events. It specifically tests the 'Desired State 
 * of World' (DSW) populator's ability to recover from discrepancies and ensure 
 * volumes are correctly attached or detached based on pod lifecycle transitions.
 * 
 * Domain: Production Systems, Distributed Control Loops, Storage Orchestration.
 */

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

/**
 * fakePodWithVol - Constructs a mock pod specification with a defined host-path volume.
 */
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

/**
 * TestPodDeletionWithDswp - Verifies automated cleanup of missed pod deletion events.
 * 
 * Algorithm: Missed event recovery via periodic reconciliation.
 * Logic: 
 * 1. Simulates pod creation and confirms tracking in DSW.
 * 2. Stops the event stream and manually deletes the pod from the informer store (mimicking a missed delete event).
 * 3. Waits for the periodic populator loop to detect the discrepancy and purge the DSW entry.
 */
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

	// Block Logic: Background controller orchestration.
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

	// Block Logic: Missed event simulation.
	// Logic: Stops the watch channel and surgically removes the pod from local cache.
	close(podStopCh)
	err = podInformer.GetStore().Delete(podInformerObj)
	if err != nil {
		t.Fatalf("Error deleting pod : %v", err)
	}

	waitToObservePods(t, podInformer, 0)
	
	// Synchronization: Waits for the next populator cycle (nominal 1-minute interval).
	// Invariant: Reconciliation must eventually clear the DSW state.
	time.Sleep(80 * time.Second)
	podsToAdd := ctrl.GetDesiredStateOfWorld().GetPodToAdd()
	if len(podsToAdd) != 0 {
		t.Fatalf("All pods should have been removed")
	}

	close(stopCh)
}

/**
 * TestPodUpdateWithWithADC - Validates volume cleanup upon pod successful completion.
 */
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

	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(podStopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(podStopCh)
	go ctrl.Run(podStopCh)

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

	// Logic: Transition pod to 'Succeeded' state.
	pod.Status.Phase = v1.PodSucceeded

	if _, err := testClient.Core().Pods(ns.Name).UpdateStatus(pod); err != nil {
		t.Errorf("Failed to update pod : %v", err)
	}

	// Invariant: Controller should detect termination and remove volumes from DSW.
	time.Sleep(20 * time.Second)
	podsToAdd := ctrl.GetDesiredStateOfWorld().GetPodToAdd()
	if len(podsToAdd) != 0 {
		t.Fatalf("All pods should have been removed")
	}

	close(podStopCh)
}

/**
 * TestPodUpdateWithKeepTerminatedPodVolumes - Tests policy-based volume preservation.
 * 
 * Functional Intent: Ensures that volumes are NOT detached if the node-level 
 * 'KeepTerminatedPodVolumes' annotation is enabled.
 */
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

	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(podStopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(podStopCh)
	go ctrl.Run(podStopCh)

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

	// Invariant: Volumes must persist in DSW due to the preservation policy.
	time.Sleep(20 * time.Second)
	podsToAdd := ctrl.GetDesiredStateOfWorld().GetPodToAdd()
	if len(podsToAdd) == 0 {
		t.Fatalf("The pod should not be removed if KeepTerminatedPodVolumesAnnotation is set")
	}

	close(podStopCh)
}

/**
 * waitToObservePods - Polling helper for informer cache synchronization.
 */
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

/**
 * waitForPodsInDSWP - Polling helper for DSW state synchronization.
 */
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

/**
 * createAdClients - Bootstraps the simulated environment with fake cloud and volume plugins.
 */
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

/**
 * TestPodAddedByDswp - Verifies automated detection of missed pod creation events.
 */
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

	// Block Logic: Missed creation event.
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
	
	// Logic: Inject pod into store without triggering a watch event.
	err = podInformer.GetStore().Add(podNew)
	if err != nil {
		t.Fatalf("Error adding pod : %v", err)
	}

	waitToObservePods(t, podInformer, 2)
	
	// Synchronization: Waits for the next 'findAndAddActivePods' reconciliation cycle (3 mins).
	// Invariant: DSW must eventually include the manually injected pod.
	time.Sleep(200 * time.Second)
	podsToAdd := ctrl.GetDesiredStateOfWorld().GetPodToAdd()
	if len(podsToAdd) != 2 {
		t.Fatalf("DSW should have two pods")
	}

	close(stopCh)
}
