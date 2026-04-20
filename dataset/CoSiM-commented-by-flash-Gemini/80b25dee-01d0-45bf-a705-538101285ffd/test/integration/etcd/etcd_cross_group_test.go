/*
Copyright 2019 The Kubernetes Authors.

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
 * @80b25dee-01d0-45bf-a705-538101285ffd/test/integration/etcd/etcd_cross_group_test.go
 * @brief Integration tests for cross-group resource storage compatibility in etcd.
 * 
 * Functional Intent: Ensures that Kubernetes objects shared across different API groups 
 * (e.g., resources that moved from 'extensions' to 'apps') maintain storage consistency. 
 * It validates that writing a resource via one API group correctly triggers watch 
 * notifications and is readable via all other associated API groups, preserving 
 * the correct versioning metadata.
 * 
 * Domain: Kubernetes API Server, etcd Storage, Multi-version API Compatibility.
 */

package etcd

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
)

/**
 * TestCrossGroupStorage - Orchestrates the storage interoperability test suite.
 * Logic: 
 * 1. Starts a live API server and initializes a test namespace.
 * 2. Scans all available resources to identify "cross-group" candidates (GVKs present 
 *    in multiple API groups).
 * 3. For each candidate group:
 *    - Creates a baseline object.
 *    - Establishes watches and clients for every API group that shares the storage path.
 *    - Sequentially writes the object into etcd using each group's version.
 *    - Verifies that all watchers receive the correct versioned event and all 
 *      clients retrieve the object in their respective expected version.
 */
func TestCrossGroupStorage(t *testing.T) {
	apiServer := StartRealAPIServerOrDie(t, func(opts *options.ServerRunOptions) {
		// force enable all resources so we can check storage.
	})
	defer apiServer.Cleanup()

	etcdStorageData := GetEtcdStorageData()

	crossGroupResources := map[schema.GroupVersionKind][]Resource{}

	apiServer.Client.CoreV1().Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}, metav1.CreateOptions{})

	// Block Logic: Identification of cross-group resource clusters.
	// Logic: Maps each system resource to its persisted GVK in etcd. Identifies 
	// clusters where multiple API groups map to the same underlying storage GVK.
	for _, resourceToPersist := range apiServer.Resources {
		gvk := resourceToPersist.Mapping.GroupVersionKind
		data, exists := etcdStorageData[resourceToPersist.Mapping.Resource]
		if !exists {
			continue
		}
		storageGVK := gvk
		if data.ExpectedGVK != nil {
			storageGVK = *data.ExpectedGVK
		}
		crossGroupResources[storageGVK] = append(crossGroupResources[storageGVK], resourceToPersist)
	}

	// Block Logic: Filtering for true cross-group overlaps.
	// Invariant: Only retains GVK clusters that originate from at least two distinct API groups.
	for gvk, resources := range crossGroupResources {
		groups := sets.NewString()
		for _, resource := range resources {
			groups.Insert(resource.Mapping.GroupVersionKind.Group)
		}
		if len(groups) < 2 {
			delete(crossGroupResources, gvk)
		}
	}

	if len(crossGroupResources) == 0 {
		t.Fatal("no cross-group resources found")
	}

	// Block Logic: Cross-version interoperability validation loop.
	for gvk, resources := range crossGroupResources {
		t.Run(gvk.String(), func(t *testing.T) {
			resource := resources[0]

			ns := ""
			if resource.Mapping.Scope.Name() == meta.RESTScopeNameNamespace {
				ns = testNamespace
			}

			data := etcdStorageData[resource.Mapping.Resource]
			resourceClient, obj, err := JSONToUnstructured(data.Stub, ns, resource.Mapping, apiServer.Dynamic)
			if err != nil {
				t.Fatal(err)
			}
			actual, err := resourceClient.Create(context.TODO(), obj, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}
			name := actual.GetName()

			// Block Logic: Infrastructure setup for multi-group monitoring.
			// Logic: Initializes parallel clients and long-running watches for every overlapping API version.
			var (
				clients       = map[schema.GroupVersionResource]dynamic.ResourceInterface{}
				versionedData = map[schema.GroupVersionResource]*unstructured.Unstructured{}
				watches       = map[schema.GroupVersionResource]watch.Interface{}
			)
			for _, resource := range resources {
				clients[resource.Mapping.Resource] = apiServer.Dynamic.Resource(resource.Mapping.Resource).Namespace(ns)
				versionedData[resource.Mapping.Resource], err = clients[resource.Mapping.Resource].Get(context.TODO(), name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("error finding resource via %s: %v", resource.Mapping.Resource.GroupVersion().String(), err)
				}
				watches[resource.Mapping.Resource], err = clients[resource.Mapping.Resource].Watch(context.TODO(), metav1.ListOptions{ResourceVersion: actual.GetResourceVersion()})
				if err != nil {
					t.Fatalf("error opening watch via %s: %v", resource.Mapping.Resource.GroupVersion().String(), err)
				}
			}

			versioner := etcd3.APIObjectVersioner{}
			// Block Logic: Round-robin storage updates.
			// Logic: For each API version, prepare the object for storage (stripping runtime-only fields), 
			// write it directly to etcd, and verify the resulting system state.
			for _, resource := range resources {
				versioned := versionedData[resource.Mapping.Resource]
				versioner.PrepareObjectForStorage(versioned)
				versionedJSON, err := versioned.MarshalJSON()
				if err != nil {
					t.Error(err)
					continue
				}

				if _, err := apiServer.KV.Put(context.Background(), data.ExpectedEtcdPath, string(versionedJSON)); err != nil {
					t.Error(err)
					continue
				}
				t.Logf("wrote %s to etcd", resource.Mapping.Resource.GroupVersion().String())

				// Block Logic: Watch propagation verification.
				// Invariant: Every versioned watcher must receive exactly one 'Modified' 
				// event where the object's GroupVersion matches the watcher's version.
				for watchResource, watcher := range watches {
					select {
					case event, ok := <-watcher.ResultChan():
						if !ok {
							t.Fatalf("watch of %s closed in response to persisting %s", watchResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String())
						}
						if event.Type != watch.Modified {
							eventJSON, _ := json.Marshal(event)
							t.Errorf("unexpected watch event sent to watch of %s in response to persisting %s: %s", watchResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String(), string(eventJSON))
							continue
						}
						if event.Object.GetObjectKind().GroupVersionKind().GroupVersion() != watchResource.GroupVersion() {
							t.Errorf("unexpected group version object sent to watch of %s in response to persisting %s: %#v", watchResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String(), event.Object)
							continue
						}
						t.Logf("     received event for %s", watchResource.GroupVersion().String())
					case <-time.After(30 * time.Second):
						t.Errorf("timed out waiting for watch event for %s in response to persisting %s", watchResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String())
						continue
					}
				}

				// Block Logic: Direct read verification.
				// Invariant: Every versioned client must be able to read the object back 
				// and receive it in their specific API version, regardless of which 
				// version was used to perform the etcd write.
				for clientResource, client := range clients {
					obj, err := client.Get(context.TODO(), name, metav1.GetOptions{})
					if err != nil {
						t.Errorf("error looking up %s after persisting %s", clientResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String())
						continue
					}
					if obj.GetObjectKind().GroupVersionKind().GroupVersion() != clientResource.GroupVersion() {
						t.Errorf("unexpected group version retrieved from %s after persisting %s: %#v", clientResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String(), obj)
						continue
					}
					t.Logf("     fetched object for %s", clientResource.GroupVersion().String())
				}
			}
		})
	}
}
