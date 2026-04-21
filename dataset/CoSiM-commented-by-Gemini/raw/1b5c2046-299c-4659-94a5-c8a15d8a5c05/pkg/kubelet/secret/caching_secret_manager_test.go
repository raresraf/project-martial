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

package secret

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/fake"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	core "k8s.io/client-go/testing"

	"github.com/stretchr/testify/assert"
)

// checkSecret is a test helper that verifies if a secret is correctly registered and
// accessible (or properly missing) in the store.
func checkSecret(t *testing.T, store *secretStore, ns, name string, shouldExist bool) {
	_, err := store.Get(ns, name)
	if shouldExist && err != nil {
		t.Errorf("unexpected actions: %#v", err)
	}
	if !shouldExist && (err == nil || !strings.Contains(err.Error(), fmt.Sprintf("secret %q/%q not registered", ns, name))) {
		t.Errorf("unexpected actions: %#v", err)
	}
}

// noObjectTTL is a mock GetObjectTTLFunc that disables custom TTLs.
func noObjectTTL() (time.Duration, bool) {
	return time.Duration(0), false
}

// TestSecretStore verifies the core functionality of secretStore, including
// registration (Add), removal (Delete), and basic retrieval (Get).
func TestSecretStore(t *testing.T) {
	fakeClient := &fake.Clientset{}
	store := newSecretStore(fakeClient, clock.RealClock{}, noObjectTTL, 0)
	store.Add("ns1", "name1")
	store.Add("ns2", "name2")
	store.Add("ns1", "name1")
	store.Add("ns1", "name1")
	store.Delete("ns1", "name1")
	store.Delete("ns2", "name2")
	store.Add("ns3", "name3")

	// Verify that Add/Delete operations do not trigger premature API calls.
	actions := fakeClient.Actions()
	assert.Equal(t, 0, len(actions), "unexpected actions: %#v", actions)
	
	// Ensure Get triggers a fetch for registered secrets.
	store.Get("ns1", "name1")
	// Ensure Get fails gracefully for unregistered secrets.
	store.Get("ns2", "name2")
	store.Get("ns3", "name3")

	actions = fakeClient.Actions()
	assert.Equal(t, 2, len(actions), "unexpected actions: %#v", actions)

	for _, a := range actions {
		assert.True(t, a.Matches("get", "secrets"), "unexpected actions: %#v", a)
	}

	checkSecret(t, store, "ns1", "name1", true)
	checkSecret(t, store, "ns2", "name2", false)
	checkSecret(t, store, "ns3", "name3", true)
	checkSecret(t, store, "ns4", "name4", false)
}

// TestSecretStoreDeletingSecret ensures that the cache correctly handles scenarios
// where a secret is deleted from the underlying API server.
func TestSecretStoreDeletingSecret(t *testing.T) {
	fakeClient := &fake.Clientset{}
	store := newSecretStore(fakeClient, clock.RealClock{}, noObjectTTL, 0)
	store.Add("ns", "name")

	result := &v1.Secret{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "name", ResourceVersion: "10"}}
	fakeClient.AddReactor("get", "secrets", func(action core.Action) (bool, runtime.Object, error) {
		return true, result, nil
	})
	secret, err := store.Get("ns", "name")
	assert.NoError(t, err)
	assert.Equal(t, result, secret)

	// Simulate secret deletion in the API server.
	fakeClient.PrependReactor("get", "secrets", func(action core.Action) (bool, runtime.Object, error) {
		return true, &v1.Secret{}, apierrors.NewNotFound(v1.Resource("secret"), "name")
	})
	secret, err = store.Get("ns", "name")
	assert.True(t, apierrors.IsNotFound(err))
	assert.Equal(t, &v1.Secret{}, secret)
}

// TestSecretStoreGetAlwaysRefresh simulates high concurrency with zero TTL
// to verify that every Get request results in an API call.
func TestSecretStoreGetAlwaysRefresh(t *testing.T) {
	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Now())
	store := newSecretStore(fakeClient, fakeClock, noObjectTTL, 0)

	for i := 0; i < 10; i++ {
		store.Add(fmt.Sprintf("ns-%d", i), fmt.Sprintf("name-%d", i))
	}
	fakeClient.ClearActions()

	wg := sync.WaitGroup{}
	wg.Add(100)
	for i := 0; i < 100; i++ {
		go func(i int) {
			store.Get(fmt.Sprintf("ns-%d", i%10), fmt.Sprintf("name-%d", i%10))
			wg.Done()
		}(i)
	}
	wg.Wait()
	actions := fakeClient.Actions()
	assert.Equal(t, 100, len(actions), "unexpected actions: %#v", actions)
}

// TestSecretStoreGetNeverRefresh verifies that secrets are served from cache
// and don't trigger redundant API calls within their TTL.
func TestSecretStoreGetNeverRefresh(t *testing.T) {
	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Now())
	store := newSecretStore(fakeClient, fakeClock, noObjectTTL, time.Minute)

	for i := 0; i < 10; i++ {
		store.Add(fmt.Sprintf("ns-%d", i), fmt.Sprintf("name-%d", i))
	}
	fakeClient.ClearActions()

	wg := sync.WaitGroup{}
	wg.Add(100)
	for i := 0; i < 100; i++ {
		go func(i int) {
			store.Get(fmt.Sprintf("ns-%d", i%10), fmt.Sprintf("name-%d", i%10))
			wg.Done()
		}(i)
	}
	wg.Wait()
	actions := fakeClient.Actions()
	// Only the initial Get for each of the 10 secrets should hit the API.
	assert.Equal(t, 10, len(actions), "unexpected actions: %#v", actions)
}

// TestCustomTTL validates that dynamically provided TTLs (e.g., from Node annotations)
// are respected by the cache.
func TestCustomTTL(t *testing.T) {
	ttl := time.Duration(0)
	ttlExists := false
	customTTL := func() (time.Duration, bool) {
		return ttl, ttlExists
	}

	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Time{})
	store := newSecretStore(fakeClient, fakeClock, customTTL, time.Minute)

	store.Add("ns", "name")
	store.Get("ns", "name")
	fakeClient.ClearActions()

	// Verify 0-TTL triggers immediate refresh.
	ttl = time.Duration(0)
	ttlExists = true
	store.Get("ns", "name")
	actions := fakeClient.Actions()
	assert.Equal(t, 1, len(actions), "unexpected actions: %#v", actions)
	fakeClient.ClearActions()

	// Verify 5-minute TTL prevents refresh until expiration.
	ttl = time.Duration(5) * time.Minute
	store.Get("ns", "name")
	actions = fakeClient.Actions()
	assert.Equal(t, 0, len(actions), "unexpected actions: %#v", actions)
	
	fakeClock.Step(4 * time.Minute)
	store.Get("ns", "name")
	assert.Equal(t, 0, len(fakeClient.Actions()))
	
	fakeClock.Step(time.Minute)
	store.Get("ns", "name")
	assert.Equal(t, 1, len(fakeClient.Actions()))
}

// TestParseNodeAnnotation verifies the logic for extracting TTL durations
// from Kubernetes Node annotations.
func TestParseNodeAnnotation(t *testing.T) {
	testCases := []struct {
		node   *v1.Node
		err    error
		exists bool
		ttl    time.Duration
	}{
		{ node: nil, err: fmt.Errorf("error"), exists: false },
		{ node: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}}, exists: false },
		{ node: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node", Annotations: map[string]string{v1.ObjectTTLAnnotationKey: "bad"}}}, exists: false },
		{ node: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node", Annotations: map[string]string{v1.ObjectTTLAnnotationKey: "60"}}}, exists: true, ttl: time.Minute },
	}
	for i, tc := range testCases {
		getNode := func() (*v1.Node, error) { return tc.node, tc.err }
		ttl, exists := GetObjectTTLFromNodeFunc(getNode)()
		assert.Equal(t, tc.exists, exists, "case %d", i)
		if exists {
			assert.Equal(t, tc.ttl, ttl, "case %d", i)
		}
	}
}

// podWithSecrets is a test helper that constructs a Pod object referencing various secrets.
func podWithSecrets(ns, podName string, toAttach secretsToAttach) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: ns, Name: podName},
		Spec: v1.PodSpec{},
	}
	for _, name := range toAttach.imagePullSecretNames {
		pod.Spec.ImagePullSecrets = append(pod.Spec.ImagePullSecrets, v1.LocalObjectReference{Name: name})
	}
	for i, secrets := range toAttach.containerEnvSecrets {
		container := v1.Container{Name: fmt.Sprintf("container-%d", i)}
		for _, name := range secrets.envFromNames {
			container.EnvFrom = append(container.EnvFrom, v1.EnvFromSource{
				SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
			})
		}
		for _, name := range secrets.envVarNames {
			container.Env = append(container.Env, v1.EnvVar{
				ValueFrom: &v1.EnvVarSource{SecretKeyRef: &v1.SecretKeySelector{LocalObjectReference: v1.LocalObjectReference{Name: name}}},
			})
		}
		pod.Spec.Containers = append(pod.Spec.Containers, container)
	}
	return pod
}

// TestCacheInvalidation ensures that updating a pod correctly invalidates
// cached secrets to ensure data consistency.
func TestCacheInvalidation(t *testing.T) {
	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Now())
	store := newSecretStore(fakeClient, fakeClock, noObjectTTL, time.Minute)
	manager := newCacheBasedSecretManager(store)

	s1 := secretsToAttach{
		imagePullSecretNames: []string{"s1"},
		containerEnvSecrets: []envSecrets{{envVarNames: []string{"s1"}, envFromNames: []string{"s10"}}, {envVarNames: []string{"s2"}}},
	}
	manager.RegisterPod(podWithSecrets("ns1", "name1", s1))
	store.Get("ns1", "s1")
	store.Get("ns1", "s10")
	store.Get("ns1", "s2")
	fakeClient.ClearActions()

	// Update pod with new secret references.
	s2 := secretsToAttach{
		imagePullSecretNames: []string{"s1"},
		containerEnvSecrets: []envSecrets{{envVarNames: []string{"s1"}}, {envVarNames: []string{"s2"}, envFromNames: []string{"s20"}}, {envVarNames: []string{"s3"}}},
	}
	manager.RegisterPod(podWithSecrets("ns1", "name1", s2))
	
	// Ensure that subsequent Gets trigger refreshes due to invalidation.
	store.Get("ns1", "s1")
	store.Get("ns1", "s2")
	store.Get("ns1", "s20")
	store.Get("ns1", "s3")
	assert.Equal(t, 4, len(fakeClient.Actions()))
}

// TestCacheRefcounts validates the reference-counting logic, ensuring secrets
// are only evicted from cache when no pods reference them.
func TestCacheRefcounts(t *testing.T) {
	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Now())
	store := newSecretStore(fakeClient, fakeClock, noObjectTTL, time.Minute)
	manager := newCacheBasedSecretManager(store)

	s1 := secretsToAttach{
		imagePullSecretNames: []string{"s1"},
		containerEnvSecrets: []envSecrets{{envVarNames: []string{"s1"}, envFromNames: []string{"s10"}}, {envVarNames: []string{"s2"}}, {envVarNames: []string{"s3"}}},
	}
	manager.RegisterPod(podWithSecrets("ns1", "name1", s1))
	manager.RegisterPod(podWithSecrets("ns1", "name2", s1))
	
	s2 := secretsToAttach{
		imagePullSecretNames: []string{"s2"},
		containerEnvSecrets: []envSecrets{{envVarNames: []string{"s4"}}, {envVarNames: []string{"s5"}, envFromNames: []string{"s50"}}},
	}
	manager.RegisterPod(podWithSecrets("ns1", "name2", s2))
	
	// Helper to check reference count for a secret in the cache.
	refs := func(ns, name string) int {
		store.lock.Lock()
		defer store.lock.Unlock()
		if item, ok := store.items[objectKey{ns, name}]; ok { return item.refCount }
		return 0
	}
	
	assert.Equal(t, 2, refs("ns1", "s1"))
	assert.Equal(t, 1, refs("ns1", "s2"))
}

// TestCachingSecretManager provides a comprehensive end-to-end test for the
// caching manager's lifecycle.
func TestCachingSecretManager(t *testing.T) {
	fakeClient := &fake.Clientset{}
	secretStore := newSecretStore(fakeClient, clock.RealClock{}, noObjectTTL, 0)
	manager := newCacheBasedSecretManager(secretStore)

	s1 := secretsToAttach{
		imagePullSecretNames: []string{"s1"},
		containerEnvSecrets: []envSecrets{{envVarNames: []string{"s1"}}, {envVarNames: []string{"s2"}}, {envFromNames: []string{"s20"}}},
	}
	manager.RegisterPod(podWithSecrets("ns1", "name1", s1))
	
	s2 := secretsToAttach{
		imagePullSecretNames: []string{"s1"},
		containerEnvSecrets: []envSecrets{{envVarNames: []string{"s3"}}, {envVarNames: []string{"s4"}}, {envFromNames: []string{"s40"}}},
	}
	manager.RegisterPod(podWithSecrets("ns1", "name1", s2))
	manager.RegisterPod(podWithSecrets("ns2", "name2", s2))

	// Verify existence and correct namespace isolation.
	checkSecret(t, secretStore, "ns1", "s1", true)
	checkSecret(t, secretStore, "ns1", "s3", true)
	checkSecret(t, secretStore, "ns1", "s2", false) // Should have been removed after update
}
