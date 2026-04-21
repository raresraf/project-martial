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
	"strconv"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	storageetcd "k8s.io/apiserver/pkg/storage/etcd"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/kubelet/util"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
)

const (
	// defaultTTL defines the fallback duration for which a secret is considered valid
	// in the cache if no node-specific TTL is configured.
	defaultTTL = time.Minute
)

// GetObjectTTLFunc is a function signature for retrieving the desired Time-To-Live
// for cached objects, typically sourced from node annotations.
type GetObjectTTLFunc func() (time.Duration, bool)

// secretStoreItem represents a single entry in the secret cache, maintaining
// a reference count to determine its survival in the cache.
type secretStoreItem struct {
	// refCount tracks how many registered pods currently reference this secret.
	refCount int
	// secret points to the actual data and metadata for the cached secret.
	secret   *secretData
}

// secretData encapsulates the state of a cached Secret, including its value,
// any retrieval errors, and timing information for expiration logic.
type secretData struct {
	sync.Mutex

	// secret is the cached Kubernetes Secret resource.
	secret         *v1.Secret
	// err stores the error from the last attempt to fetch this secret.
	err            error
	// lastUpdateTime records when the cache was last refreshed from the API server.
	lastUpdateTime time.Time
}

// secretStore implements a reference-counted local cache for Secrets.
// It optimizes performance by reducing redundant API calls while ensuring
// that data is refreshed according to a configurable TTL policy.
type secretStore struct {
	kubeClient clientset.Interface
	clock      clock.Clock

	lock  sync.Mutex
	// items maps object keys to their corresponding cached items.
	items map[objectKey]*secretStoreItem

	// defaultTTL is the baseline expiration duration.
	defaultTTL time.Duration
	// getTTL is an optional hook to override the TTL dynamically.
	getTTL     GetObjectTTLFunc
}

// newSecretStore initializes a new secretStore with the provided client and timing configuration.
func newSecretStore(kubeClient clientset.Interface, clock clock.Clock, getTTL GetObjectTTLFunc, ttl time.Duration) *secretStore {
	return &secretStore{
		kubeClient: kubeClient,
		clock:      clock,
		items:      make(map[objectKey]*secretStoreItem),
		defaultTTL: ttl,
		getTTL:     getTTL,
	}
}

// isSecretOlder compares two Secret versions using their ResourceVersion.
// It returns true if the 'newSecret' is strictly older than 'oldSecret'.
func isSecretOlder(newSecret, oldSecret *v1.Secret) bool {
	if newSecret == nil || oldSecret == nil {
		return false
	}
	newVersion, _ := storageetcd.Versioner.ObjectResourceVersion(newSecret)
	oldVersion, _ := storageetcd.Versioner.ObjectResourceVersion(oldSecret)
	return newVersion < oldVersion
}

// Add establishes a new reference to a secret in the cache.
// If the secret is not present, it initializes a new tracking entry.
// Note: This does not trigger an immediate fetch; retrieval is deferred until Get().
func (s *secretStore) Add(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}

	s.lock.Lock()
	defer s.lock.Unlock()
	item, exists := s.items[key]
	if !exists {
		item = &secretStoreItem{
			refCount: 0,
			secret:   &secretData{},
		}
		s.items[key] = item
	}

	item.refCount++
	// Invalidate the cached data to ensure a fresh fetch on the next Get call.
	item.secret = nil
}

// Delete removes a reference to a secret from the cache.
// If the reference count reaches zero, the item is evicted from the cache.
func (s *secretStore) Delete(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}

	s.lock.Lock()
	defer s.lock.Unlock()
	if item, ok := s.items[key]; ok {
		item.refCount--
		if item.refCount == 0 {
			delete(s.items, key)
		}
	}
}

// GetObjectTTLFromNodeFunc returns a GetObjectTTLFunc that derives the TTL
// from a specific annotation on the Node resource.
func GetObjectTTLFromNodeFunc(getNode func() (*v1.Node, error)) GetObjectTTLFunc {
	return func() (time.Duration, bool) {
		node, err := getNode()
		if err != nil {
			return time.Duration(0), false
		}
		if node != nil && node.Annotations != nil {
			if value, ok := node.Annotations[v1.ObjectTTLAnnotationKey]; ok {
				if intValue, err := strconv.Atoi(value); err == nil {
					return time.Duration(intValue) * time.Second, true
				}
			}
		}
		return time.Duration(0), false
	}
}

// isSecretFresh determines if the cached secret data is still within its TTL.
func (s *secretStore) isSecretFresh(data *secretData) bool {
	secretTTL := s.defaultTTL
	if ttl, ok := s.getTTL(); ok {
		secretTTL = ttl
	}
	return s.clock.Now().Before(data.lastUpdateTime.Add(secretTTL))
}

// Get retrieves a secret from the cache, performing a refresh from the API server
// if the cached entry is missing, invalidated, or expired.
func (s *secretStore) Get(namespace, name string) (*v1.Secret, error) {
	key := objectKey{namespace: namespace, name: name}

	// Phase 1: Identify or initialize the cached data entry.
	data := func() *secretData {
		s.lock.Lock()
		defer s.lock.Unlock()
		item, exists := s.items[key]
		if !exists {
			return nil
		}
		if item.secret == nil {
			item.secret = &secretData{}
		}
		return item.secret
	}()
	if data == nil {
		return nil, fmt.Errorf("secret %q/%q not registered", namespace, name)
	}

	// Phase 2: Ensure the data is fresh under lock.
	data.Lock()
	defer data.Unlock()
	if data.err != nil || !s.isSecretFresh(data) {
		opts := metav1.GetOptions{}
		if data.secret != nil && data.err == nil {
			// Optimize: use apiserver cache for periodic background refreshes.
			util.FromApiserverCache(&opts)
		}
		secret, err := s.kubeClient.CoreV1().Secrets(namespace).Get(name, opts)
		
		// Strategic fallback: if we failed to fetch but have no cached data, return error.
		if err != nil && !apierrors.IsNotFound(err) && data.secret == nil && data.err == nil {
			return secret, err
		}
		
		// Update cache if:
		// 1. Fetch succeeded and is at least as new as what we have.
		// 2. Fetch returned a 'Not Found' error (marking the secret as deleted).
		if (err == nil && !isSecretOlder(secret, data.secret)) || apierrors.IsNotFound(err) {
			data.secret = secret
			data.err = err
			data.lastUpdateTime = s.clock.Now()
		}
	}
	return data.secret, data.err
}

// NewCachingSecretManager initializes a Manager that optimizes Secret access through
// a reference-counted local cache.
//
// Lifecycle Logic:
// 1. Pod Registration: Invalidates any existing cache entries for the pod's secrets.
// 2. Retrieval (GetSecret):
//    - Hits the local cache if the data is present and fresh.
//    - Triggers a background refresh from the API server if data is stale.
//    - Falls back to the API server if data is missing or invalidated.
func NewCachingSecretManager(kubeClient clientset.Interface, getTTL GetObjectTTLFunc) Manager {
	secretStore := newSecretStore(kubeClient, clock.RealClock{}, getTTL, defaultTTL)
	return newCacheBasedSecretManager(secretStore)
}
