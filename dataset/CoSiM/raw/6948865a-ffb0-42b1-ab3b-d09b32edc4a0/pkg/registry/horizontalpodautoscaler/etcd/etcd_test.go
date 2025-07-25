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

package etcd

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/expapi/v1"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"

	"github.com/coreos/go-etcd/etcd"
)

var scheme *runtime.Scheme
var codec runtime.Codec

func init() {
	// Ensure that expapi/v1 packege is used, so that it will get initialized and register HorizontalPodAutoscaler object.
	dummy := v1.HorizontalPodAutoscaler{}
	dummy.Spec = v1.HorizontalPodAutoscalerSpec{}
}

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, testapi.Codec(), etcdtest.PathPrefix())
	storage := NewREST(etcdStorage)
	return storage, fakeEtcdClient, etcdStorage
}

func validNewHorizontalPodAutoscaler(name string) *expapi.HorizontalPodAutoscaler {
	return &expapi.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: expapi.HorizontalPodAutoscalerSpec{
			ScaleRef: &expapi.SubresourceReference{
				Subresource: "scale",
			},
			MinCount: 1,
			MaxCount: 5,
			Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	autoscaler := validNewHorizontalPodAutoscaler("foo")
	autoscaler.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		autoscaler,
		// invalid
		&expapi.HorizontalPodAutoscaler{},
	)
}

func TestUpdate(t *testing.T) {
	storage, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	key, err := storage.KeyFunc(test.TestContext(), "foo")
	if err != nil {
		t.Fatal(err)
	}
	key = etcdtest.AddPrefix(key)
	fakeEtcdClient.ExpectNotFoundGet(key)
	fakeEtcdClient.ChangeIndex = 2
	autoscaler := validNewHorizontalPodAutoscaler("foo")
	existing := validNewHorizontalPodAutoscaler("exists")
	existing.Namespace = test.TestNamespace()
	obj, err := storage.Create(test.TestContext(), existing)
	if err != nil {
		t.Fatalf("unable to create object: %v", err)
	}
	older := obj.(*expapi.HorizontalPodAutoscaler)
	older.ResourceVersion = "1"
	test.TestUpdate(
		autoscaler,
		existing,
		older,
	)
}

func TestDelete(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	autoscaler := validNewHorizontalPodAutoscaler("foo2")
	key, _ := storage.KeyFunc(ctx, "foo2")
	key = etcdtest.AddPrefix(key)
	createFn := func() runtime.Object {
		fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         runtime.EncodeOrDie(testapi.Codec(), autoscaler),
					ModifiedIndex: 1,
				},
			},
		}
		return autoscaler
	}
	gracefulSetFn := func() bool {
		if fakeEtcdClient.Data[key].R.Node == nil {
			return false
		}
		return fakeEtcdClient.Data[key].R.Node.TTL == 30
	}
	test.TestDelete(createFn, gracefulSetFn)
}

func TestGet(t *testing.T) {
	storage, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	autoscaler := validNewHorizontalPodAutoscaler("foo")
	test.TestGet(autoscaler)
}

func TestList(t *testing.T) {
	storage, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	key := etcdtest.AddPrefix(storage.KeyRootFunc(test.TestContext()))
	autoscaler := validNewHorizontalPodAutoscaler("foo")
	test.TestList(
		autoscaler,
		func(objects []runtime.Object) []runtime.Object {
			return registrytest.SetObjectsForKey(fakeEtcdClient, key, objects)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeEtcdClient, resourceVersion)
		})
}
