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

package registrytest

import (
	"testing"

	"github.com/coreos/go-etcd/etcd"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

func NewEtcdStorage(t *testing.T) (storage.Interface, *tools.FakeEtcdClient) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeClient, testapi.Codec(), etcdtest.PathPrefix())
	return etcdStorage, fakeClient
}

type keyFunc func(api.Context, string) (string, error)
type newFunc func() runtime.Object

func GetObject(fakeClient *tools.FakeEtcdClient, keyFn keyFunc, newFn newFunc, ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	meta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return nil, err
	}
	key, err := keyFn(ctx, meta.Name)
	if err != nil {
		return nil, err
	}
	key = etcdtest.AddPrefix(key)
	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		return nil, err
	}
	result := newFn()
	if err := testapi.Codec().DecodeInto([]byte(resp.Node.Value), result); err != nil {
		return nil, err
	}
	return result, nil
}

func SetObject(fakeClient *tools.FakeEtcdClient, keyFn keyFunc, ctx api.Context, obj runtime.Object) error {
	meta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return err
	}
	key, err := keyFn(ctx, meta.Name)
	if err != nil {
		return err
	}
	key = etcdtest.AddPrefix(key)
	_, err = fakeClient.Set(key, runtime.EncodeOrDie(testapi.Codec(), obj), 0)
	return err
}

func SetObjectsForKey(fakeClient *tools.FakeEtcdClient, key string, objects []runtime.Object) []runtime.Object {
	result := make([]runtime.Object, len(objects))
	if len(objects) > 0 {
		nodes := make([]*etcd.Node, len(objects))
		for i, obj := range objects {
			encoded := runtime.EncodeOrDie(testapi.Codec(), obj)
			decoded, _ := testapi.Codec().Decode([]byte(encoded))
			nodes[i] = &etcd.Node{Value: encoded}
			result[i] = decoded
		}
		fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Nodes: nodes,
				},
			},
			E: nil,
		}
	} else {
		fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{},
			E: fakeClient.NewError(tools.EtcdErrorCodeNotFound),
		}
	}
	return result
}

func SetResourceVersion(fakeClient *tools.FakeEtcdClient, resourceVersion uint64) {
	fakeClient.ChangeIndex = resourceVersion
}
