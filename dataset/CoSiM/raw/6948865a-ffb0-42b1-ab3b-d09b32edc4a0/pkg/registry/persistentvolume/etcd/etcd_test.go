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
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, latest.Codec, etcdtest.PathPrefix())
	storage, statusStorage := NewStorage(etcdStorage)
	return storage, statusStorage, fakeEtcdClient, etcdStorage
}

func validNewPersistentVolume(name string) *api.PersistentVolume {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{Path: "/foo"},
			},
			PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRetain,
		},
		Status: api.PersistentVolumeStatus{
			Phase:   api.VolumePending,
			Message: "bar",
			Reason:  "foo",
		},
	}
	return pv
}

func validChangedPersistentVolume() *api.PersistentVolume {
	pv := validNewPersistentVolume("foo")
	pv.ResourceVersion = "1"
	return pv
}

func TestCreate(t *testing.T) {
	storage, _, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError).ClusterScope()
	pv := validNewPersistentVolume("foo")
	pv.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		pv,
		// invalid
		&api.PersistentVolume{
			ObjectMeta: api.ObjectMeta{Name: "*BadName!"},
		},
	)
}

func TestDelete(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError).ClusterScope()

	pv := validChangedPersistentVolume()
	key, _ := storage.KeyFunc(ctx, pv.Name)
	key = etcdtest.AddPrefix(key)
	createFn := func() runtime.Object {
		fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         runtime.EncodeOrDie(latest.Codec, pv),
					ModifiedIndex: 1,
				},
			},
		}
		return pv
	}
	gracefulSetFn := func() bool {
		if fakeEtcdClient.Data[key].R.Node == nil {
			return false
		}
		return fakeEtcdClient.Data[key].R.Node.TTL == 30
	}
	test.TestDelete(createFn, gracefulSetFn)
}

func TestEtcdGetPersistentVolumes(t *testing.T) {
	storage, _, fakeClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError).ClusterScope()
	persistentVolume := validNewPersistentVolume("foo")
	test.TestGet(persistentVolume)
}

func TestEtcdListPersistentVolumes(t *testing.T) {
	storage, _, fakeClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError).ClusterScope()
	key := etcdtest.AddPrefix(storage.KeyRootFunc(test.TestContext()))
	persistentVolume := validNewPersistentVolume("foo")
	test.TestList(
		persistentVolume,
		func(objects []runtime.Object) []runtime.Object {
			return registrytest.SetObjectsForKey(fakeClient, key, objects)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeClient, resourceVersion)
		})
}

func TestPersistentVolumesDecode(t *testing.T) {
	storage, _, _, _ := newStorage(t)
	expected := validNewPersistentVolume("foo")
	body, err := latest.Codec.Encode(expected)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := storage.New()
	if err := latest.Codec.DecodeInto(body, actual); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("mismatch: %s", util.ObjectDiff(expected, actual))
	}
}

func TestEtcdUpdatePersistentVolumes(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeClient, _ := newStorage(t)
	persistentVolume := validChangedPersistentVolume()

	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, validNewPersistentVolume("foo")), 0)

	_, _, err := storage.Update(ctx, persistentVolume)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var persistentVolumeOut api.PersistentVolume
	err = latest.Codec.DecodeInto([]byte(response.Node.Value), &persistentVolumeOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	persistentVolume.ObjectMeta.ResourceVersion = persistentVolumeOut.ObjectMeta.ResourceVersion
	if !api.Semantic.DeepEqual(persistentVolume, &persistentVolumeOut) {
		t.Errorf("Unexpected persistentVolume: %#v, expected %#v", &persistentVolumeOut, persistentVolume)
	}
}

func TestDeletePersistentVolumes(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeClient, _ := newStorage(t)
	persistentVolume := validNewPersistentVolume("foo")
	name := persistentVolume.Name
	key, _ := storage.KeyFunc(ctx, name)
	key = etcdtest.AddPrefix(key)
	fakeClient.ChangeIndex = 1
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(latest.Codec, persistentVolume),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
	}
	_, err := storage.Delete(ctx, name, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEtcdUpdateStatus(t *testing.T) {
	storage, statusStorage, fakeClient, etcdStorage := newStorage(t)
	ctx := api.NewContext()
	fakeClient.TestIndex = true

	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	pvStart := validNewPersistentVolume("foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, pvStart), 1)

	pvIn := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
		},
		Status: api.PersistentVolumeStatus{
			Phase: api.VolumeBound,
		},
	}

	expected := *pvStart
	expected.ResourceVersion = "2"
	expected.Labels = pvIn.Labels
	expected.Status = pvIn.Status

	_, _, err := statusStorage.Update(ctx, pvIn)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var pvOut api.PersistentVolume
	key, _ = storage.KeyFunc(ctx, "foo")
	if err := etcdStorage.Get(key, &pvOut, false); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !api.Semantic.DeepEqual(expected, pvOut) {
		t.Errorf("unexpected object: %s", util.ObjectDiff(expected, pvOut))
	}
}
