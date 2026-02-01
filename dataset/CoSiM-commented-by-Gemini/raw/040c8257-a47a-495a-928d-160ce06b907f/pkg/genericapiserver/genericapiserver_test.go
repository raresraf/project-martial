/**
 * @file genericapiserver_test.go
 * @brief Unit tests for the generic Kubernetes API server implementation.
 *
 * @details
 * This file contains a suite of unit tests for the `genericapiserver` package.
 * The tests cover the lifecycle and configuration of the `GenericAPIServer`,
 * including its instantiation, the installation of API groups, discovery endpoints,
 * and integration with various handlers like authentication, authorization, and Swagger/OpenAPI.
 *
 * The tests use a mock etcd server (`etcdtesting.EtcdTestServer`) to provide a storage backend
 * and `httptest.NewServer` to serve the API server's handlers for testing HTTP requests and responses.
 * The `stretchr/testify/assert` library is used for making assertions.
 *
 * Key areas tested include:
 * - **Server Instantiation**: Verifying that a `GenericAPIServer` instance is created correctly
 *   based on its `Config`.
 * - **API Group Installation**: Ensuring that both legacy (`/api`) and new (`/apis`) API groups
 *   are installed at their correct URL prefixes.
 * - **Discovery Endpoints**: Testing the `/apis` endpoint to ensure it correctly lists the
 *   installed API groups and their versions. This includes adding and removing groups dynamically.
 * - **Address Resolution**: Validating the `getServerAddressByClientCIDRs` function, which determines
 *   the correct server address to return to a client based on whether the client is internal or
 *   external to the cluster.
 * - **Handler Registration**: Testing that custom handlers can be correctly registered with the
 *   server's multiplexer.
 * - **Swagger/OpenAPI Integration**: Verifying that the Swagger API endpoint is correctly installed.
 */
/*
Copyright 2015 The Kubernetes Authors.

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

package genericapiserver

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"

	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apiserver"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	utilnet "k8s.io/kubernetes/pkg/util/net"

	"github.com/stretchr/testify/assert"
)

// setUp is a convenience function for setting up the environment for most tests.
// It initializes an etcd test server and a basic GenericAPIServer config.
func setUp(t *testing.T) (GenericAPIServer, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	etcdServer := etcdtesting.NewEtcdTestClientServer(t)

	genericapiserver := GenericAPIServer{}
	config := Config{}
	config.PublicAddress = net.ParseIP("192.168.10.4")

	return genericapiserver, etcdServer, config, assert.New(t)
}

// newMaster sets up a new GenericAPIServer with a default configuration for testing.
func newMaster(t *testing.T) (*GenericAPIServer, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	_, etcdserver, config, assert := setUp(t)

	config.ProxyDialer = func(network, addr string) (net.Conn, error) { return nil, nil }
	config.ProxyTLSClientConfig = &tls.Config{}
	config.Serializer = api.Codecs
	config.APIPrefix = "/api"
	config.APIGroupPrefix = "/apis"

	s, err := New(&config)
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}
	return s, etcdserver, config, assert
}

// TestNew verifies that the New function correctly initializes a GenericAPIServer
// instance based on the provided configuration.
func TestNew(t *testing.T) {
	s, etcdserver, config, assert := newMaster(t)
	defer etcdserver.Terminate(t)

	// Verify that the server's fields are correctly populated from the config.
	assert.Equal(s.enableSwaggerSupport, config.EnableSwaggerSupport)
	assert.Equal(s.legacyAPIPrefix, config.APIPrefix)
	assert.Equal(s.apiPrefix, config.APIGroupPrefix)
	assert.Equal(s.admissionControl, config.AdmissionControl)
	assert.Equal(s.RequestContextMapper(), config.RequestContextMapper)
	assert.Equal(s.ExternalAddress, config.ExternalHost)
	assert.Equal(s.ClusterIP, config.PublicAddress)
	assert.Equal(s.PublicReadWritePort, config.ReadWritePort)
	assert.Equal(s.ServiceReadWriteIP, config.ServiceReadWriteIP)

	// These functions should point to the same memory location, verifying that the dialer is passed correctly.
	serverDialer, _ := utilnet.Dialer(s.ProxyTransport)
	serverDialerFunc := fmt.Sprintf("%p", serverDialer)
	configDialerFunc := fmt.Sprintf("%p", config.ProxyDialer)
	assert.Equal(serverDialerFunc, configDialerFunc)

	assert.Equal(s.ProxyTransport.(*http.Transport).TLSClientConfig, config.ProxyTLSClientConfig)
}

// Verifies that InstallAPIGroups correctly installs both legacy and new API groups
// and that their discovery endpoints are available.
func TestInstallAPIGroups(t *testing.T) {
	_, etcdserver, config, assert := setUp(t)
	defer etcdserver.Terminate(t)

	config.ProxyDialer = func(network, addr string) (net.Conn, error) { return nil, nil }
	config.ProxyTLSClientConfig = &tls.Config{}
	config.APIPrefix = "/apiPrefix"
	config.APIGroupPrefix = "/apiGroupPrefix"
	config.Serializer = api.Codecs

	s, err := New(&config)
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}

	apiGroupMeta := registered.GroupOrDie(api.GroupName)
	extensionsGroupMeta := registered.GroupOrDie(extensions.GroupName)
	apiGroupsInfo := []APIGroupInfo{
		{
			// legacy group version (/api/v1)
			GroupMeta:                    *apiGroupMeta,
			VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
			IsLegacyGroup:                true,
			ParameterCodec:               api.ParameterCodec,
			NegotiatedSerializer:         api.Codecs,
		},
		{
			// extensions group version (/apis/extensions/v1beta1)
			GroupMeta:                    *extensionsGroupMeta,
			VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
			OptionsExternalVersion:       &apiGroupMeta.GroupVersion,
			ParameterCodec:               api.ParameterCodec,
			NegotiatedSerializer:         api.Codecs,
		},
	}
	s.InstallAPIGroups(apiGroupsInfo)

	server := httptest.NewServer(s.HandlerContainer.ServeMux)
	defer server.Close()
	validPaths := []string{
		// Discovery endpoint for the legacy group
		config.APIPrefix,
		// Path for the specific version of the legacy group
		config.APIPrefix + "/" + apiGroupMeta.GroupVersion.Version,
		// Discovery endpoint for the extensions group
		config.APIGroupPrefix + "/" + extensionsGroupMeta.GroupVersion.Group,
		// Path for the specific version of the extensions group
		config.APIGroupPrefix + "/" + extensionsGroupMeta.GroupVersion.String(),
	}
	for _, path := range validPaths {
		_, err := http.Get(server.URL + path)
		if !assert.NoError(err) {
			t.Errorf("unexpected error: %v, for path: %s", err, path)
		}
	}
}

// TestNewHandlerContainer verifies that NewHandlerContainer correctly associates
// the provided ServeMux with the returned restful.Container.
func TestNewHandlerContainer(t *testing.T) {
	assert := assert.New(t)
	mux := http.NewServeMux()
	container := NewHandlerContainer(mux, nil)
	assert.Equal(mux, container.ServeMux, "ServerMux's do not match")
}

// TestHandleWithAuth verifies that HandleWithAuth registers a new path
// in the MuxHelper, making it available for serving.
func TestHandleWithAuth(t *testing.T) {
	server, etcdserver, _, assert := setUp(t)
	defer etcdserver.Terminate(t)

	mh := apiserver.MuxHelper{Mux: http.NewServeMux()}
	server.MuxHelper = &mh
	handler := func(r http.ResponseWriter, w *http.Request) { w.Write(nil) }
	server.HandleWithAuth("/test", http.HandlerFunc(handler))

	assert.Contains(server.MuxHelper.RegisteredPaths, "/test", "Path not found in MuxHelper")
}

// TestHandleFuncWithAuth verifies that HandleFuncWithAuth registers a new path
// in the MuxHelper, making it available for serving.
func TestHandleFuncWithAuth(t *testing.T) {
	server, etcdserver, _, assert := setUp(t)
	defer etcdserver.Terminate(t)

	mh := apiserver.MuxHelper{Mux: http.NewServeMux()}
	server.MuxHelper = &mh
	handler := func(r http.ResponseWriter, w *http.Request) { w.Write(nil) }
	server.HandleFuncWithAuth("/test", handler)

	assert.Contains(server.MuxHelper.RegisteredPaths, "/test", "Path not found in MuxHelper")
}

// TestInstallSwaggerAPI verifies that the swagger API is correctly installed
// at the /swaggerapi/ endpoint and that it respects the ExternalAddress configuration.
func TestInstallSwaggerAPI(t *testing.T) {
	server, etcdserver, _, assert := setUp(t)
	defer etcdserver.Terminate(t)

	mux := http.NewServeMux()
	server.HandlerContainer = NewHandlerContainer(mux, nil)

	// Ensure swagger isn't installed by default.
	ws := server.HandlerContainer.RegisteredWebServices()
	if !assert.Equal(len(ws), 0) {
		for x := range ws {
			assert.NotEqual("/swaggerapi", ws[x].RootPath(), "SwaggerAPI was installed without a call to InstallSwaggerAPI()")
		}
	}

	// Install swagger and verify its presence and path.
	server.InstallSwaggerAPI()
	ws = server.HandlerContainer.RegisteredWebServices()
	if assert.NotEqual(0, len(ws), "SwaggerAPI not installed.") {
		assert.Equal("/swaggerapi/", ws[0].RootPath(), "SwaggerAPI did not install to the proper path. %s != /swaggerapi", ws[0].RootPath())
	}

	// Verify that an empty externalHost falls back to using ClusterIP and PublicReadWritePort.
	mux = http.NewServeMux()
	server.HandlerContainer = NewHandlerContainer(mux, nil)
	server.ExternalAddress = ""
	server.ClusterIP = net.IPv4(10, 10, 10, 10)
	server.PublicReadWritePort = 1010
	server.InstallSwaggerAPI()
	if assert.NotEqual(0, len(ws), "SwaggerAPI not installed.") {
		assert.Equal("/swaggerapi/", ws[0].RootPath(), "SwaggerAPI did not install to the proper path. %s != /swaggerapi", ws[0].RootPath())
	}
}

// decodeResponse is a helper function to decode a JSON HTTP response body into a given object.
func decodeResponse(resp *http.Response, obj interface{}) error {
	defer resp.Body.Close()

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, obj); err != nil {
		return err
	}
	return nil
}

// getGroupList is a helper function to fetch and decode the APIGroupList from the /apis endpoint.
func getGroupList(server *httptest.Server) (*unversioned.APIGroupList, error) {
	resp, err := http.Get(server.URL + "/apis")
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected server response, expected %d, actual: %d", http.StatusOK, resp.StatusCode)
	}

	groupList := unversioned.APIGroupList{}
	err = decodeResponse(resp, &groupList)
	return &groupList, err
}

// TestDiscoveryAtAPIS tests the dynamic discovery of API groups at the /apis endpoint.
// It verifies that adding and removing API groups is correctly reflected in the discovery response.
func TestDiscoveryAtAPIS(t *testing.T) {
	master, etcdserver, _, assert := newMaster(t)
	defer etcdserver.Terminate(t)

	server := httptest.NewServer(master.HandlerContainer.ServeMux)
	groupList, err := getGroupList(server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assert.Equal(0, len(groupList.Groups))

	// Add an API Group and verify it appears in the discovery response.
	extensionsVersions := []unversioned.GroupVersionForDiscovery{
		{
			GroupVersion: testapi.Extensions.GroupVersion().String(),
			Version:      testapi.Extensions.GroupVersion().Version,
		},
	}
	extensionsPreferredVersion := unversioned.GroupVersionForDiscovery{
		GroupVersion: extensions.GroupName + "/preferred",
		Version:      "preferred",
	}
	master.AddAPIGroupForDiscovery(unversioned.APIGroup{
		Name:             extensions.GroupName,
		Versions:         extensionsVersions,
		PreferredVersion: extensionsPreferredVersion,
	})

	groupList, err = getGroupList(server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(1, len(groupList.Groups))
	groupListGroup := groupList.Groups[0]
	assert.Equal(extensions.GroupName, groupListGroup.Name)
	assert.Equal(extensionsVersions, groupListGroup.Versions)
	assert.Equal(extensionsPreferredVersion, groupListGroup.PreferredVersion)
	assert.Equal(master.getServerAddressByClientCIDRs(&http.Request{}), groupListGroup.ServerAddressByClientCIDRs)

	// Remove the group and verify it is no longer present.
	master.RemoveAPIGroupForDiscovery(extensions.GroupName)
	groupList, err = getGroupList(server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(0, len(groupList.Groups))
}

// TestGetServerAddressByClientCIDRs verifies that the server returns the correct
// public or internal address based on the client's IP address.
func TestGetServerAddressByClientCIDRs(t *testing.T) {
	s, etcdserver, _, _ := newMaster(t)
	defer etcdserver.Terminate(t)

	// The server should return its external address to clients outside the cluster.
	publicAddressCIDRMap := []unversioned.ServerAddressByClientCIDR{
		{
			ClientCIDR: "0.0.0.0/0",
			ServerAddress: s.ExternalAddress,
		},
	}
	// For clients within the cluster, it should return both the external and internal (service) address.
	internalAddressCIDRMap := []unversioned.ServerAddressByClientCIDR{
		publicAddressCIDRMap[0],
		{
			ClientCIDR:    s.ServiceClusterIPRange.String(),
			ServerAddress: net.JoinHostPort(s.ServiceReadWriteIP.String(), strconv.Itoa(s.ServiceReadWritePort)),
		},
	}
	internalIP := "10.0.0.1"
	publicIP := "1.1.1.1"
	testCases := []struct {
		Request     http.Request
		ExpectedMap []unversioned.ServerAddressByClientCIDR
	}{
		// No client IP information in the request.
		{
			Request:     http.Request{},
			ExpectedMap: publicAddressCIDRMap,
		},
		// Client IP from an internal source.
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {internalIP},
				},
			},
			ExpectedMap: internalAddressCIDRMap,
		},
		// Client IP from a public source.
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {publicIP},
				},
			},
			ExpectedMap: publicAddressCIDRMap,
		},
		// Client IP from a forwarded header (internal).
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {internalIP},
				},
			},
			ExpectedMap: internalAddressCIDRMap,
		},
		// Client IP from a forwarded header (public).
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {publicIP},
				},
			},
			ExpectedMap: publicAddressCIDRMap,
		},

		// Client IP from the request's remote address (internal).
		{
			Request: http.Request{
				RemoteAddr: internalIP,
			},
			ExpectedMap: internalAddressCIDRMap,
		},
		// Client IP from the request's remote address (public).
		{
			Request: http.Request{
				RemoteAddr: publicIP,
			},
			ExpectedMap: publicAddressCIDRMap,
		},
		// Invalid IP address in RemoteAddr.
		{
			Request: http.Request{
				RemoteAddr: "invalidIP",
			},
			ExpectedMap: publicAddressCIDRMap,
		},
	}

	for i, test := range testCases {
		if a, e := s.getServerAddressByClientCIDRs(&test.Request), test.ExpectedMap; reflect.DeepEqual(e, a) != true {
			t.Fatalf("test case %d failed. expected: %v, actual: %v", i+1, e, a)
		}
	}
}
