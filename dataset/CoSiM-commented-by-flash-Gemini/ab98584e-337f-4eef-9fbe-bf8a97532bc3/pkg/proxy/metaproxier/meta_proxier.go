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

// Package metaproxier implements a dual-stack Kubernetes proxy provider that
// dispatches API calls to separate IPv4 and IPv6 proxy implementations.
// It acts as a facade, allowing a single proxy instance to manage network
// rules for both address families.
package metaproxier

import (
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/config"
)

// metaProxier is a dual-stack proxy that wraps separate IPv4 and IPv6
// proxy providers. It dispatches API events (Service, EndpointSlice, Node)
// to the appropriate underlying proxier based on the address family.
type metaProxier struct {
	// ipv4Proxier is the actual proxy provider for IPv4 traffic.
	ipv4Proxier proxy.Provider
	// ipv6Proxier is the actual proxy provider for IPv6 traffic.
	ipv6Proxier proxy.Provider
	// NoopNodeHandler provides a default, no-operation implementation for node events,
	// as node handling is not yet implemented for the meta proxier.
	config.NoopNodeHandler
}

// NewMetaProxier returns a new dual-stack "meta-proxier".
// Proxier API calls will be dispatched to the provided ProxyProvider instances
// based on their address family.
//
// Args:
//   ipv4Proxier: The ProxyProvider instance responsible for IPv4 proxying.
//   ipv6Proxier: The ProxyProvider instance responsible for IPv6 proxying.
//
// Returns:
//   A proxy.Provider interface implemented by the metaProxier.
func NewMetaProxier(ipv4Proxier, ipv6Proxier proxy.Provider) proxy.Provider {
	return proxy.Provider(&metaProxier{
		ipv4Proxier: ipv4Proxier,
		ipv6Proxier: ipv6Proxier,
	})
}

// Sync immediately synchronizes the ProxyProvider's current state to
// proxy rules for both IPv4 and IPv6.
func (proxier *metaProxier) Sync() {
	proxier.ipv4Proxier.Sync() // Synchronize IPv4 proxy rules.
	proxier.ipv6Proxier.Sync() // Synchronize IPv6 proxy rules.
}

// SyncLoop runs periodic work for both IPv4 and IPv6 proxiers.
// This method is expected to run as a goroutine or as the main loop of the app.
// It will not return; the IPv6 SyncLoop runs in a separate goroutine,
// while the IPv4 SyncLoop blocks the current goroutine.
func (proxier *metaProxier) SyncLoop() {
	// Start the IPv6 proxy's SyncLoop in a new goroutine to allow concurrent execution.
	go proxier.ipv6Proxier.SyncLoop()
	// Run the IPv4 proxy's SyncLoop, which is typically a blocking call.
	proxier.ipv4Proxier.SyncLoop()
}

// OnServiceAdd is called whenever creation of a new service object is observed.
// It dispatches the event to both the IPv4 and IPv6 proxy providers.
func (proxier *metaProxier) OnServiceAdd(service *v1.Service) {
	proxier.ipv4Proxier.OnServiceAdd(service) // Notify IPv4 proxier of new service.
	proxier.ipv6Proxier.OnServiceAdd(service) // Notify IPv6 proxier of new service.
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed. It dispatches the event to both the
// IPv4 and IPv6 proxy providers.
func (proxier *metaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	proxier.ipv4Proxier.OnServiceUpdate(oldService, service) // Notify IPv4 proxier of service update.
	proxier.ipv6Proxier.OnServiceUpdate(oldService, service) // Notify IPv6 proxier of service update.
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed. It dispatches the event to both the IPv4 and IPv6
// proxy providers.
func (proxier *metaProxier) OnServiceDelete(service *v1.Service) {
	proxier.ipv4Proxier.OnServiceDelete(service) // Notify IPv4 proxier of service deletion.
	proxier.ipv6Proxier.OnServiceDelete(service) // Notify IPv6 proxier of service deletion.
}

// OnServiceSynced is called once all the initial service event handlers were
// called and the state is fully propagated to local cache.
// It dispatches the event to both the IPv4 and IPv6 proxy providers.
func (proxier *metaProxier) OnServiceSynced() {
	proxier.ipv4Proxier.OnServiceSynced() // Notify IPv4 proxier that services are synced.
	proxier.ipv6Proxier.OnServiceSynced() // Notify IPv6 proxier that services are synced.
}

// OnEndpointSliceAdd is called whenever creation of a new endpoint slice object
// is observed. It dispatches the event to the appropriate proxy provider
// based on the EndpointSlice's AddressType (IPv4 or IPv6).
func (proxier *metaProxier) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceAdd(endpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceAdd(endpointSlice)
	default:
		// Log an error if an unsupported EndpointSlice address type is encountered.
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSliceUpdate is called whenever modification of an existing endpoint
// slice object is observed. It dispatches the event to the appropriate proxy
// provider based on the new EndpointSlice's AddressType.
func (proxier *metaProxier) OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice *discovery.EndpointSlice) {
	switch newEndpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	default:
		// Log an error if an unsupported EndpointSlice address type is encountered.
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", newEndpointSlice.AddressType)
	}
}

// OnEndpointSliceDelete is called whenever deletion of an existing endpoint slice
// object is observed. It dispatches the event to the appropriate proxy provider
// based on the EndpointSlice's AddressType.
func (proxier *metaProxier) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceDelete(endpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceDelete(endpointSlice)
	default:
		// Log an error if an unsupported EndpointSlice address type is encountered.
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSlicesSynced is called once all the initial endpoint slice event handlers were
// called and the state is fully propagated to local cache.
// It dispatches the event to both the IPv4 and IPv6 proxy providers.
func (proxier *metaProxier) OnEndpointSlicesSynced() {
	proxier.ipv4Proxier.OnEndpointSlicesSynced() // Notify IPv4 proxier that endpoint slices are synced.
	proxier.ipv6Proxier.OnEndpointSlicesSynced() // Notify IPv6 proxier that endpoint slices are synced.
}

// OnNodeAdd is called whenever creation of new node object is observed.
// It dispatches the event to both the IPv4 and IPv6 proxy providers.
func (proxier *metaProxier) OnNodeAdd(node *v1.Node) {
	proxier.ipv4Proxier.OnNodeAdd(node) // Notify IPv4 proxier of new node.
	proxier.ipv6Proxier.OnNodeAdd(node) // Notify IPv6 proxier of new node.
}

// OnNodeUpdate is called whenever modification of an existing
// node object is observed. It dispatches the event to both the
// IPv4 and IPv6 proxy providers.
func (proxier *metaProxier) OnNodeUpdate(oldNode, node *v1.Node) {
	proxier.ipv4Proxier.OnNodeUpdate(oldNode, node) // Notify IPv4 proxier of node update.
	proxier.ipv6Proxier.OnNodeUpdate(oldNode, node) // Notify IPv6 proxier of node update.
}

// OnNodeDelete is called whenever deletion of an existing node
// object is observed. It dispatches the event to both the IPv4 and IPv6
// proxy providers.
func (proxier *metaProxier) OnNodeDelete(node *v1.Node) {
	proxier.ipv4Proxier.OnNodeDelete(node) // Notify IPv4 proxier of node deletion.
	proxier.ipv6Proxier.OnNodeDelete(node) // Notify IPv6 proxier of node deletion.
}

// OnNodeSynced is called once all the initial node event handlers were
// called and the state is fully propagated to local cache.
// It dispatches the event to both the IPv4 and IPv6 proxy providers.
func (proxier *metaProxier) OnNodeSynced() {
	proxier.ipv4Proxier.OnNodeSynced() // Notify IPv4 proxier that nodes are synced.
	proxier.ipv6Proxier.OnNodeSynced() // Notify IPv6 proxier that nodes are synced.
}

// OnServiceCIDRsChanged is called whenever a change is observed
// in any of the ServiceCIDRs, and provides a complete list of service CIDRs.
// It dispatches the event to both the IPv4 and IPv6 proxy providers.
func (proxier *metaProxier) OnServiceCIDRsChanged(cidrs []string) {
	proxier.ipv4Proxier.OnServiceCIDRsChanged(cidrs) // Notify IPv4 proxier of Service CIDR changes.
	proxier.ipv6Proxier.OnServiceCIDRsChanged(cidrs) // Notify IPv6 proxier of Service CIDR changes.
}
