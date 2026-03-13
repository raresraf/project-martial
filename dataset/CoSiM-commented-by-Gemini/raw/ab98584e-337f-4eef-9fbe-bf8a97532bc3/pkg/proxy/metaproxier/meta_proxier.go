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

// package metaproxier provides a meta-proxier for dual-stack (IPv4/IPv6) support.
// It wraps two separate single-stack proxy providers and dispatches API calls
// to the appropriate one based on the address family. This allows Kubernetes
// to manage network proxy rules for both IPv4 and IPv6 environments simultaneously.
package metaproxier

import (
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/config"
)

// metaProxier is a proxy.Provider implementation that wraps two single-stack
// proxy providers (one for IPv4, one for IPv6). It multiplexes calls to the
// underlying providers.
type metaProxier struct {
	// ipv4Proxier is the proxy.Provider that handles IPv4 traffic.
	ipv4Proxier proxy.Provider
	// ipv6Proxier is the proxy.Provider that handles IPv6 traffic.
	ipv6Proxier proxy.Provider
	// NoopNodeHandler is embedded to satisfy the proxy.Provider interface
	// for node handling methods, as the meta-proxier itself does not
	// implement custom node handling logic.
	config.NoopNodeHandler
}

// NewMetaProxier returns a dual-stack "meta-proxier". It takes two
// single-stack proxy providers and returns a new proxy.Provider that
// dispatches calls to them.
func NewMetaProxier(ipv4Proxier, ipv6Proxier proxy.Provider) proxy.Provider {
	// Return nil if both underlying proxiers are nil.
	if ipv4Proxier == nil && ipv6Proxier == nil {
		return nil
	}
	return proxy.Provider(&metaProxier{
		ipv4Proxier: ipv4Proxier,
		ipv6Proxier: ipv6Proxier,
	})
}

// Sync delegates the call to both underlying proxiers to synchronize
// their state to proxy rules.
func (proxier *metaProxier) Sync() {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.Sync()
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.Sync()
	}
}

// SyncLoop starts the synchronization loops for both underlying proxiers.
// It is a blocking call that runs forever.
func (proxier *metaProxier) SyncLoop() {
	// The IPv4 and IPv6 proxiers are assumed to be non-nil.
	// The IPv6 proxier's SyncLoop is started in a new goroutine.
	go proxier.ipv6Proxier.SyncLoop()
	// The IPv4 proxier's SyncLoop is called directly and will block,
	// effectively running both loops concurrently.
	proxier.ipv4Proxier.SyncLoop()
}

// OnServiceAdd forwards the service addition event to both underlying proxiers.
func (proxier *metaProxier) OnServiceAdd(service *v1.Service) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceAdd(service)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceAdd(service)
	}
}

// OnServiceUpdate forwards the service update event to both underlying proxiers.
func (proxier *metaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceUpdate(oldService, service)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceUpdate(oldService, service)
	}
}

// OnServiceDelete forwards the service deletion event to both underlying proxiers.
func (proxier *metaProxier) OnServiceDelete(service *v1.Service) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceDelete(service)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceDelete(service)
	}
}

// OnServiceSynced forwards the service synced event to both underlying proxiers.
func (proxier *metaProxier) OnServiceSynced() {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceSynced()
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceSynced()
	}
}

// OnEndpointSliceAdd dispatches the endpoint slice addition event to the
// appropriate proxier based on the slice's address type.
func (proxier *metaProxier) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	// Pre-condition: Check which IP family the EndpointSlice belongs to.
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		// Logic: Dispatch IPv4-specific data to the IPv4 proxier.
		if proxier.ipv4Proxier != nil {
			proxier.ipv4Proxier.OnEndpointSliceAdd(endpointSlice)
		}
	case discovery.AddressTypeIPv6:
		// Logic: Dispatch IPv6-specific data to the IPv6 proxier.
		if proxier.ipv6Proxier != nil {
			proxier.ipv6Proxier.OnEndpointSliceAdd(endpointSlice)
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSliceUpdate dispatches the endpoint slice update event to the
// appropriate proxier based on the slice's address type.
func (proxier *metaProxier) OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice *discovery.EndpointSlice) {
	// Pre-condition: Check which IP family the EndpointSlice belongs to.
	switch newEndpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		// Logic: Dispatch IPv4-specific data to the IPv4 proxier.
		if proxier.ipv4Proxier != nil {
			proxier.ipv4Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
		}
	case discovery.AddressTypeIPv6:
		// Logic: Dispatch IPv6-specific data to the IPv6 proxier.
		if proxier.ipv6Proxier != nil {
			proxier.ipv6Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", newEndpointSlice.AddressType)
	}
}

// OnEndpointSliceDelete dispatches the endpoint slice deletion event to the
// appropriate proxier based on the slice's address type.
func (proxier *metaProxier) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	// Pre-condition: Check which IP family the EndpointSlice belongs to.
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		// Logic: Dispatch IPv4-specific data to the IPv4 proxier.
		if proxier.ipv4Proxier != nil {
			proxier.ipv4Proxier.OnEndpointSliceDelete(endpointSlice)
		}
	case discovery.AddressTypeIPv6:
		// Logic: Dispatch IPv6-specific data to the IPv6 proxier.
		if proxier.ipv6Proxier != nil {
			proxier.ipv6Proxier.OnEndpointSliceDelete(endpointSlice)
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSlicesSynced forwards the endpoint slices synced event to both
// underlying proxiers.
func (proxier *metaProxier) OnEndpointSlicesSynced() {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnEndpointSlicesSynced()
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnEndpointSlicesSynced()
	}
}

// OnNodeAdd forwards the node addition event to both underlying proxiers.
func (proxier *metaProxier) OnNodeAdd(node *v1.Node) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnNodeAdd(node)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnNodeAdd(node)
	}
}

// OnNodeUpdate forwards the node update event to both underlying proxiers.
func (proxier *metaProxier) OnNodeUpdate(oldNode, node *v1.Node) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnNodeUpdate(oldNode, node)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnNodeUpdate(oldNode, node)
	}
}

// OnNodeDelete forwards the node deletion event to both underlying proxiers.
func (proxier *metaProxier) OnNodeDelete(node *v1.Node) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnNodeDelete(node)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnNodeDelete(node)
	}
}

// OnNodeSynced forwards the node synced event to both underlying proxiers.
func (proxier *metaProxier) OnNodeSynced() {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnNodeSynced()
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnNodeSynced()
	}
}

// OnServiceCIDRsChanged forwards the service CIDRs changed event to both
// underlying proxier.
func (proxier *metaProxier) OnServiceCIDRsChanged(cidrs []string) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceCIDRsChanged(cidrs)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceCIDRsChanged(cidrs)
	}
}
