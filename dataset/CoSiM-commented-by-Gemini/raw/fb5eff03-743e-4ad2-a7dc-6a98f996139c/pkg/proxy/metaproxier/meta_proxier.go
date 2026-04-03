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

// Package metaproxier provides a meta-proxy that wraps multiple proxy.Provider
// instances and dispatches calls to the appropriate one based on IP family.
// This is a key component for enabling dual-stack (IPv4/IPv6) networking in
// kube-proxy, allowing a single logical proxier to manage rules for both
// IP families.
package metaproxier

import (
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
)

// metaProxier is an implementation of proxy.Provider that wraps two single-stack
// (one IPv4, one IPv6) proxiers and dispatches calls to them. It acts as a
// unified dual-stack proxy provider.
type metaProxier struct {
	// ipv4Proxier is the actual proxy.Provider for handling IPv4 traffic and rules.
	ipv4Proxier proxy.Provider
	// ipv6Proxier is the actual proxy.Provider for handling IPv6 traffic and rules.
	ipv6Proxier proxy.Provider
}

// NewMetaProxier returns a dual-stack "meta-proxier" that implements the
// proxy.Provider interface. API calls are dispatched to the underlying
// single-stack ProxyProvider instances based on the address family of the
// objects, or broadcast to both for non-family-specific objects.
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

// Sync immediately synchronizes the state of both underlying proxiers to
// their respective proxy rules in the data plane (e.g., iptables, IPVS).
func (proxier *metaProxier) Sync() {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.Sync()
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.Sync()
	}
}

// SyncLoop runs the periodic work of both underlying proxiers. This is expected
// to run as a goroutine or as the main loop of the application. It does not return.
// The IPv6 proxier's loop is started in a new goroutine, while the IPv4 loop
// runs in the current goroutine, blocking indefinitely.
func (proxier *metaProxier) SyncLoop() {
	if proxier.ipv4Proxier == nil {
		proxier.ipv6Proxier.SyncLoop() // never returns
	} else if proxier.ipv6Proxier == nil {
		proxier.ipv4Proxier.SyncLoop() // never returns
	} else {
		go proxier.ipv6Proxier.SyncLoop() // Use go-routine here!
		proxier.ipv4Proxier.SyncLoop()    // never returns
	}
}

// OnServiceAdd forwards the Service addition event to both underlying proxiers,
// as a Service can have implications for both IP families (e.g. ClusterIPs).
func (proxier *metaProxier) OnServiceAdd(service *v1.Service) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceAdd(service)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceAdd(service)
	}
}

// OnServiceUpdate forwards the Service update event to both underlying proxiers.
func (proxier *metaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceUpdate(oldService, service)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceUpdate(oldService, service)
	}
}

// OnServiceDelete forwards the Service deletion event to both underlying proxiers.
func (proxier *metaProxier) OnServiceDelete(service *v1.Service) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceDelete(service)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceDelete(service)
	}
}

// OnServiceSynced forwards the Service synced event to both underlying proxiers.
// This indicates that the initial list of services has been processed.
func (proxier *metaProxier) OnServiceSynced() {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceSynced()
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceSynced()
	}
}

// OnEndpointSliceAdd dispatches the EndpointSlice addition event to the
// appropriate proxier based on the slice's address type (IPv4 or IPv6).
func (proxier *metaProxier) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	// This is the core dispatch logic for dual-stack.
	// The event is sent only to the proxier matching the endpoint's IP family.
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		if proxier.ipv4Proxier != nil {
			proxier.ipv4Proxier.OnEndpointSliceAdd(endpointSlice)
		}
	case discovery.AddressTypeIPv6:
		if proxier.ipv6Proxier != nil {
			proxier.ipv6Proxier.OnEndpointSliceAdd(endpointSlice)
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSliceUpdate dispatches the EndpointSlice update event to the
// appropriate proxier based on the slice's address type.
func (proxier *metaProxier) OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice *discovery.EndpointSlice) {
	// The address type of an EndpointSlice is immutable, so we only need to check the new slice.
	switch newEndpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		if proxier.ipv4Proxier != nil {
			proxier.ipv4Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
		}
	case discovery.AddressTypeIPv6:
		if proxier.ipv6Proxier != nil {
			proxier.ipv6Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", newEndpointSlice.AddressType)
	}
}

// OnEndpointSliceDelete dispatches the EndpointSlice deletion event to the
// appropriate proxier based on the slice's address type.
func (proxier *metaProxier) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		if proxier.ipv4Proxier != nil {
			proxier.ipv4Proxier.OnEndpointSliceDelete(endpointSlice)
		}
	case discovery.AddressTypeIPv6:
		if proxier.ipv6Proxier != nil {
			proxier.ipv6Proxier.OnEndpointSliceDelete(endpointSlice)
		}
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

// OnEndpointSlicesSynced forwards the EndpointSlices synced event to both proxiers.
func (proxier *metaProxier) OnEndpointSlicesSynced() {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnEndpointSlicesSynced()
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnEndpointSlicesSynced()
	}
}

// OnNodeAdd forwards the Node addition event to both underlying proxiers.
func (proxier *metaProxier) OnNodeAdd(node *v1.Node) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnNodeAdd(node)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnNodeAdd(node)
	}
}

// OnNodeUpdate forwards the Node update event to both underlying proxiers.
func (proxier *metaProxier) OnNodeUpdate(oldNode, node *v1.Node) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnNodeUpdate(oldNode, node)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnNodeUpdate(oldNode, node)
	}
}

// OnNodeDelete forwards the Node deletion event to both underlying proxiers.
func (proxier *metaProxier) OnNodeDelete(node *v1.Node) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnNodeDelete(node)
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnNodeDelete(node)
	}
}

// OnNodeSynced forwards the Node synced event to both underlying proxiers.
func (proxier *metaProxier) OnNodeSynced() {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnNodeSynced()
	}
	if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnNodeSynced()
	}
}

// OnServiceCIDRsChanged is called whenever a change is observed
// in any of the ServiceCIDRs, and provides complete list of service cidrs.
func (proxier *metaProxier) OnServiceCIDRsChanged(cidrs []string) {
	if proxier.ipv4Proxier != nil {
		proxier.ipv4Proxier.OnServiceCIDRsChanged(cidrs)
	}
if proxier.ipv6Proxier != nil {
		proxier.ipv6Proxier.OnServiceCIDRsChanged(cidrs)
	}
}
