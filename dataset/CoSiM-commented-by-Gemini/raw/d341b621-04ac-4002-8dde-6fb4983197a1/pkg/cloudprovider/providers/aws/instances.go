/**
 * @file instances.go
 * @brief Manages EC2 instance identification and caching for the Kubernetes AWS cloud provider.
 *
 * @details This file provides utilities for handling EC2 instance identifiers and implements
 * a caching layer for EC2 `DescribeInstances` API calls. This is a critical component
 * for performance and stability, as it reduces the number of API calls to AWS, helping
 * to avoid rate limiting and reduce latency.
 *
 * Key Functionalities:
 * 1.  **ID Parsing**: It defines methods to parse the Kubernetes `Node.Spec.ProviderID`
 *     format (e.g., `aws:///us-west-2a/i-12345...`) and extract the canonical
 *     EC2 instance ID (e.g., `i-12345...`).
 * 2.  **Instance Caching**: It implements an `instanceCache` that stores the results of
 *     `DescribeInstances` calls. Subsequent requests for instance data can be served
 *     from this cache if they meet specific criteria (e.g., maximum age), significantly
 *     improving the performance of operations that need to query instance metadata.
 *
 * Production Systems (Go/TypeScript):
 * This caching mechanism is vital in a large-scale Kubernetes cluster on AWS. Many core
 * Kubernetes components (like the controller manager and scheduler) frequently need to
 * look up instance details (e.g., zone, instance type, IP addresses). Without a cache,
 * these frequent lookups would quickly lead to AWS API throttling, causing cluster
 * operations to fail or be severely delayed.
 */
/*
Copyright 2017 The Kubernetes Authors.

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

package aws

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"regexp"
	"sync"
	"time"
)

// awsInstanceRegMatch is a regex for sanity-checking AWS instance ID format.
var awsInstanceRegMatch = regexp.MustCompile("^i-[^/]*$")

// awsInstanceID represents the canonical ID of an EC2 instance in the AWS API (e.g. "i-12345678").
type awsInstanceID string

// awsString converts the awsInstanceID to a string pointer for the AWS SDK.
func (i awsInstanceID) awsString() *string {
	return aws.String(string(i))
}

// kubernetesInstanceID represents the ProviderID for an instance in the Kubernetes API.
// It can appear in several forms:
//  * aws:///<zone>/<awsInstanceId>
//  * aws:////<awsInstanceId>
//  * <awsInstanceId>
type kubernetesInstanceID string

// mapToAWSInstanceID extracts the canonical awsInstanceID from the kubernetesInstanceID.
func (name kubernetesInstanceID) mapToAWSInstanceID() (awsInstanceID, error) {
	s := string(name)

	if !strings.HasPrefix(s, "aws://") {
		// Assume a bare AWS instance ID. Normalize it to a URL format.
		s = "aws://" + "/" + "/" + s
	}
	url, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("Invalid instance name (%s): %v", name, err)
	}
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS instance (%s)", name)
	}

	awsID := ""
	// The path can be /<az>/<instance-id> or just /<instance-id>
	tokens := strings.Split(strings.Trim(url.Path, "/"), "/")
	if len(tokens) == 1 {
		awsID = tokens[0]
	} else if len(tokens) == 2 {
		awsID = tokens[1]
	}

	// Sanity check the extracted ID.
	if awsID == "" || !awsInstanceRegMatch.MatchString(awsID) {
		return "", fmt.Errorf("Invalid format for AWS instance (%s)", name)
	}

	return awsInstanceID(awsID), nil
}

// mapToAWSInstanceIDs extracts awsInstanceIDs from a slice of Nodes, returning an error
// if any Node cannot be mapped.
func mapToAWSInstanceIDs(nodes []*v1.Node) ([]awsInstanceID, error) {
	var instanceIDs []awsInstanceID
	for _, node := range nodes {
		if node.Spec.ProviderID == "" {
			return nil, fmt.Errorf("node %q did not have ProviderID set", node.Name)
		}
		instanceID, err := kubernetesInstanceID(node.Spec.ProviderID).mapToAWSInstanceID()
		if err != nil {
			return nil, fmt.Errorf("unable to parse ProviderID %q for node %q", node.Spec.ProviderID, node.Name)
		}
		instanceIDs = append(instanceIDs, instanceID)
	}

	return instanceIDs, nil
}

// mapToAWSInstanceIDsTolerant extracts awsInstanceIDs from a slice of Nodes,
// skipping and logging a warning for any Node that cannot be mapped.
func mapToAWSInstanceIDsTolerant(nodes []*v1.Node) []awsInstanceID {
	var instanceIDs []awsInstanceID
	for _, node := range nodes {
		if node.Spec.ProviderID == "" {
			glog.Warningf("node %q did not have ProviderID set", node.Name)
			continue
		}
		instanceID, err := kubernetesInstanceID(node.Spec.ProviderID).mapToAWSInstanceID()
		if err != nil {
			glog.Warningf("unable to parse ProviderID %q for node %q", node.Spec.ProviderID, node.Name)
			continue
		}
		instanceIDs = append(instanceIDs, instanceID)
	}

	return instanceIDs
}

// describeInstance gets the full information about a single EC2 instance from the AWS API.
func describeInstance(ec2Client EC2, instanceID awsInstanceID) (*ec2.Instance, error) {
	request := &ec2.DescribeInstancesInput{
		InstanceIds: []*string{instanceID.awsString()},
	}

	instances, err := ec2Client.DescribeInstances(request)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances found for instance: %s", instanceID)
	}
	if len(instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for instance: %s", instanceID)
	}
	return instances[0], nil
}

// instanceCache manages a thread-safe, time-based cache of EC2 instance descriptions.
type instanceCache struct {
	cloud    *Cloud
	mutex    sync.Mutex
	snapshot *allInstancesSnapshot
}

// describeAllInstancesUncached performs a live call to the AWS EC2 API to describe all instances
// matching a set of filters, and updates the cache with the result.
func (c *instanceCache) describeAllInstancesUncached() (*allInstancesSnapshot, error) {
	now := time.Now()
	glog.V(4).Infof("EC2 DescribeInstances - fetching all instances")
	filters := []*ec2.Filter{}
	instances, err := c.cloud.describeInstances(filters)
	if err != nil {
		return nil, err
	}

	m := make(map[awsInstanceID]*ec2.Instance)
	for _, i := range instances {
		id := awsInstanceID(aws.StringValue(i.InstanceId))
		m[id] = i
	}

	snapshot := &allInstancesSnapshot{now, m}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.snapshot != nil && snapshot.olderThan(c.snapshot) {
		glog.Infof("Not caching concurrent AWS DescribeInstances results")
	} else {
		c.snapshot = snapshot
	}

	return snapshot, nil
}

// cacheCriteria specifies the conditions a cached snapshot must meet to be considered valid.
type cacheCriteria struct {
	// MaxAge is the maximum acceptable age of a cached snapshot.
	MaxAge time.Duration
	// HasInstances is a list of instance IDs that must be present in the snapshot.
	HasInstances []awsInstanceID
}

// describeAllInstancesCached returns all instances, using cached results if they meet
// the specified criteria, otherwise performing a fresh API call.
func (c *instanceCache) describeAllInstancesCached(criteria cacheCriteria) (*allInstancesSnapshot, error) {
	var err error
	snapshot := c.getSnapshot()
	// Invalidate cache if it's too old or doesn't contain required instances.
	if snapshot != nil && !snapshot.MeetsCriteria(criteria) {
		snapshot = nil
	}

	// Block Logic: If the cache was invalid (or empty), perform a live API call.
	if snapshot == nil {
		snapshot, err = c.describeAllInstancesUncached()
		if err != nil {
			return nil, err
		}
	} else {
		glog.V(6).Infof("EC2 DescribeInstances - using cached results")
	}

	return snapshot, nil
}

// getSnapshot safely retrieves the current snapshot from the cache.
func (c *instanceCache) getSnapshot() *allInstancesSnapshot {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.snapshot
}

// olderThan is a helper for comparing snapshot timestamps.
func (s *allInstancesSnapshot) olderThan(other *allInstancesSnapshot) bool {
	return other.timestamp.After(s.timestamp)
}

// MeetsCriteria checks if the snapshot satisfies the given cache criteria.
func (s *allInstancesSnapshot) MeetsCriteria(criteria cacheCriteria) bool {
	// Pre-condition: Check if the snapshot is within the maximum age limit.
	if criteria.MaxAge > 0 {
		now := time.Now()
		if now.Sub(s.timestamp) > criteria.MaxAge {
			glog.V(6).Infof("instanceCache snapshot cannot be used as is older than MaxAge=%s", criteria.MaxAge)
			return false
		}
	}

	// Pre-condition: Check if the snapshot contains all the required instances.
	if len(criteria.HasInstances) != 0 {
		for _, id := range criteria.HasInstances {
			if nil == s.instances[id] {
				glog.V(6).Infof("instanceCache snapshot cannot be used as does not contain instance %s", id)
				return false
			}
		}
	}

	return true
}

// allInstancesSnapshot holds the results from a single `DescribeInstances` API call,
// along with a timestamp for cache invalidation.
type allInstancesSnapshot struct {
	timestamp time.Time
	instances map[awsInstanceID]*ec2.Instance
}

// FindInstances returns the instances from the snapshot that match the specified IDs.
func (s *allInstancesSnapshot) FindInstances(ids []awsInstanceID) map[awsInstanceID]*ec2.Instance {
	m := make(map[awsInstanceID]*ec2.Instance)
	for _, id := range ids {
		instance := s.instances[id]
		if instance != nil {
			m[id] = instance
		}
	}
	return m
}
