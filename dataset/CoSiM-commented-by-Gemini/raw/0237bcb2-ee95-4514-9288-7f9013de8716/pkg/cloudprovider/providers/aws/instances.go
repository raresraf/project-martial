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

// Package aws provides utilities for interacting with AWS EC2 instances within the Kubernetes cloud provider.
// It includes logic for mapping instance IDs between Kubernetes and AWS formats, caching instance descriptions,
// and retrieving instance metadata from the EC2 API.
package aws

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"sync"
	"time"
)

// awsInstanceID represents the ID of the instance in the AWS API, e.g. i-12345678
// The "traditional" format is "i-12345678"
// A new longer format is also being introduced: "i-12345678abcdef01"
// We should not assume anything about the length or format, though it seems
// reasonable to assume that instances will continue to start with "i-".
type awsInstanceID string

func (i awsInstanceID) awsString() *string {
	// Functional Utility: Converts the custom `awsInstanceID` type to a pointer to a string,
	// suitable for use with AWS SDK functions.
	return aws.String(string(i))
}

// kubernetesInstanceID represents the id for an instance in the kubernetes API;
// the following form
//  * aws:///<zone>/<awsInstanceId>
//  * aws:////<awsInstanceId>
//  * <awsInstanceId>
type kubernetesInstanceID string

// mapToAWSInstanceID extracts the awsInstanceID from the kubernetesInstanceID.
// It parses different formats of Kubernetes ProviderID to derive the canonical AWS instance ID.
// Returns an error if the format is invalid or cannot be parsed.
func (name kubernetesInstanceID) mapToAWSInstanceID() (awsInstanceID, error) {
	s := string(name)

	// Block Logic: Handles ProviderIDs that do not start with "aws://".
	// Assumes a bare AWS instance ID and constructs a compliant URI.
	if !strings.HasPrefix(s, "aws://") {
		// Assume a bare aws volume id (vol-1234...)
		// Build a URL with an empty host (AZ)
		s = "aws://" + "/" + "/" + s
	}
	// Functional Utility: Parses the URI string into a URL structure.
	url, err := url.Parse(s)
	// Block Logic: Error handling for invalid URL parsing.
	if err != nil {
		return "", fmt.Errorf("Invalid instance name (%s): %v", name, err)
	}
	// Block Logic: Validates that the scheme of the parsed URL is "aws".
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS instance (%s)", name)
	}

	awsID := ""
	// Functional Utility: Splits the URL path to extract the instance ID and optionally the availability zone.
	tokens := strings.Split(strings.Trim(url.Path, "/"), "/")
	// Block Logic: Differentiates between formats with and without an availability zone.
	if len(tokens) == 1 {
		// instanceId
		awsID = tokens[0]
	} else if len(tokens) == 2 {
		// az/instanceId
		awsID = tokens[1]
	}

	// We sanity check the resulting volume; the two known formats are
	// i-12345678 and i-12345678abcdef01
	// TODO: Regex match?
	// Block Logic: Performs a basic sanity check on the extracted AWS instance ID.
	// It verifies that the ID is not empty, does not contain slashes, and starts with "i-".
	if awsID == "" || strings.Contains(awsID, "/") || !strings.HasPrefix(awsID, "i-") {
		return "", fmt.Errorf("Invalid format for AWS instance (%s)", name)
	}

	return awsInstanceID(awsID), nil
}

// mapToAWSInstanceIDs extracts the awsInstanceIDs from the provided Kubernetes Nodes.
// It returns a slice of `awsInstanceID` and an error if any node's ProviderID is
// missing or cannot be mapped to a valid AWS instance ID.
func mapToAWSInstanceIDs(nodes []*v1.Node) ([]awsInstanceID, error) {
	var instanceIDs []awsInstanceID
	// Block Logic: Iterates through the list of Kubernetes nodes.
	for _, node := range nodes {
		// Block Logic: Checks if the ProviderID is set for the current node.
		// Returns an error if ProviderID is missing.
		if node.Spec.ProviderID == "" {
			return nil, fmt.Errorf("node %q did not have ProviderID set", node.Name)
		}
		// Functional Utility: Maps the Kubernetes ProviderID to an AWS instance ID.
		instanceID, err := kubernetesInstanceID(node.Spec.ProviderID).mapToAWSInstanceID()
		// Block Logic: Error handling for failed ProviderID parsing.
		if err != nil {
			return nil, fmt.Errorf("unable to parse ProviderID %q for node %q", node.Spec.ProviderID, node.Name)
		}
		// Functional Utility: Appends the successfully mapped AWS instance ID to the result slice.
		instanceIDs = append(instanceIDs, instanceID)
	}

	return instanceIDs, nil
}

// mapToAWSInstanceIDsTolerant extracts the awsInstanceIDs from the provided Kubernetes Nodes.
// Unlike `mapToAWSInstanceIDs`, this function tolerates errors and skips nodes
// that have missing or invalid ProviderIDs, logging a warning instead of returning an error.
func mapToAWSInstanceIDsTolerant(nodes []*v1.Node) []awsInstanceID {
	var instanceIDs []awsInstanceID
	// Block Logic: Iterates through the list of Kubernetes nodes.
	for _, node := range nodes {
		// Block Logic: Checks if the ProviderID is set. Logs a warning and skips if missing.
		if node.Spec.ProviderID == "" {
			glog.Warningf("node %q did not have ProviderID set", node.Name)
			continue
		}
		// Functional Utility: Maps the Kubernetes ProviderID to an AWS instance ID.
		instanceID, err := kubernetesInstanceID(node.Spec.ProviderID).mapToAWSInstanceID()
		// Block Logic: Error handling for failed ProviderID parsing. Logs a warning and skips if invalid.
		if err != nil {
			glog.Warningf("unable to parse ProviderID %q for node %q", node.Spec.ProviderID, node.Name)
			continue
		}
		// Functional Utility: Appends the successfully mapped AWS instance ID to the result slice.
		instanceIDs = append(instanceIDs, instanceID)
	}

	return instanceIDs
}

// describeInstance retrieves the full information about a single instance from the EC2 API.
// It takes an EC2 client and an `awsInstanceID` and returns a pointer to an `ec2.Instance` object.
// Returns an error if the instance is not found, multiple instances are found, or an API error occurs.
func describeInstance(ec2Client EC2, instanceID awsInstanceID) (*ec2.Instance, error) {
	// Functional Utility: Constructs a DescribeInstancesInput request with the specified instance ID.
	request := &ec2.DescribeInstancesInput{
		InstanceIds: []*string{instanceID.awsString()},
	}

	// Functional Utility: Calls the EC2 DescribeInstances API.
	instances, err := ec2Client.DescribeInstances(request)
	// Block Logic: Error handling for API call failures.
	if err != nil {
		return nil, err
	}
	// Block Logic: Validates that exactly one instance was returned for the given ID.
	// Returns an error if no instances or multiple instances are found.
	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances found for instance: %s", instanceID)
	}
	// Block Logic: Validates that exactly one instance was returned.
	if len(instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for instance: %s", instanceID)
	}
	return instances[0], nil
}

// instanceCache manages the cache of EC2 DescribeInstances API calls.
// It aims to reduce the number of direct API calls to AWS by storing
// a snapshot of all instances for a certain period.
type instanceCache struct {
	// cloud is a reference to the main Cloud provider object.
	cloud *Cloud

	// mutex protects access to the `snapshot` field to ensure thread safety.
	mutex    sync.Mutex
	// snapshot stores the last retrieved `allInstancesSnapshot`.
	snapshot *allInstancesSnapshot
}

// describeAllInstancesUncached retrieves the full information about all instances from the EC2 API without using the cache.
// It constructs a DescribeInstances request with filters and processes the response.
// The retrieved instances are then stored in a new `allInstancesSnapshot`.
func (c *instanceCache) describeAllInstancesUncached() (*allInstancesSnapshot, error) {
	now := time.Now()

	glog.V(4).Infof("EC2 DescribeInstances - fetching all instances")

	// Functional Utility: Defines filters for the DescribeInstances API call (currently empty, fetching all).
	filters := []*ec2.Filter{}
	// Functional Utility: Calls the underlying cloud provider's describeInstances method.
	instances, err := c.cloud.describeInstances(filters)
	// Block Logic: Error handling for API call failures.
	if err != nil {
		return nil, err
	}

	// Functional Utility: Transforms the list of EC2 instances into a map for easier lookup by `awsInstanceID`.
	m := make(map[awsInstanceID]*ec2.Instance)
	// Block Logic: Populates the map with `awsInstanceID` as keys and `ec2.Instance` pointers as values.
	for _, i := range instances {
		id := awsInstanceID(aws.StringValue(i.InstanceId))
		m[id] = i
	}

	// Functional Utility: Creates a new snapshot with the current timestamp and the fetched instances.
	snapshot := &allInstancesSnapshot{now, m}

	// Block Logic: Acquires a lock to safely update the cache.
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Block Logic: Implements a simple mechanism to prevent older concurrent results from overwriting newer ones.
	if c.snapshot != nil && snapshot.olderThan(c.snapshot) {
		// If this happens a lot, we could run this function in a mutex and only return one result
		glog.Infof("Not caching concurrent AWS DescribeInstances results")
	} else {
		// Functional Utility: Updates the cache with the new, fresher snapshot.
		c.snapshot = snapshot
	}

	return snapshot, nil
}

// cacheCriteria holds criteria that must hold for a cached snapshot to be considered valid and used.
type cacheCriteria struct {
	// MaxAge indicates the maximum age of a cached snapshot we can accept.
	// If set to 0 (i.e. unset), cached values will not time out because of age.
	MaxAge time.Duration

	// HasInstances is a list of awsInstanceIDs that must be present in a cached snapshot
	// for it to be considered valid. If any specified instance is missing, the snapshot
	// will be deemed invalid and a fresh fetch will be performed.
	HasInstances []awsInstanceID
}

// describeAllInstancesCached returns all instances, utilizing cached results if they meet the specified criteria.
// If the cache is stale or does not meet criteria, it performs an uncached API call.
func (c *instanceCache) describeAllInstancesCached(criteria cacheCriteria) (*allInstancesSnapshot, error) {
	var err error
	// Functional Utility: Retrieves the current snapshot from the cache.
	snapshot := c.getSnapshot()
	// Block Logic: Checks if the retrieved snapshot is valid according to the provided `cacheCriteria`.
	if snapshot != nil && !snapshot.MeetsCriteria(criteria) {
		snapshot = nil
	}

	// Block Logic: If no valid snapshot is found, perform an uncached API call.
	if snapshot == nil {
		snapshot, err = c.describeAllInstancesUncached()
		// Block Logic: Error handling for uncached API call failures.
		if err != nil {
			return nil, err
		}
	} else {
		glog.V(6).Infof("EC2 DescribeInstances - using cached results")
	}

	return snapshot, nil
}

// getSnapshot returns the current `allInstancesSnapshot` from the cache in a thread-safe manner.
func (c *instanceCache) getSnapshot() *allInstancesSnapshot {
	// Block Logic: Acquires a lock to safely read the snapshot.
	c.mutex.Lock()
	defer c.mutex.Unlock()

	return c.snapshot
}

// olderThan is a helper method for `allInstancesSnapshot` to determine if the current snapshot
// is older than another provided snapshot.
// Functional Utility: Compares the timestamps of two snapshots.
func (s *allInstancesSnapshot) olderThan(other *allInstancesSnapshot) bool {
	// After() is technically broken by time changes until we have monotonic time
	return other.timestamp.After(s.timestamp)
}

// MeetsCriteria evaluates if the current `allInstancesSnapshot` satisfies the given `cacheCriteria`.
// It checks both the age of the snapshot and the presence of required instance IDs.
func (s *allInstancesSnapshot) MeetsCriteria(criteria cacheCriteria) bool {
	// Block Logic: Checks if a maximum age is specified and if the snapshot exceeds that age.
	if criteria.MaxAge > 0 {
		// Sub() is technically broken by time changes until we have monotonic time
		now := time.Now()
		// Invariant: If the snapshot is older than `MaxAge`, it's considered stale.
		if now.Sub(s.timestamp) > criteria.MaxAge {
			glog.V(6).Infof("instanceCache snapshot cannot be used as is older than MaxAge=%s", criteria.MaxAge)
			return false
		}
	}

	// Block Logic: Checks if specific instance IDs are required and if they are all present in the snapshot.
	if len(criteria.HasInstances) != 0 {
		// Invariant: All required instance IDs must be found in the snapshot.
		for _, id := range criteria.HasInstances {
			if nil == s.instances[id] {
				glog.V(6).Infof("instanceCache snapshot cannot be used as does not contain instance %s", id)
				return false
			}
		}
	}

	return true
}

// allInstancesSnapshot holds the results from querying for all instances from the EC2 API,
// along with the timestamp indicating when the data was fetched. This structure is used
// for caching purposes.
type allInstancesSnapshot struct {
	// timestamp records when this snapshot of instance data was created.
	timestamp time.Time
	// instances is a map storing EC2 instance data, keyed by `awsInstanceID`.
	instances map[awsInstanceID]*ec2.Instance
}

// FindInstances returns a new map containing only the `ec2.Instance` objects
// corresponding to the specified `awsInstanceID`s. If an ID is not found in the snapshot,
// it is gracefully ignored.
func (s *allInstancesSnapshot) FindInstances(ids []awsInstanceID) map[awsInstanceID]*ec2.Instance {
	// Functional Utility: Initializes an empty map to store the found instances.
	m := make(map[awsInstanceID]*ec2.Instance)
	// Block Logic: Iterates through the provided list of instance IDs.
	for _, id := range ids {
		// Functional Utility: Looks up the instance in the snapshot's instance map.
		instance := s.instances[id]
		// Block Logic: If the instance is found, adds it to the result map.
		if instance != nil {
			m[id] = instance
		}
	}
	return m
}

