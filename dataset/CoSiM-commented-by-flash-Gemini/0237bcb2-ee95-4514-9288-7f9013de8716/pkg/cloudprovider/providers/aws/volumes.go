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

// Package aws provides functionalities for interacting with AWS EBS volumes within the Kubernetes cloud provider context.
// This includes parsing various volume ID formats used by Kubernetes and mapping them to standard AWS volume IDs.
// Architectural Intent: To abstract AWS-specific EBS volume management details, offering a consistent interface
// for Kubernetes components that need to provision, attach, or detach storage volumes in an AWS environment.
// It handles the different ways Kubernetes might refer to an AWS volume, ensuring compatibility and correct identification.
package aws

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
)

// awsVolumeID represents the ID of the volume in the AWS API, e.g. vol-12345678.
// The "traditional" format is "vol-12345678".
// A new longer format is also being introduced: "vol-12345678abcdef01".
// This type provides a strong type for AWS volume IDs, enhancing type safety and readability.
type awsVolumeID string

// awsString converts the awsVolumeID type to a pointer to a string,
// which is the format expected by many AWS SDK functions.
func (i awsVolumeID) awsString() *string {
	return aws.String(string(i))
}

// KubernetesVolumeID represents the ID for a volume in the Kubernetes API.
// It can appear in a few recognized forms:
//  * aws://<zone>/<awsVolumeId> (full URI with availability zone)
//  * aws:///<awsVolumeId> (full URI without availability zone)
//  * <awsVolumeId> (bare AWS volume ID)
// This type encapsulates these varied formats for consistent processing.
type KubernetesVolumeID string

// mapToAWSVolumeID extracts the bare awsVolumeID from a KubernetesVolumeID string.
// It parses different URI formats that Kubernetes might use to refer to an AWS volume.
// @return The extracted awsVolumeID or an error if the format is invalid.
func (name KubernetesVolumeID) mapToAWSVolumeID() (awsVolumeID, error) {
	// name looks like aws://availability-zone/awsVolumeId

	// The original idea of the URL-style name was to put the AZ into the
	// host, so we could find the AZ immediately from the name without
	// querying the API.  But it turns out we don't actually need it for
	// multi-AZ clusters, as we put the AZ into the labels on the PV instead.
	// However, if in future we want to support multi-AZ cluster
	// volume-awareness without using PersistentVolumes, we likely will
	// want the AZ in the host.

	s := string(name)

	// Conditional Logic: If the string does not have the "aws://" prefix, assume it's a bare AWS ID.
	// Invariant: Ensures consistent URI parsing by prefixing with "aws://" if absent.
	if !strings.HasPrefix(s, "aws://") {
		// Assume a bare aws volume id (vol-1234...)
		// Block Logic: Build a URL with an empty host (AZ) to standardize parsing.
		s = "aws://" + "" + "/" + s
	}
	// Functional Utility: Parse the string as a URL to extract components.
	url, err := url.Parse(s)
	// Conditional Logic: Handle URL parsing errors.
	if err != nil {
		// TODO: Maybe we should pass a URL into the Volume functions
		return "", fmt.Errorf("Invalid disk name (%s): %v", name, err)
	}
	// Conditional Logic: Validate the URL scheme.
	// Precondition: The scheme must be "aws" for a valid Kubernetes AWS volume ID.
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS volume (%s)", name)
	}

	awsID := url.Path
	// Functional Utility: Trim leading/trailing slashes from the extracted path to get the bare ID.
	awsID = strings.Trim(awsID, "/")

	// Block Logic: Sanity check the extracted AWS volume ID.
	// Precondition: The ID should not contain "/" and must start with "vol-".
	// This helps validate against malformed or unexpected IDs.
	// TODO: Regex match?
	if strings.Contains(awsID, "/") || !strings.HasPrefix(awsID, "vol-") {
		return "", fmt.Errorf("Invalid format for AWS volume (%s)", name)
	}

	return awsVolumeID(awsID), nil
}
