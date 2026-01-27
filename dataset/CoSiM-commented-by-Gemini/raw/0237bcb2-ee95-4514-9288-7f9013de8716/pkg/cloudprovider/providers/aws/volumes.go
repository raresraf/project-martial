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

// Package aws provides utilities for interacting with AWS Elastic Block Store (EBS) volumes
// within the Kubernetes cloud provider. It includes logic for mapping volume IDs between
// Kubernetes and AWS formats.
package aws

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
)

// awsVolumeID represents the ID of the volume in the AWS API, e.g. vol-12345678
// The "traditional" format is "vol-12345678"
// A new longer format is also being introduced: "vol-12345678abcdef01"
// We should not assume anything about the length or format, though it seems
// reasonable to assume that volumes will continue to start with "vol-".
type awsVolumeID string

func (i awsVolumeID) awsString() *string {
	// Functional Utility: Converts the custom `awsVolumeID` type to a pointer to a string,
	// suitable for use with AWS SDK functions.
	return aws.String(string(i))
}

// KubernetesVolumeID represents the ID for a volume in the Kubernetes API.
// Several formats are recognized for this ID:
//   - `aws://<zone>/<awsVolumeId>`: Full URI with availability zone.
//   - `aws:///<awsVolumeId>`: URI without a specified availability zone.
//   - `<awsVolumeId>`: Bare AWS volume ID.
type KubernetesVolumeID string

// mapToAWSVolumeID extracts the canonical `awsVolumeID` from a `KubernetesVolumeID`.
// It parses different URI formats to derive the underlying AWS volume identifier.
// Returns an error if the format is invalid or cannot be parsed correctly.
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

	// Block Logic: Handles KubernetesVolumeIDs that do not explicitly start with "aws://".
	// Assumes a bare AWS volume ID and constructs a compliant URI with an empty host for the AZ.
	if !strings.HasPrefix(s, "aws://") {
		// Assume a bare aws volume id (vol-1234...)
		// Build a URL with an empty host (AZ)
		s = "aws://" + "" + "/" + s
	}
	// Functional Utility: Parses the string representation of the volume ID into a URL structure.
	url, err := url.Parse(s)
	// Block Logic: Error handling for URL parsing failures.
	if err != nil {
		// TODO: Maybe we should pass a URL into the Volume functions
		return "", fmt.Errorf("Invalid disk name (%s): %v", name, err)
	}
	// Block Logic: Validates that the scheme of the parsed URL is "aws".
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS volume (%s)", name)
	}

	// Functional Utility: Extracts the path part of the URL and trims leading/trailing slashes
	// to isolate the AWS volume ID.
	awsID := url.Path
	awsID = strings.Trim(awsID, "/")

	// We sanity check the resulting volume; the two known formats are
	// vol-12345678 and vol-12345678abcdef01
	// TODO: Regex match?
	// Block Logic: Performs a basic sanity check on the extracted AWS volume ID.
	// It checks if the ID contains slashes or does not start with the "vol-" prefix,
	// indicating an invalid format.
	if strings.Contains(awsID, "/") || !strings.HasPrefix(awsID, "vol-") {
		return "", fmt.Errorf("Invalid format for AWS volume (%s)", name)
	}

	return awsVolumeID(awsID), nil
}
