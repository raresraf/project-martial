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

/**
 * @file image_puller.go
 * @brief Implements the image pulling logic for Kubernetes Kubelet.
 *
 * This file defines the `imagePuller` struct and its associated methods,
 * responsible for orchestrating the pulling of container images from registries.
 * It integrates with the container runtime, handles image pull policies,
 * reports events, and implements back-off strategies for failed pulls.
 * The primary goal is to ensure that required container images are available
 * on the node for Pods to start.
 */

package container

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

/**
 * @brief imagePuller is an implementation of the ImagePuller interface.
 *
 * This struct is responsible for pulling container images using the underlying
 * container runtime. It wraps the `Runtime.PullImage()` method and provides
 * additional functionality such as image presence checks, event reporting,
 * and back-off handling for failed image pulls.
 *
 * @field recorder record.EventRecorder: Used for recording events related to image pulling (e.g., "image pulling", "image pulled").
 * @field runtime Runtime: The interface to the container runtime, used to perform actual image pull operations.
 * @field backOff *flowcontrol.Backoff: Manages back-off delays for image pull operations to prevent hammering the registry on repeated failures.
 */
type imagePuller struct {
	recorder record.EventRecorder
	runtime  Runtime
	backOff  *flowcontrol.Backoff
}

// enforce compatibility.
var _ ImagePuller = &imagePuller{}

/**
 * @brief NewImagePuller creates a new ImagePuller instance.
 *
 * This function constructs an `imagePuller` that implements the `ImagePuller`
 * interface. It initializes the `imagePuller` with an event recorder for
 * reporting pull-related events, a container runtime for executing image pull
 * operations, and a back-off manager for handling retries.
 *
 * @param recorder record.EventRecorder: The event recorder to use for publishing events.
 * @param runtime Runtime: The container runtime interface to interact with.
 * @param imageBackOff *flowcontrol.Backoff: The back-off manager for image pull operations.
 * @return ImagePuller: A new instance of the ImagePuller.
 */
func NewImagePuller(recorder record.EventRecorder, runtime Runtime, imageBackOff *flowcontrol.Backoff) ImagePuller {
	return &imagePuller{
		recorder: recorder,
		runtime:  runtime,
		backOff:  imageBackOff,
	}
}

/**
 * @brief shouldPullImage determines if an image pull is required.
 *
 * This function evaluates whether an image should be pulled based on the
 * container's `ImagePullPolicy` and whether the image is already present
 * on the machine. It implements the logic for `PullAlways`, `PullNever`,
 * and `PullIfNotPresent` policies.
 *
 * @param container *api.Container: The container definition with its image pull policy.
 * @param imagePresent bool: A boolean indicating if the image is already available locally.
 * @return bool: True if the image should be pulled, false otherwise.
 */
func shouldPullImage(container *api.Container, imagePresent bool) bool {
	if container.ImagePullPolicy == api.PullNever {
		return false
	}

	if container.ImagePullPolicy == api.PullAlways ||
		(container.ImagePullPolicy == api.PullIfNotPresent && (!imagePresent)) {
		return true
	}

	return false
}

/**
 * @brief logIt records an event and logs a message.
 * Functional Utility: This helper method centralizes event recording and
 *                     logging for image pull operations. If a `ref` (ObjectReference)
 *                     is provided, it records an event using the `puller.recorder`.
 *                     Otherwise, it logs the message directly using `glog`.
 *
 * @param ref *api.ObjectReference: A reference to the API object associated with the event.
 * @param eventtype string: The type of the event (e.g., api.EventTypeNormal, api.EventTypeWarning).
 * @param event string: A short, machine-readable description of the event.
 * @param prefix string: A prefix for the log message (e.g., podName/containerImage).
 * @param msg string: The human-readable message describing the event or action.
 * @param logFn func(args ...interface{}): The glog function to use for logging if no ref is provided.
 */
func (puller *imagePuller) logIt(ref *api.ObjectReference, eventtype, event, prefix, msg string, logFn func(args ...interface{})) {
	if ref != nil {
		puller.recorder.Event(ref, eventtype, event, msg)
	} else {
		logFn(fmt.Sprint(prefix, " ", msg))
	}
}

/**
 * @brief PullImage pulls the image for the specified pod and container.
 * Functional Utility: This method orchestrates the entire image pulling process.
 *                     It first checks if the image needs to be pulled based on
 *                     the `ImagePullPolicy`. It handles back-off for failed
 *                     pulls, records events (e.g., "Pulling image", "Pulled image"),
 *                     and interacts with the underlying container runtime to
 *                     perform the actual image pull.
 *
 * @param pod *api.Pod: The Pod for which the image is being pulled.
 * @param container *api.Container: The container for which the image is being pulled.
 * @param pullSecrets []api.Secret: A slice of secrets used for authenticating with the image registry.
 * @return (error, string): An error if the image pull fails, and a descriptive message.
 */
func (puller *imagePuller) PullImage(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string) {
	logPrefix := fmt.Sprintf("%s/%s", pod.Name, container.Image)
	ref, err := GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	spec := ImageSpec{container.Image}
	present, err := puller.runtime.IsImagePresent(spec)
	if err != nil {
		msg := fmt.Sprintf("Failed to inspect image %q: %v", container.Image, err)
		puller.logIt(ref, api.EventTypeWarning, events.FailedToInspectImage, logPrefix, msg, glog.Warning)
		return ErrImageInspect, msg
	}

	if !shouldPullImage(container, present) {
		if present {
			msg := fmt.Sprintf("Container image %q already present on machine", container.Image)
			puller.logIt(ref, api.EventTypeNormal, events.PulledImage, logPrefix, msg, glog.Info)
			return nil, ""
		} else {
			msg := fmt.Sprintf("Container image %q is not present with pull policy of Never", container.Image)
			puller.logIt(ref, api.EventTypeWarning, events.ErrImageNeverPullPolicy, logPrefix, msg, glog.Warning)
			return ErrImageNeverPull, msg
		}
	}

	backOffKey := fmt.Sprintf("%s_%s", pod.UID, container.Image)
	if puller.backOff.IsInBackOffSinceUpdate(backOffKey, puller.backOff.Clock.Now()) {
		msg := fmt.Sprintf("Back-off pulling image %q", container.Image)
		puller.logIt(ref, api.EventTypeNormal, events.BackOffPullImage, logPrefix, msg, glog.Info)
		return ErrImagePullBackOff, msg
	}
	puller.logIt(ref, api.EventTypeNormal, events.PullingImage, logPrefix, fmt.Sprintf("pulling image %q", container.Image), glog.Info)
	if err := puller.runtime.PullImage(spec, pullSecrets); err != nil {
		puller.logIt(ref, api.EventTypeWarning, events.FailedToPullImage, logPrefix, fmt.Sprintf("Failed to pull image %q: %v", container.Image, err), glog.Warning)
		puller.backOff.Next(backOffKey, puller.backOff.Clock.Now())
		if err == RegistryUnavailable {
			msg := fmt.Sprintf("image pull failed for %s because the registry is unavailable.", container.Image)
			return err, msg
		} else {
			return ErrImagePull, err.Error()
		}
	}
	puller.logIt(ref, api.EventTypeNormal, events.PulledImage, logPrefix, fmt.Sprintf("Successfully pulled image %q", container.Image), glog.Info)
	puller.backOff.DeleteEntry(backOffKey)
	puller.backOff.GC()
	return nil, ""
}
