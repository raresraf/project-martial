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

// Package upgrade implements the `kubeadm upgrade apply` command.
package upgrade

import (
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/util/version"
)

const (
	// upgradeManifestTimeout defines the timeout for static pod manifest upgrades.
	upgradeManifestTimeout = 1 * time.Minute
)

// applyFlags holds the information about the flags that can be passed to the apply command.
// This is a common pattern for organizing CLI command state.
type applyFlags struct {
	nonInteractiveMode bool
	force              bool
	dryRun             bool
	etcdUpgrade        bool
	newK8sVersionStr   string
	newK8sVersion      *version.Version
	imagePullTimeout   time.Duration
	parent             *cmdUpgradeFlags
}

// SessionIsInteractive returns true if the session is of an interactive type.
// This can be opted out of with flags like -y, -f, or --dry-run.
func (f *applyFlags) SessionIsInteractive() bool {
	return !f.nonInteractiveMode
}

// NewCmdApply returns the cobra command for `kubeadm upgrade apply`.
// It defines the command's behavior, flags, and execution logic.
func NewCmdApply(parentFlags *cmdUpgradeFlags) *cobra.Command {
	flags := &applyFlags{
		parent:           parentFlags,
		imagePullTimeout: 15 * time.Minute,
		etcdUpgrade:      false,
	}

	cmd := &cobra.Command{
		Use:   "apply [version]",
		Short: "Upgrade your Kubernetes cluster to the specified version.",
		// The Run function is the main entry point for the command's execution.
		Run: func(cmd *cobra.Command, args []string) {
			// Ensure the user is root, a common requirement for system-level changes.
			err := runPreflightChecks(flags.parent.skipPreFlight)
			kubeadmutil.CheckErr(err)

			// Validate that the user has provided exactly one argument: the version.
			err = cmdutil.ValidateExactArgNumber(args, []string{"version"})
			kubeadmutil.CheckErr(err)

			// Store the requested version string.
			flags.newK8sVersionStr = args[0]

			// Set implicit flags based on user input (e.g., --dry-run implies --yes).
			err = SetImplicitFlags(flags)
			kubeadmutil.CheckErr(err)

			// Execute the core upgrade logic.
			err = RunApply(flags)
			kubeadmutil.CheckErr(err)
		},
	}

	// Define and bind the command-line flags to the applyFlags struct.
	cmd.Flags().BoolVarP(&flags.nonInteractiveMode, "yes", "y", flags.nonInteractiveMode, "Perform the upgrade and do not prompt for confirmation (non-interactive mode).")
	cmd.Flags().BoolVarP(&flags.force, "force", "f", flags.force, "Force upgrading although some requirements might not be met. This also implies non-interactive mode.")
	cmd.Flags().BoolVar(&flags.dryRun, "dry-run", flags.dryRun, "Do not change any state, just output what actions would be performed.")
	cmd.Flags().BoolVar(&flags.etcdUpgrade, "etcd-upgrade", flags.etcdUpgrade, "Perform the upgrade of etcd.")
	cmd.Flags().DurationVar(&flags.imagePullTimeout, "image-pull-timeout", flags.imagePullTimeout, "The maximum amount of time to wait for the control plane pods to be downloaded.")

	return cmd
}

// RunApply orchestrates the actual upgrade functionality.
// Its primary architectural role is to sequence the upgrade through a series of phases,
// ensuring safety and correctness at each step.
func RunApply(flags *applyFlags) error {

	// Phase 1: Pre-flight checks and configuration loading.
	// This ensures the cluster is in a healthy, upgradeable state before any changes are made.
	upgradeVars, err := enforceRequirements(flags.parent.kubeConfigPath, flags.parent.cfgPath, flags.parent.printConfig, flags.dryRun)
	if err != nil {
		return err
	}

	// Update the in-memory configuration with the new version.
	upgradeVars.cfg.KubernetesVersion = flags.newK8sVersionStr

	// Convert the versioned, external configuration to an internal type for processing.
	internalcfg := &kubeadmapi.MasterConfiguration{}
	legacyscheme.Scheme.Convert(upgradeVars.cfg, internalcfg, nil)

	// Normalize and validate the requested Kubernetes version.
	if err := configutil.NormalizeKubernetesVersion(internalcfg); err != nil {
		return err
	}

	// Use the normalized version string for all subsequent operations.
	flags.newK8sVersionStr = internalcfg.KubernetesVersion
	k8sVer, err := version.ParseSemantic(flags.newK8sVersionStr)
	if err != nil {
		return fmt.Errorf("unable to parse normalized version %q as a semantic version", flags.newK8sVersionStr)
	}
	flags.newK8sVersion = k8sVer

	// Phase 2: Version skew and policy enforcement.
	// This is a critical safety gate to prevent unsupported upgrades.
	if err := EnforceVersionPolicies(flags, upgradeVars.versionGetter); err != nil {
		return fmt.Errorf("[upgrade/version] FATAL: %v", err)
	}

	// Phase 3: User confirmation.
	// If in interactive mode, require explicit user consent before proceeding.
	if flags.SessionIsInteractive() {
		if err := InteractivelyConfirmUpgrade("Are you sure you want to proceed with the upgrade?"); err != nil {
			return err
		}
	}

	// Phase 4: Image pre-pulling.
	// This architectural pattern uses a DaemonSet to pull required container images
	// onto all control-plane nodes *before* starting the upgrade, minimizing downtime.
	prepuller := upgrade.NewDaemonSetPrepuller(upgradeVars.client, upgradeVars.waiter, internalcfg)
	upgrade.PrepullImagesInParallel(prepuller, flags.imagePullTimeout)

	// Phase 5: Control plane upgrade.
	// This is the core action where the control plane components are upgraded.
	if err := PerformControlPlaneUpgrade(flags, upgradeVars.client, upgradeVars.waiter, internalcfg); err != nil {
		return fmt.Errorf("[upgrade/apply] FATAL: %v", err)
	}

	// Phase 6: Post-upgrade tasks.
	// This includes upgrading RBAC rules and essential addons like CoreDNS and kube-proxy.
	if err := upgrade.PerformPostUpgradeTasks(upgradeVars.client, internalcfg); err != nil {
		return fmt.Errorf("[upgrade/postupgrade] FATAL post-upgrade error: %v", err)
	}

	// If this was a dry run, report success and exit.
	if flags.dryRun {
		fmt.Println("[dryrun] Finished dryrunning successfully!")
		return nil
	}

	// Final success message and instructions for the user.
	fmt.Println("")
	fmt.Printf("[upgrade/successful] SUCCESS! Your cluster was upgraded to %q. Enjoy!
", flags.newK8sVersionStr)
	fmt.Println("")
	fmt.Println("[upgrade/kubelet] Now that your control plane is upgraded, please proceed with upgrading your kubelets in turn.")

	return nil
}

// SetImplicitFlags handles dynamically defaulting flags based on each other's value.
func SetImplicitFlags(flags *applyFlags) error {
	// If we are in dry-run or force mode, we should automatically execute this command non-interactively.
	if flags.dryRun || flags.force {
		flags.nonInteractiveMode = true
	}

	if len(flags.newK8sVersionStr) == 0 {
		return fmt.Errorf("version string can't be empty")
	}

	return nil
}

// EnforceVersionPolicies ensures that the user-specified version is valid to upgrade to.
// It separates errors into fatal (non-skippable) and skippable (if --force is used).
func EnforceVersionPolicies(flags *applyFlags, versionGetter upgrade.VersionGetter) error {
	fmt.Printf("[upgrade/version] You have chosen to upgrade to version %q
", flags.newK8sVersionStr)

	// Check for version skew errors.
	versionSkewErrs := upgrade.EnforceVersionPolicies(versionGetter, flags.newK8sVersionStr, flags.newK8sVersion, flags.parent.allowExperimentalUpgrades, flags.parent.allowRCUpgrades)
	if versionSkewErrs != nil {

		// If there are fatal errors, the upgrade cannot proceed.
		if len(versionSkewErrs.Mandatory) > 0 {
			return fmt.Errorf("The --version argument is invalid due to these fatal errors:

%v
Please fix the misalignments highlighted above and try upgrading again", kubeadmutil.FormatErrMsg(versionSkewErrs.Mandatory))
		}

		// If there are skippable errors, check for the --force flag.
		if len(versionSkewErrs.Skippable) > 0 {
			// If --force is not specified, return an error.
			if !flags.force {
				return fmt.Errorf("The --version argument is invalid due to these errors:

%v
Can be bypassed if you pass the --force flag", kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
			}
			// If --force is specified, print a warning but continue.
			fmt.Printf("[upgrade/version] Found %d potential version compatibility errors but skipping since the --force flag is set: 

%v", len(versionSkewErrs.Skippable), kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
		}
	}
	return nil
}

// PerformControlPlaneUpgrade executes the upgrade procedure for the control plane.
// It detects the type of cluster (self-hosted or static pod-hosted) and calls the appropriate function.
func PerformControlPlaneUpgrade(flags *applyFlags, client clientset.Interface, waiter apiclient.Waiter, internalcfg *kubeadmapi.MasterConfiguration) error {

	// Check if the cluster is self-hosted and act accordingly.
	if upgrade.IsControlPlaneSelfHosted(client) {
		fmt.Printf("[upgrade/apply] Upgrading your Self-Hosted control plane to version %q...
", flags.newK8sVersionStr)

		// Upgrade the self-hosted cluster.
		return upgrade.SelfHostedControlPlane(client, waiter, internalcfg, flags.newK8sVersion)
	}

	// The cluster is hosted using static pods, the default and most common method.
	fmt.Printf("[upgrade/apply] Upgrading your Static Pod-hosted control plane to version %q...
", flags.newK8sVersionStr)

	// If this is a dry run, simulate the upgrade without making changes.
	if flags.dryRun {
		return DryRunStaticPodUpgrade(internalcfg)
	}

	// Perform the actual upgrade for a static pod-hosted cluster.
	return PerformStaticPodUpgrade(client, waiter, internalcfg, flags.etcdUpgrade)
}

// PerformStaticPodUpgrade handles the upgrade of control plane components for a static pod-hosted cluster.
// The core architectural pattern is to write new manifest files to the `/etc/kubernetes/manifests` directory.
// The kubelet on the node watches this directory and automatically restarts the control plane pods with the new configuration.
func PerformStaticPodUpgrade(client clientset.Interface, waiter apiclient.Waiter, internalcfg *kubeadmapi.MasterConfiguration, etcdUpgrade bool) error {
	pathManager, err := upgrade.NewKubeStaticPodPathManagerUsingTempDirs(constants.GetStaticPodDirectory())
	if err != nil {
		return err
	}

	// This function handles the manifest writing and waits for the components to become healthy.
	return upgrade.StaticPodControlPlane(waiter, pathManager, internalcfg, etcdUpgrade)
}

// DryRunStaticPodUpgrade simulates the upgrade of a static pod-hosted control plane.
// It generates the new manifests in a temporary directory and prints them to show what
// *would* be changed, without affecting the live system. This is a crucial feature for safety and predictability.
func DryRunStaticPodUpgrade(internalcfg *kubeadmapi.MasterConfiguration) error {

	dryRunManifestDir, err := constants.CreateTempDirForKubeadm("kubeadm-upgrade-dryrun")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dryRunManifestDir)

	// Generate the new static pod manifests in the temporary directory.
	if err := controlplane.CreateInitStaticPodManifestFiles(dryRunManifestDir, internalcfg); err != nil {
		return err
	}

	// Prepare to print the contents of the generated manifests.
	files := []dryrunutil.FileToPrint{}
	for _, component := range constants.MasterComponents {
		realPath := constants.GetStaticPodFilepath(component, dryRunManifestDir)
		outputPath := constants.GetStaticPodFilepath(component, constants.GetStaticPodDirectory())
		files = append(files, dryrunutil.NewFileToPrint(realPath, outputPath))
	}

	// Print the dry-run files to standard output.
	return dryrunutil.PrintDryRunFiles(files, os.Stdout)
}
