"""
@a3dd4c1a-5bc7-4db9-9ba2-82139d4cb64b/python/servo/gstreamer.py
@brief Build-system utility for orchestrating GStreamer dependency packaging and dynamic library path rewriting.
Domain: Build Engineering, Multi-platform Deployment (macOS/Windows).
Architecture: Integrates with the Servo 'mach' build tool to automate the collection of transitive shared library dependencies.
Functional Utility: Handles platform-specific library discovery (otool/dumpbin), performs binary path modification (install_name_tool), and generates Rust-side plugin registries.
Synchronization: Primarily sequential script execution within a build pipeline; uses atomic file markers for up-to-date checking.
"""

# Copyright 2013 The Servo Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import os.path
import shutil
import subprocess
import sys
from typing import Set

# This file is called as a script from components/servo/build.rs, so
# we need to explicitly modify the search path here.
sys.path[0:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
from servo.platform.build_target import BuildTarget  # noqa: E402

GSTREAMER_BASE_LIBS = [
    # gstreamer
    "gstbase",
    "gstcontroller",
    "gstnet",
    "gstreamer",
    # gst-plugins-base
    "gstapp",
    "gstaudio",
    "gstfft",
    "gstgl",
    "gstpbutils",
    "gstplay",
    "gstriff",
    "gstrtp",
    "gstrtsp",
    "gstsctp",
    "gstsdp",
    "gsttag",
    "gstvideo",
    # gst-plugins-bad
    "gstcodecparsers",
    "gstplayer",
    "gstwebrtc",
    "gstwebrtcnice",
]
"""
These are the GStreamer base libraries used by both MacOS and Windows
platforms. These are distinct from GStreamer plugins, but GStreamer plugins
may have shared object dependencies on them.
"""

GSTREAMER_PLUGIN_LIBS = [
    # gstreamer
    "gstcoreelements",
    "gstnice",
    # gst-plugins-base
    "gstapp",
    "gstaudioconvert",
    "gstaudioresample",
    "gstgio",
    "gstogg",
    "gstopengl",
    "gstopus",
    "gstplayback",
    "gsttheora",
    "gsttypefindfunctions",
    "gstvideoconvertscale",
    "gstvolume",
    "gstvorbis",
    # gst-plugins-good
    "gstaudiofx",
    "gstaudioparsers",
    "gstautodetect",
    "gstdeinterlace",
    "gstid3demux",
    "gstinterleave",
    "gstisomp4",
    "gstmatroska",
    "gstrtp",
    "gstrtpmanager",
    "gstvideofilter",
    "gstvpx",
    "gstwavparse",
    # gst-plugins-bad
    "gstaudiobuffersplit",
    "gstdtls",
    "gstid3tag",
    "gstproxy",
    "gstvideoparsersbad",
    "gstwebrtc",
    # gst-libav
    "gstlibav",
]
"""
The list of plugin libraries themselves, used for both MacOS and Windows.
"""

GSTREAMER_MAC_PLUGIN_LIBS = [
    # gst-plugins-good
    "gstosxaudio",
    "gstosxvideo",
    # gst-plugins-bad
    "gstapplemedia",
]
"""
Plugins that are only used for MacOS.
"""

GSTREAMER_WIN_PLUGIN_LIBS = [
    # gst-plugins-bad
    "gstwasapi"
]
"""
Plugins that are only used for Windows.
"""

GSTREAMER_WIN_DEPENDENCY_LIBS = [
    "avcodec-59.dll",
    "avfilter-8.dll",
    "avformat-59.dll",
    "avutil-57.dll",
    "bz2.dll",
    "ffi-7.dll",
    "gio-2.0-0.dll",
    "glib-2.0-0.dll",
    "gmodule-2.0-0.dll",
    "gobject-2.0-0.dll",
    "graphene-1.0-0.dll",
    "intl-8.dll",
    "libcrypto-1_1-x64.dll",
    "libjpeg-8.dll",
    "libogg-0.dll",
    "libpng16-16.dll",
    "libssl-1_1-x64.dll",
    "libvorbis-0.dll",
    "libvorbisenc-2.dll",
    "libwinpthread-1.dll",
    "nice-10.dll",
    "opus-0.dll",
    "orc-0.4-0.dll",
    "pcre2-8-0.dll",
    "swresample-4.dll",
    "theora-0.dll",
    "theoradec-1.dll",
    "theoraenc-1.dll",
    "z-1.dll",
]
"""
DLLs that GStreamer ships in the Windows distribution that are necessary for
using the plugin selection that we have. This list is curated by a combination
of using `dumpbin` and the errors that appear when starting Servo.
"""


def windows_dlls():
    """
    @brief Aggregates the full set of required Windows DLLs.
    Logic: Combines fixed third-party dependencies with versioned GStreamer base libraries.
    """
    return GSTREAMER_WIN_DEPENDENCY_LIBS + [f"{lib}-1.0-0.dll" for lib in GSTREAMER_BASE_LIBS]


def windows_plugins():
    """
    @brief Returns the list of Windows-compatible GStreamer plugin filenames.
    """
    libs = [*GSTREAMER_PLUGIN_LIBS, *GSTREAMER_WIN_PLUGIN_LIBS]
    return [f"{lib}.dll" for lib in libs]


def macos_plugins():
    """
    @brief Returns the list of macOS-compatible GStreamer plugin filenames.
    """
    plugins = [*GSTREAMER_PLUGIN_LIBS, *GSTREAMER_MAC_PLUGIN_LIBS]

    return [f"lib{plugin}.dylib" for plugin in plugins]


def write_plugin_list(target):
    """
    @brief Generates a Rust static array containing the list of required plugins.
    Functional Utility: Provides the metadata used by Servo's runtime to verify plugin presence.
    @param target The build target triple.
    """
    plugins = []
    if "apple-" in target:
        plugins = macos_plugins()
    elif "-windows-" in target:
        plugins = windows_plugins()
    print(
        """/* This is a generated file. Do not modify. */

pub(crate) static GSTREAMER_PLUGINS: &[&str] = &[
%s
];
"""
        % ",\n".join(map(lambda x: '"' + x + '"', plugins))
    )


def is_macos_system_library(library_path: str) -> bool:
    """
    @brief Filter to exclude standard Apple system libraries from packaging.
    Logic: Identifies libraries residing in protected system paths or related to ASAN instrumentation.
    """
    return library_path.startswith("/System/Library") or library_path.startswith("/usr/lib") or ".asan." in library_path


def rewrite_dependencies_to_be_relative(binary: str, dependency_lines: Set[str], relative_path: str):
    """
    @brief Modifies the Mach-O load commands of a binary to use relative rpaths.
    Functional Utility: Crucial for making macOS bundles portable. Replaces absolute system paths with '@executable_path' relative variants.
    Strategy: Wraps the 'install_name_tool' CLI utility.
    """
    for dependency_line in dependency_lines:
        # Condition: Skip system libraries or those already using @rpath relative addressing.
        if is_macos_system_library(dependency_line) or dependency_line.startswith("@rpath/"):
            continue

        new_path = os.path.join("@executable_path", relative_path, os.path.basename(dependency_line))
        arguments = ["install_name_tool", "-change", dependency_line, new_path, binary]
        try:
            # Side Effect: Mutates the binary on disk.
            subprocess.check_call(arguments)
        except subprocess.CalledProcessError as exception:
            print(f"{arguments} install_name_tool exited with return value {exception.returncode}")


def make_rpath_path_absolute(dylib_path_from_otool: str, rpath: str):
    """
    @brief Resolves @rpath relative paths into absolute filesystem paths.
    Logic: Heuristic search across common GStreamer installation subdirectories (root, parent, gstreamer-1.0).
    """
    if not dylib_path_from_otool.startswith("@rpath/"):
        return dylib_path_from_otool

    # Not every dependency is in the same directory as the binary that is references. For
    # instance, plugins dylibs can be found in "gstreamer-1.0".
    path_relative_to_rpath = dylib_path_from_otool.replace("@rpath/", "")
    for relative_directory in ["", "..", "gstreamer-1.0"]:
        full_path = os.path.join(rpath, relative_directory, path_relative_to_rpath)
        if os.path.exists(full_path):
            return os.path.normpath(full_path)

    raise Exception("Unable to satisfy rpath dependency: " + dylib_path_from_otool)


def find_non_system_dependencies_with_otool(binary_path: str) -> Set[str]:
    """
    @brief Extracts shared library dependencies from a binary using 'otool'.
    Logic: Parses the output of 'otool -L' and filters out standard system linkages.
    """
    process = subprocess.Popen(["/usr/bin/otool", "-L", binary_path], stdout=subprocess.PIPE)
    output = set()

    for line in map(lambda line: line.decode("utf8"), process.stdout):
        if not line.startswith("\t"):
            continue
        dependency = line.split(" ", 1)[0][1:]

        # No need to do any processing for system libraries. They should be
        # present on all macOS systems.
        if not (is_macos_system_library(dependency) or 'librustc-stable_rt' in dependency):
            output.add(dependency)
    return output


def package_gstreamer_dylibs(binary_path: str, library_target_directory: str, target: BuildTarget):
    """
    @brief Orchestrates the recursive collection and relocation of GStreamer dynamic libraries.
    Flow: 
    1. Identifies immediate dependencies of the Servo binary.
    2. Identifies required plugins.
    3. Recursively resolves transitive dependencies for each library.
    4. Copies files to target and rewrites Mach-O headers to use relative paths.
    
    Optimization: Employs a marker file (.gstreamer-version) to skip re-packaging if inventory is up-to-date.
    """

    # This import only works when called from `mach`.
    import servo.platform

    gstreamer_root = servo.platform.get().gstreamer_root(target)
    gstreamer_version = servo.platform.macos.GSTREAMER_PLUGIN_VERSION
    gstreamer_root_libs = os.path.join(gstreamer_root, "lib")

    # This is the relative path from the directory we are packaging the dylibs into and
    # the binary we are packaging them for.
    relative_path = os.path.relpath(library_target_directory, os.path.dirname(binary_path)) + "/"

    # This might be None if we are cross-compiling.
    if not gstreamer_root:
        return True

    # Detect when the packaged library versions do not reflect our current version of GStreamer,
    # by writing a marker file with the packaged GStreamer version into the target directory.
    marker_file = os.path.join(library_target_directory, f".gstreamer-{gstreamer_version}")

    print()
    if os.path.exists(library_target_directory) and os.path.exists(marker_file):
        print(" • GStreamer packaging is up-to-date")
        return True

    if os.path.exists(library_target_directory):
        print(f" • Packaged GStreamer is out of date. Rebuilding into {library_target_directory}")
        shutil.rmtree(library_target_directory)
    else:
        print(f" • Packaging GStreamer into {library_target_directory}")

    os.makedirs(library_target_directory, exist_ok=True)
    try:
        # Collect all the initial binary dependencies for Servo and the plugins that it uses,
        # which are loaded dynmically at runtime and don't appear in `otool` output.
        binary_dependencies = set(find_non_system_dependencies_with_otool(binary_path))
        binary_dependencies.update(
            [os.path.join(gstreamer_root_libs, "gstreamer-1.0", plugin) for plugin in macos_plugins()]
        )

        rewrite_dependencies_to_be_relative(binary_path, binary_dependencies, relative_path)

        number_copied = 0
        pending_to_be_copied = binary_dependencies
        already_copied = set()

        # Block Logic: Transitive closure loop for dependency resolution.
        # Invariant: Continues until no new unique dylibs are discovered in the dependency graph.
        while pending_to_be_copied:
            checking = set(pending_to_be_copied)
            pending_to_be_copied.clear()

            for otool_dependency in checking:
                already_copied.add(otool_dependency)

                # Logic: Converts @rpath strings to real filesystem paths for copying.
                original_dylib_path = make_rpath_path_absolute(otool_dependency, gstreamer_root_libs)
                transitive_dependencies = set(find_non_system_dependencies_with_otool(original_dylib_path))

                # First copy the dylib into the directory where we are collecting them all for
                # packaging, and rewrite its dependencies to be relative to the executable we
                # are packaging them for.
                new_dylib_path = os.path.join(library_target_directory, os.path.basename(original_dylib_path))
                if not os.path.exists(new_dylib_path):
                    number_copied += 1
                    shutil.copyfile(original_dylib_path, new_dylib_path)
                    rewrite_dependencies_to_be_relative(new_dylib_path, transitive_dependencies, relative_path)

                # Now queue up any transitive dependencies for processing in further iteration loops.
                transitive_dependencies.difference_update(already_copied)
                pending_to_be_copied.update(transitive_dependencies)

    except Exception as exception:
        print(f"ERROR: could not package required dylibs: {exception}")
        raise exception

    # Persistence: Updates marker file to signify successful current-version packaging.
    with open(marker_file, "w") as file:
        file.write(gstreamer_version)

    if number_copied:
        print(f" • Processed {number_copied} GStreamer dylibs. ")
        print("   This can cause the startup to be slow due to macOS security protections.")
    return True


if __name__ == "__main__":
    write_plugin_list(sys.argv[1])
