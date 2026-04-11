# Copyright 2013 The Servo Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

"""
Build script for packaging GStreamer dependencies for the Servo browser engine.

This script is not a runtime library but a tool used during the build process.
Its primary responsibilities are:
1. Generating a Rust source file that lists the required GStreamer plugins,
   allowing Servo to load them dynamically.
2. On macOS, finding all necessary GStreamer dynamic libraries (.dylib), copying
   them into the application bundle, and rewriting their internal paths to be
   relative to the main executable. This makes the application self-contained.
3. On Windows, defining the set of required DLLs for packaging.
"""

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
A list of core GStreamer libraries. These are not plugins themselves but are
dependencies for many plugins and GStreamer applications.
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
The list of GStreamer plugin libraries required by Servo, common to both
macOS and Windows platforms. These provide the actual media processing
capabilities like decoding, encoding, and rendering.
"""

GSTREAMER_MAC_PLUGIN_LIBS = [
    # gst-plugins-good
    "gstosxaudio",
    "gstosxvideo",
    # gst-plugins-bad
    "gstapplemedia",
]
"""
A list of GStreamer plugin libraries that are specific to the macOS platform,
providing integration with native media frameworks like AVFoundation.
"""

GSTREAMER_WIN_PLUGIN_LIBS = [
    # gst-plugins-bad
    "gstwasapi"
]
"""
A list of GStreamer plugin libraries specific to the Windows platform,
such as the WASAPI audio sink.
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
A curated list of third-party DLLs that are dependencies of the selected
GStreamer plugins on Windows. These must be packaged with the application
for it to function correctly.
"""


def windows_dlls():
    """Constructs the full list of necessary GStreamer-related DLLs for Windows."""
    return GSTREAMER_WIN_DEPENDENCY_LIBS + [f"{lib}-1.0-0.dll" for lib in GSTREAMER_BASE_LIBS]


def windows_plugins():
    """Constructs the full list of GStreamer plugin DLLs for Windows."""
    libs = [*GSTREAMER_PLUGIN_LIBS, *GSTREAMER_WIN_PLUGIN_LIBS]
    return [f"{lib}.dll" for lib in libs]


def macos_plugins():
    """Constructs the full list of GStreamer plugin dylibs for macOS."""
    plugins = [*GSTREAMER_PLUGIN_LIBS, *GSTREAMER_MAC_PLUGIN_LIBS]
    return [f"lib{plugin}.dylib" for plugin in plugins]


def write_plugin_list(target):
    """
    Generates and prints a Rust source file containing a static list of plugin filenames.

    This function determines the correct plugin set based on the build target
    (macOS or Windows) and prints a Rust array definition to stdout, which is
    then redirected into a source file by the build system. This allows the Servo
    application to know which plugins it should attempt to load at runtime.

    Args:
        target: A string identifying the build target platform (e.g., "aarch64-apple-darwin").
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
        % ",
".join(map(lambda x: '"' + x + '"', plugins))
    )


def is_macos_system_library(library_path: str) -> bool:
    """
    Checks if a given library path refers to a macOS system library.

    System libraries (e.g., in /usr/lib) should not be packaged with the
    application, as they are assumed to be present on the target system.

    Returns:
        True if the path is a system library, False otherwise.
    """
    return library_path.startswith("/System/Library") or library_path.startswith("/usr/lib") or ".asan." in library_path


def rewrite_dependencies_to_be_relative(binary: str, dependency_lines: Set[str], relative_path: str):
    """
    Uses `install_name_tool` to change the dependency paths within a binary.

    This function iterates through a set of dependency paths found in a binary
    and rewrites them to be relative to the application's executable path
    (using `@executable_path`). This is a crucial step for creating a portable
    macOS application bundle.

    Args:
        binary: The path to the binary (executable or dylib) to modify.
        dependency_lines: A set of original dependency paths to find and replace.
        relative_path: The new relative path to prepend to the dependency filename.
    """
    for dependency_line in dependency_lines:
        if is_macos_system_library(dependency_line) or dependency_line.startswith("@rpath/"):
            continue

        new_path = os.path.join("@executable_path", relative_path, os.path.basename(dependency_line))
        arguments = ["install_name_tool", "-change", dependency_line, new_path, binary]
        try:
            subprocess.check_call(arguments)
        except subprocess.CalledProcessError as exception:
            print(f"{arguments} install_name_tool exited with return value {exception.returncode}")


def make_rpath_path_absolute(dylib_path_from_otool: str, rpath: str):
    """
    Resolves a dylib path containing `@rpath` into an absolute path.

    `@rpath` is a placeholder that tells the dynamic linker to search in a list
    of runtime search paths. This function emulates that search to find the
    actual location of the library on the build machine.

    Args:
        dylib_path_from_otool: The dependency path as reported by `otool`.
        rpath: The base runtime path to search within.

    Returns:
        The resolved absolute path to the dylib.
    """
    if not dylib_path_from_otool.startswith("@rpath/"):
        return dylib_path_from_otool

    path_relative_to_rpath = dylib_path_from_otool.replace("@rpath/", "")
    for relative_directory in ["", "..", "gstreamer-1.0"]:
        full_path = os.path.join(rpath, relative_directory, path_relative_to_rpath)
        if os.path.exists(full_path):
            return os.path.normpath(full_path)

    raise Exception("Unable to satisfy rpath dependency: " + dylib_path_from_otool)


def find_non_system_dependencies_with_otool(binary_path: str) -> Set[str]:
    """
    Uses the `otool -L` command to find all non-system dynamic library dependencies.

    Args:
        binary_path: The path to the binary to inspect.

    Returns:
        A set of strings, where each string is a dependency path.
    """
    process = subprocess.Popen(["/usr/bin/otool", "-L", binary_path], stdout=subprocess.PIPE)
    output = set()

    for line in map(lambda line: line.decode("utf8"), process.stdout):
        if not line.startswith("	"):
            continue
        dependency = line.split(" ", 1)[0][1:]

        if not (is_macos_system_library(dependency) or 'librustc-stable_rt' in dependency):
            output.add(dependency)
    return output


def package_gstreamer_dylibs(binary_path: str, library_target_directory: str, target: BuildTarget):
    """
    The main function for packaging GStreamer dependencies on macOS.

    This function orchestrates the entire process of finding, copying, and
    rewriting GStreamer dylibs to create a self-contained application bundle.
    It recursively traverses the dependency graph of the main executable and
    all required plugins.
    """
    import servo.platform

    gstreamer_root = servo.platform.get().gstreamer_root(target)
    gstreamer_version = servo.platform.macos.GSTREAMER_PLUGIN_VERSION
    gstreamer_root_libs = os.path.join(gstreamer_root, "lib")

    relative_path = os.path.relpath(library_target_directory, os.path.dirname(binary_path)) + "/"

    if not gstreamer_root:
        return True

    # Use a marker file to avoid repackaging if the GStreamer version hasn't changed.
    marker_file = os.path.join(library_target_directory, f".gstreamer-{gstreamer_version}")

    print()
    if os.path.exists(library_target_directory) and os.path.exists(marker_file):
        print(" • GStreamer packaging is up-to-date")
        return True

    if os.path.exists(library_target_directory):
        shutil.rmtree(library_target_directory)
    else:
        print(f" • Packaging GStreamer into {library_target_directory}")

    os.makedirs(library_target_directory, exist_ok=True)
    try:
        # Block Logic: This is a breadth-first traversal of the dependency graph.
        # `pending_to_be_copied` acts as a queue of dependencies to process.
        binary_dependencies = set(find_non_system_dependencies_with_otool(binary_path))
        binary_dependencies.update(
            [os.path.join(gstreamer_root_libs, "gstreamer-1.0", plugin) for plugin in macos_plugins()]
        )

        rewrite_dependencies_to_be_relative(binary_path, binary_dependencies, relative_path)

        number_copied = 0
        pending_to_be_copied = binary_dependencies
        already_copied = set()

        # Invariant: Continue as long as there are new, unprocessed dependencies.
        while pending_to_be_copied:
            checking = set(pending_to_be_copied)
            pending_to_be_copied.clear()

            for otool_dependency in checking:
                already_copied.add(otool_dependency)

                original_dylib_path = make_rpath_path_absolute(otool_dependency, gstreamer_root_libs)
                transitive_dependencies = set(find_non_system_dependencies_with_otool(original_dylib_path))

                # Copy the dylib and rewrite its internal dependency paths.
                new_dylib_path = os.path.join(library_target_directory, os.path.basename(original_dylib_path))
                if not os.path.exists(new_dylib_path):
                    number_copied += 1
                    shutil.copyfile(original_dylib_path, new_dylib_path)
                    rewrite_dependencies_to_be_relative(new_dylib_path, transitive_dependencies, relative_path)

                # Add newly discovered transitive dependencies to the queue for the next iteration.
                transitive_dependencies.difference_update(already_copied)
                pending_to_be_copied.update(transitive_dependencies)

    except Exception as exception:
        print(f"ERROR: could not package required dylibs: {exception}")
        raise exception

    with open(marker_file, "w") as file:
        file.write(gstreamer_version)

    if number_copied:
        print(f" • Processed {number_copied} GStreamer dylibs. ")
        print("   This can cause the startup to be slow due to macOS security protections.")
    return True


if __name__ == "__main__":
    """
    Entry point for the script.

    When called from the command line, this script takes a build target as an
    argument and calls `write_plugin_list` to generate the Rust source file
    containing the list of GStreamer plugins.
    """
    write_plugin_list(sys.argv[1])
