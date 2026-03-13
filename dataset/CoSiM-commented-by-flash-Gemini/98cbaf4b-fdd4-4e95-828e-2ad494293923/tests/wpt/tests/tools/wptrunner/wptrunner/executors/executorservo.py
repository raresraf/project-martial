# mypy: allow-untyped-defs

import base64
import json
import os
import subprocess
import tempfile
import threading
import traceback
import uuid

from mozprocess import ProcessHandler

from tools.serve.serve import make_hosts_file

from .base import (RefTestImplementation,
                   crashtest_result_converter,
                   testharness_result_converter,
                   reftest_result_converter,
                   TimedRunner)
from .process import ProcessTestExecutor
from .protocol import ConnectionlessProtocol
from ..browsers.base import browser_command


pytestrunner = None
webdriver = None


class ServoExecutor(ProcessTestExecutor):
    """
    @brief Base executor for running Web Platform Tests (WPT) with Servo.
    
    This class extends `ProcessTestExecutor` to provide common functionalities
    for all Servo-based WPT executors, including setup, teardown, output handling,
    and building Servo-specific command-line arguments.
    """
    def __init__(self, logger, browser, server_config, timeout_multiplier, debug_info,
                 pause_after_test, reftest_screenshot="unexpected"):
        """
        @brief Initializes the ServoExecutor.
        
        @param logger The logger instance.
        @param browser The browser configuration.
        @param server_config The server configuration.
        @param timeout_multiplier Multiplier for test timeouts.
        @param debug_info Debugging information.
        @param pause_after_test Whether to pause after test execution.
        @param reftest_screenshot Strategy for reftest screenshots.
        """
        ProcessTestExecutor.__init__(self, logger, browser, server_config,
                                     timeout_multiplier=timeout_multiplier,
                                     debug_info=debug_info,
                                     reftest_screenshot=reftest_screenshot)
        self.pause_after_test = pause_after_test
        self.environment = {}
        self.protocol = ConnectionlessProtocol(self, browser)

        # Functional Utility: Creates a temporary hosts file for local server configuration.
        hosts_fd, self.hosts_path = tempfile.mkstemp()
        with os.fdopen(hosts_fd, "w") as f:
            f.write(make_hosts_file(server_config, "127.0.0.1"))

        # Functional Utility: Sets up environment variables for test execution, including the hosts file.
        self.env_for_tests = os.environ.copy()
        self.env_for_tests["HOST_FILE"] = self.hosts_path
        self.env_for_tests["RUST_BACKTRACE"] = "1"

    def teardown(self):
        """
        @brief Cleans up temporary resources.
        
        Removes the temporary hosts file created during initialization.
        """
        try:
            os.unlink(self.hosts_path)
        except OSError:
            pass
        ProcessTestExecutor.teardown(self)

    def on_environment_change(self, new_environment):
        """
        @brief Handles changes in the test environment.
        
        Updates the internal environment dictionary with the new environment settings.
        
        @param new_environment A dictionary representing the new environment.
        @return The result of the superclass's `on_environment_change` method.
        """
        self.environment = new_environment
        return super().on_environment_change(new_environment)

    def on_output(self, line):
        """
        @brief Processes output lines from the Servo subprocess.
        
        Decodes the output line and either prints it to stdout (if interactive)
        or logs it through the logger.
        
        @param line The raw byte string output line.
        """
        line = line.decode("utf8", "replace")
        # Block Logic: Differentiates between interactive and non-interactive output handling.
        if self.interactive:
            print(line)
        else:
            self.logger.process_output(self.proc.pid, line, " ".join(self.command), self.test.url)

    def find_wpt_prefs(self):
        """
        @brief Locates the `wpt-prefs.json` file.
        
        Searches for the `wpt-prefs.json` file in common locations relative to the current
        working directory, accounting for different execution environments (e.g., local
        development vs. WPT runner).
        
        @return The absolute path to the `wpt-prefs.json` file.
        """
        default_path = os.path.join("resources", "wpt-prefs.json")
        # Block Logic: Iterates through potential base directories to find `wpt-prefs.json`.
        for dir in [".", "./_venv3/servo"]:
            candidate = os.path.abspath(os.path.join(dir, default_path))
            if os.path.isfile(candidate):
                return candidate
        self.logger.error("Unable to find wpt-prefs.json")
        return default_path

    def build_servo_command(self, test, extra_args=None):
        """
        @brief Constructs the command-line arguments for running Servo.
        
        Assembles a list of arguments for the Servo executable, including common flags,
        user stylesheets, preferences from the environment, and test-specific URL.
        
        @param test The test object containing URL and other test-specific information.
        @param extra_args Optional list of additional command-line arguments.
        @return A tuple `(debug_args, command)` representing the full Servo command.
        """
        args = [
            "--hard-fail", "-u", "Servo/wptrunner",
            # See https://github.com/servo/servo/issues/30080.
            # For some reason rustls does not like the certificate generated by the WPT tooling.
            "--ignore-certificate-errors",
            "-z", self.test_url(test),
        ]
        # Block Logic: Appends user stylesheets to the Servo command.
        for stylesheet in self.browser.user_stylesheets:
            args += ["--user-stylesheet", stylesheet]
        # Block Logic: Appends environment preferences to the Servo command.
        for pref, value in self.environment.get('prefs', {}).items():
            args += ["--pref", f"{pref}={value}"]
        args += ["--prefs-file", self.wpt_prefs_path]
        # Block Logic: Appends CA certificate path if available.
        if self.browser.ca_certificate_path:
            args += ["--certificate-path", self.browser.ca_certificate_path]
        # Block Logic: Appends extra arguments if provided.
        if extra_args:
            args += extra_args
        args += self.browser.binary_args
        debug_args, command = browser_command(self.binary, args, self.debug_info)
        # Block Logic: Removes the headless flag if `pause_after_test` is enabled for interactive debugging.
        if self.pause_after_test:
            command.remove("-z")
        return debug_args + command


class ServoTestharnessExecutor(ServoExecutor):
    """
    @brief Executor for running `testharness.js` tests with Servo.
    
    This executor specifically handles tests written using the `testharness.js` framework,
    parsing their JSON output and converting it into a standard test result format.
    """
    convert_result = testharness_result_converter

    def __init__(self, logger, browser, server_config, timeout_multiplier=1, debug_info=None,
                 pause_after_test=False, **kwargs):
        """
        @brief Initializes the ServoTestharnessExecutor.
        
        @param logger The logger instance.
        @param browser The browser configuration.
        @param server_config The server configuration.
        @param timeout_multiplier Multiplier for test timeouts.
        @param debug_info Debugging information.
        @param pause_after_test Whether to pause after test execution.
        """
        ServoExecutor.__init__(self, logger, browser, server_config,
                               timeout_multiplier=timeout_multiplier,
                               debug_info=debug_info,
                               pause_after_test=pause_after_test)
        self.result_data = None
        self.result_flag = None

    def do_test(self, test):
        """
        @brief Executes a single `testharness.js` test.
        
        Launches Servo with the test URL, monitors its output for test results,
        and handles timeouts.
        
        @param test The test object to execute.
        @return A tuple `(result, subtests)` representing the test outcome.
        """
        self.test = test
        self.result_data = None
        self.result_flag = threading.Event()

        self.command = self.build_servo_command(test)

        # Block Logic: Differentiates between interactive and non-interactive process handling.
        if not self.interactive:
            # Functional Utility: Uses `mozprocess.ProcessHandler` for non-interactive test execution.
            self.proc = ProcessHandler(self.command,
                                       processOutputLine=[self.on_output],
                                       onFinish=self.on_finish,
                                       env=self.env_for_tests,
                                       storeOutput=False)
            self.proc.run()
        else:
            # Functional Utility: Uses `subprocess.Popen` for interactive test execution.
            self.proc = subprocess.Popen(self.command, env=self.env_for_tests)

        try:
            # Block Logic: Waits for test output or timeout.
            timeout = test.timeout * self.timeout_multiplier
            if not self.interactive and not self.pause_after_test:
                wait_timeout = timeout + 5
                self.result_flag.wait(wait_timeout)
            else:
                wait_timeout = None
                self.proc.wait()

            proc_is_running = True

            # Block Logic: Determines test result based on output data or process state.
            if self.result_flag.is_set():
                if self.result_data is not None:
                    result = self.convert_result(test, self.result_data)
                else:
                    self.proc.wait()
                    result = (test.make_result("CRASH", None), [])
                    proc_is_running = False
            else:
                result = (test.make_result("TIMEOUT", None), [])

            # Block Logic: Handles process termination based on `pause_after_test` and `proc_is_running`.
            if proc_is_running:
                if self.pause_after_test:
                    self.logger.info("Pausing until the browser exits")
                    self.proc.wait()
                else:
                    self.proc.kill()
        except:  # noqa
            self.proc.kill()
            raise

        return result

    def on_output(self, line):
        """
        @brief Overrides `on_output` to capture `testharness.js` results.
        
        Looks for lines starting with "ALERT: RESULT: " to extract JSON test results.
        
        @param line The raw byte string output line from the Servo subprocess.
        """
        prefix = "ALERT: RESULT: "
        decoded_line = line.decode("utf8", "replace")
        # Block Logic: Checks if the line contains the testharness result prefix.
        if decoded_line.startswith(prefix):
            self.result_data = json.loads(decoded_line[len(prefix):])
            self.result_flag.set()
        else:
            ServoExecutor.on_output(self, line)

    def on_finish(self):
        """
        @brief Sets the result flag when the process finishes.
        
        Notifies any waiting threads that the Servo subprocess has completed its execution.
        """
        self.result_flag.set()


class TempFilename:
    """
    @brief A context manager for creating and managing temporary filenames.
    
    Ensures that temporary files are automatically cleaned up after use.
    """
    def __init__(self, directory):
        """
        @brief Initializes TempFilename.
        
        @param directory The directory where the temporary file will be created.
        """
        self.directory = directory
        self.path = None

    def __enter__(self):
        """
        @brief Enters the runtime context, creating a unique temporary file path.
        
        @return The generated temporary file path.
        """
        self.path = os.path.join(self.directory, str(uuid.uuid4()))
        return self.path

    def __exit__(self, *args, **kwargs):
        """
        @brief Exits the runtime context, cleaning up the temporary file.
        """
        try:
            os.unlink(self.path)
        except OSError:
            pass


class ServoRefTestExecutor(ServoExecutor):
    """
    @brief Executor for running reftests with Servo.
    
    This executor specifically handles reftests, which involve capturing screenshots
    of rendered pages and comparing them against reference images.
    """
    convert_result = reftest_result_converter

    def __init__(self, logger, browser, server_config, binary=None, timeout_multiplier=1,
                 screenshot_cache=None, debug_info=None, pause_after_test=False,
                 reftest_screenshot="unexpected", **kwargs):
        """
        @brief Initializes the ServoRefTestExecutor.
        
        @param logger The logger instance.
        @param browser The browser configuration.
        @param server_config The server configuration.
        @param binary Path to the Servo binary.
        @param timeout_multiplier Multiplier for test timeouts.
        @param screenshot_cache Cache for reftest screenshots.
        @param debug_info Debugging information.
        @param pause_after_test Whether to pause after test execution.
        @param reftest_screenshot Strategy for reftest screenshots.
        """
        ServoExecutor.__init__(self,
                               logger,
                               browser,
                               server_config,
                               timeout_multiplier=timeout_multiplier,
                               debug_info=debug_info,
                               reftest_screenshot=reftest_screenshot,
                               pause_after_test=pause_after_test)

        self.screenshot_cache = screenshot_cache
        self.reftest_screenshot = reftest_screenshot
        self.implementation = RefTestImplementation(self)
        # Functional Utility: Creates a temporary directory for storing reftest outputs.
        self.tempdir = tempfile.mkdtemp()

    def reset(self):
        """
        @brief Resets the reftest implementation state.
        """
        self.implementation.reset()

    def teardown(self):
        """
        @brief Cleans up temporary resources.
        
        Removes the temporary directory created for reftest outputs.
        """
        os.rmdir(self.tempdir)
        ServoExecutor.teardown(self)

    def screenshot(self, test, viewport_size, dpi, page_ranges):
        """
        @brief Captures a screenshot of the rendered page.
        
        Launches Servo in headless mode to render the test URL and saves the output
        as a PNG image.
        
        @param test The test object for which to capture the screenshot.
        @param viewport_size The desired viewport size (e.g., "800x600").
        @param dpi The device pixel ratio.
        @param page_ranges Ranges of pages to screenshot (unused).
        @return A tuple `(success, data)` where `success` is a boolean indicating
                if the screenshot was successful, and `data` contains the base64
                encoded image data or an error message.
        """
        with TempFilename(self.tempdir) as output_path:
            output_path = f"{output_path}.png"
            extra_args = ["--exit",
                          "--output=%s" % output_path,
                          "--window-size", viewport_size or "800x600"]

            # Block Logic: Appends DPI argument if provided.
            if dpi:
                extra_args += ["--device-pixel-ratio", dpi]

            self.command = self.build_servo_command(test, extra_args)

            # Block Logic: Differentiates between interactive and non-interactive process handling.
            if not self.interactive:
                # Functional Utility: Uses `mozprocess.ProcessHandler` for non-interactive screenshot capture.
                self.proc = ProcessHandler(self.command,
                                           processOutputLine=[self.on_output],
                                           env=self.env_for_tests)

                try:
                    self.proc.run()
                    timeout = test.timeout * self.timeout_multiplier + 5
                    rv = self.proc.wait(timeout=timeout)
                except KeyboardInterrupt:
                    self.proc.kill()
                    raise
            else:
                # Functional Utility: Uses `subprocess.Popen` for interactive screenshot capture.
                self.proc = subprocess.Popen(self.command, env=self.env_for_tests)
                try:
                    rv = self.proc.wait()
                except KeyboardInterrupt:
                    self.proc.kill()
                    raise

            # Block Logic: Checks process return value and existence of output file to determine success.
            if rv is None:
                self.proc.kill()
                return False, ("EXTERNAL-TIMEOUT", None)

            if rv != 0 or not os.path.exists(output_path):
                return False, ("CRASH", None)

            with open(output_path, "rb") as f:
                # Might need to strip variable headers or something here
                data = f.read()
                # Returning the screenshot as a string could potentially be avoided,
                # see https://github.com/web-platform-tests/wpt/issues/28929.
                return True, [base64.b64encode(data).decode()]

    def do_test(self, test):
        """
        @brief Executes a single reftest.
        
        Runs the reftest implementation to capture screenshots and returns the converted result.
        
        @param test The test object to execute.
        @return The converted reftest result.
        """
        self.test = test
        result = self.implementation.run_test(test)

        return self.convert_result(test, result)


class ServoTimedRunner(TimedRunner):
    """
    @brief A timed runner specifically adapted for Servo's testing environment.
    
    This class is used to execute a given function (`run_func`) with a timeout,
    capturing any exceptions and formatting them for test results.
    """
    def run_func(self):
        """
        @brief Executes the wrapped function and captures its result or any exceptions.
        
        Sets `self.result` to `(True, func_result)` on success, or `(False, error_message)`
        on failure.
        """
        try:
            self.result = (True, self.func(self.protocol, self.url, self.timeout))
        except Exception as e:
            message = getattr(e, "message", "")
            if message:
                message += "
"
            message += traceback.format_exc(e)
            self.result = False, ("INTERNAL-ERROR", message)
        finally:
            self.result_flag.set()

    def set_timeout(self):
        """
        @brief Placeholder for timeout setting.
        
        This method is a no-op in this implementation, as timeouts are handled externally
        by `TimedRunner`'s `run` method.
        """
        pass


class ServoCrashtestExecutor(ServoExecutor):
    """
    @brief Executor for running crashtests with Servo.
    
    This executor is designed to verify that Servo handles expected crashes gracefully
    and reports them correctly.
    """
    convert_result = crashtest_result_converter

    def __init__(self, logger, browser, server_config, binary=None, timeout_multiplier=1,
                 screenshot_cache=None, debug_info=None, pause_after_test=False,
                 **kwargs):
        """
        @brief Initializes the ServoCrashtestExecutor.
        
        @param logger The logger instance.
        @param browser The browser configuration.
        @param server_config The server configuration.
        @param binary Path to the Servo binary.
        @param timeout_multiplier Multiplier for test timeouts.
        @param screenshot_cache Cache for reftest screenshots (unused).
        @param debug_info Debugging information.
        @param pause_after_test Whether to pause after test execution.
        """
        ServoExecutor.__init__(self,
                               logger,
                               browser,
                               server_config,
                               timeout_multiplier=timeout_multiplier,
                               debug_info=debug_info,
                               pause_after_test=pause_after_test)

        self.pause_after_test = pause_after_test
        self.protocol = ConnectionlessProtocol(self, browser)

    def do_test(self, test):
        """
        @brief Executes a single crashtest.
        
        Launches Servo with the test URL in a mode that expects a crash,
        and then verifies the process's exit status.
        
        @param test The test object to execute.
        @return The converted crashtest result.
        """
        # Block Logic: Determines the timeout for the test execution.
        timeout = (test.timeout * self.timeout_multiplier if self.debug_info is None
                   else None)

        test_url = self.test_url(test)
        # We want to pass the full test object into build_servo_command,
        # so stash it in the class
        self.test = test
        success, data = ServoTimedRunner(self.logger, self.do_crashtest, self.protocol,
                                         test_url, timeout, self.extra_timeout).run()
        # Ensure that no processes hang around if they timeout.
        self.proc.kill()

        if success:
            return self.convert_result(test, data)

        return (test.make_result(*data), [])

    def do_crashtest(self, protocol, url, timeout):
        """
        @brief Internal method to run the actual crashtest subprocess.
        
        Launches Servo with the `--exit` flag and monitors its process for a crash.
        
        @param protocol The test protocol (unused).
        @param url The URL of the crashtest.
        @param timeout The timeout for the test (unused).
        @return A dictionary indicating the status of the crashtest ("PASS" or "CRASH").
        """
        # Functional Utility: Builds the Servo command with `--exit` flag to indicate expected termination.
        self.command = self.build_servo_command(self.test, extra_args=["-x"])

        # Block Logic: Differentiates between interactive and non-interactive process handling.
        if not self.interactive:
            # Functional Utility: Uses `mozprocess.ProcessHandler` for non-interactive crashtest execution.
            self.proc = ProcessHandler(self.command,
                                       env=self.env_for_tests,
                                       processOutputLine=[self.on_output],
                                       storeOutput=False)
            self.proc.run()
        else:
            # Functional Utility: Uses `subprocess.Popen` for interactive crashtest execution.
            self.proc = subprocess.Popen(self.command, env=self.env_for_tests)

        self.proc.wait()

        # Block Logic: Checks the process return code to determine if a crash occurred.
        if self.proc.poll() >= 0:
            return {"status": "PASS", "message": None}
        return {"status": "CRASH", "message": None}
