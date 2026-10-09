import subprocess
import os
import sys
import time
import argparse
import re
import signal
from junit_xml import TestCase, TestSuite

# Configuration: output file name
OUTPUT_XML = "test-results.xml"
# Number of trailing output lines attached to a failed case in the report
ERROR_LOG_TAIL_LINES = 10
# Max characters kept per log line (tensor dumps / assert diffs can be huge single lines)
ERROR_LOG_MAX_LINE_CHARS = 500


def error_log_tail(output, case_command, reason):
    """Return the last lines of a failed case's output for <system-err>, with long lines cut."""
    lines = []
    for line in output.splitlines()[-ERROR_LOG_TAIL_LINES:]:
        if len(line) > ERROR_LOG_MAX_LINE_CHARS:
            line = f"{line[:ERROR_LOG_MAX_LINE_CHARS]} ...[truncated {len(line) - ERROR_LOG_MAX_LINE_CHARS} chars]"
        lines.append(line)
    header = f"===== last {len(lines)} lines of `{case_command}` ({reason}) ====="
    return "\n".join([header, *lines])


def run_one_case(i, case_command, total_count):
    """
    Run a single test case.
    case_command: full shell command string with arguments.
    """
    bin_path = case_command.split(' ')[0]
    test_name = os.path.basename(bin_path)

    junit_case = TestCase(name=test_name, classname="shTest")

    print(f"\n[{i}/{total_count}] Running: {test_name}")
    print(f"Command: {case_command}")
    print("-" * 50)

    start_time = time.time()
    failed_keyword_found = False
    stdout_output = []

    try:
        process = subprocess.Popen(
            case_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        output, _ = process.communicate(timeout=1200)
        if output:
            print(output, end='')
            stdout_output.append(output)
            failed_keyword_found = bool(re.search(
                r"\bfail(?:ed|ure)?\b|Traceback \(most recent call last\):|\bAssertionError\b",
                output,
                re.IGNORECASE,
            ))
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration

        if process.returncode == 0 and not failed_keyword_found:
            print(f">>> Result: [PASS]")
            return junit_case, True
        else:
            msg = f"Exit code: {process.returncode}, Failed keyword: {failed_keyword_found}"
            print(f">>> Result: [FAIL] ({msg})")
            junit_case.add_failure_info(message="Test failed", output="".join(stdout_output))
            junit_case.stderr = error_log_tail("".join(stdout_output), case_command, msg)
            return junit_case, False

    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        if output:
            print(output, end='')
            stdout_output.append(output)
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration
        print(f"Timeout: {test_name}")
        junit_case.add_failure_info(
            message="Timeout",
            output="".join(stdout_output) or "Test timed out",
        )
        junit_case.stderr = error_log_tail("".join(stdout_output) or "Test timed out", case_command, "timeout")
        return junit_case, False

    except Exception as e:
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration
        print(f"Exception: {test_name}, error: {e}")
        junit_case.add_error_info(message="Exception occurred", output=str(e))
        junit_case.stderr = error_log_tail("".join(stdout_output) or str(e), case_command, "exception")
        return junit_case, False


def parse_commands(filepath, is_sh):
    """
    Parse test commands from a file.
    - .sh file: only accept lines starting with './examples/'
    - Other files: accept all non-empty, non-comment lines.
    """
    test_commands = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if is_sh and not line.startswith("./examples/"):
                continue
            test_commands.append(line)
    return test_commands


def resolve_build_dir(caselist_path, build_dir):
    """Resolve the test working directory for both .sh and .list inputs."""
    if build_dir:
        return os.path.abspath(build_dir)

    caselist_dir = os.path.dirname(caselist_path)
    if os.path.splitext(caselist_path)[1].lower() == ".sh":
        return os.path.join(caselist_dir, "build")
    return caselist_dir


def run_cases(caselist_path, build_dir, output_xml=OUTPUT_XML):
    if not os.path.exists(caselist_path):
        print(f"Error: test list file not found: {caselist_path}")
        sys.exit(1)

    # Determine file type by extension
    is_sh = os.path.splitext(caselist_path)[1].lower() == ".sh"

    # Parse commands based on file type
    test_commands = parse_commands(caselist_path, is_sh)

    if not test_commands:
        print("No valid test commands found in list")
        return

    # Switch to build directory
    try:
        os.chdir(build_dir)
        print(f"Working directory changed to: {os.getcwd()}")
    except Exception as e:
        print(f"Failed to change directory: {e}")
        sys.exit(1)

    all_test_objects = []
    global_passed_count = 0
    total_count = len(test_commands)

    # Run all cases
    for i, cmd in enumerate(test_commands, 1):
        try:
            junit_case_obj, is_success = run_one_case(i, cmd, total_count)
            all_test_objects.append(junit_case_obj)
            if is_success:
                global_passed_count += 1
        except Exception as e:
            print(f"Critical error: {e}")

    # Resolve after changing directory so relative paths use the test working directory.
    output_xml_path = os.path.abspath(output_xml)
    output_xml_dir = os.path.dirname(output_xml_path)
    os.makedirs(output_xml_dir, exist_ok=True)

    # Generate JUnit XML report
    suite = TestSuite("actlize GEMM Tests", all_test_objects)
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        TestSuite.to_file(f, [suite], prettyprint=True)

    print(f"\nTests completed. Total: {total_count}, Passed: {global_passed_count}")
    print(f"Report generated: {output_xml_path}")

    if global_passed_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run actlize tests from a caselist or shell script.")
    parser.add_argument("--caselist", required=True, help="Path to the caselist.list or run.sh file")
    parser.add_argument(
        "--build-dir",
        help=(
            "Directory in which test commands run. Defaults to <run.sh dir>/build "
            "for .sh input and the caselist directory for other input files."
        ),
    )
    parser.add_argument(
        "--output-xml",
        default=OUTPUT_XML,
        help=(
            "JUnit XML output path. Absolute paths are used directly; relative "
            f"paths are resolved from the test working directory (default: {OUTPUT_XML})."
        ),
    )
    args = parser.parse_args()

    caselist_abs_path = os.path.abspath(args.caselist)
    build_dir = resolve_build_dir(caselist_abs_path, args.build_dir)

    run_cases(caselist_abs_path, build_dir, args.output_xml)
