from std.python import Python


def main() raises:
    var glob = Python.import_module("glob")
    var os = Python.import_module("os")
    var pty = Python.import_module("pty")
    var subprocess = Python.import_module("subprocess")
    var sys = Python.import_module("sys")
    var re = Python.import_module("re")
    var time = Python.import_module("time")
    var py_str = Python.import_module("builtins").str

    var test_files_obj = glob.glob("tests/test_*.mojo")
    var py_builtins = Python.import_module("builtins")
    var test_files = py_builtins.sorted(test_files_obj)

    var total_files = Int(String(len(test_files)))
    if total_files == 0:
        print("No test files found in tests/")
        return

    var conda_prefix = os.environ.get("CONDA_PREFIX", "")
    var linker_flags = Python.list()
    if conda_prefix:
        linker_flags.append("-Xlinker")
        linker_flags.append("-L" + String(conda_prefix) + "/lib")
        linker_flags.append("-Xlinker")
        linker_flags.append("-llapack")

    var total_tests = 0
    var total_passed = 0
    var total_failed = 0
    var total_skipped = 0
    var failed_files = Python.list()

    var start_time = time.time()
    var summary_pattern = re.compile(
        r"Summary\s+\[\s*[\d\.]+\s*\]\s+(\d+)\s+tests"
        r" run:\s+(\d+)\s+passed\s*,\s*(\d+)\s+failed\s*,\s*(\d+)\s+skipped"
    )

    for i in range(total_files):
        var test_file = test_files[i]
        var cmd = Python.list()
        cmd.append("mojo")
        cmd.append("run")
        cmd.append("-I")
        cmd.append(".")
        cmd.extend(linker_flags)
        cmd.append(test_file)

        # Allocate a pseudo-terminal (PTY) so Mojo detects a real TTY and outputs full native ANSI colors
        var pty_pair = pty.openpty()
        var master_fd = pty_pair[0]
        var s_fd = pty_pair[1]

        var proc = subprocess.Popen(
            cmd, stdout=s_fd, stderr=s_fd, close_fds=True
        )
        os.close(s_fd)

        var master_file = os.fdopen(master_fd, "r")
        var file_tests = 0
        var file_passed = 0
        var file_failed = 0
        var file_skipped = 0

        while True:
            var line: String
            try:
                var line_obj = master_file.readline()
                if not line_obj:
                    break
                line = String(line_obj)
            except:
                break

            sys.stdout.write(line)
            sys.stdout.flush()

            # Strip ANSI escape sequences before pattern matching
            var clean_line = String(re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line))
            var res = summary_pattern.search(clean_line)
            if res:
                file_tests = Int(String(res.group(1)))
                file_passed = Int(String(res.group(2)))
                file_failed = Int(String(res.group(3)))
                file_skipped = Int(String(res.group(4)))

        try:
            master_file.close()
        except:
            pass
        proc.wait()

        total_tests += file_tests
        total_passed += file_passed
        total_failed += file_failed
        total_skipped += file_skipped

        if proc.returncode != 0 or file_failed > 0:
            failed_files.append(test_file)

    var elapsed_val = Float64(String(time.time() - start_time))
    var failed_files_count = Int(String(len(failed_files)))
    var passed_files = total_files - failed_files_count
    var is_success = failed_files_count == 0 and total_failed == 0

    # --- ANSI Styling ---
    var RESET = "\033[0m"
    var BOLD = "\033[1m"

    var FG_GREEN = "\033[38;5;48m"
    var FG_RED = "\033[38;5;203m"
    var FG_YELLOW = "\033[38;5;221m"
    var FG_CYAN = "\033[38;5;75m"
    var FG_WHITE = "\033[38;5;255m"
    var FG_MUTED = "\033[38;5;242m"

    var BG_GREEN = "\033[48;5;40;38;5;232;1m"
    var BG_RED = "\033[48;5;196;38;15;1m"

    var hr = String(py_str("─") * 48)

    # --- 1. Header Divider & Status Badge ---
    print("\n  " + FG_MUTED + hr + RESET)
    if is_success:
        print(
            "  "
            + BG_GREEN
            + " [PASS] "
            + RESET
            + " "
            + FG_GREEN
            + BOLD
            + "All test suites passed successfully"
            + RESET
        )
    else:
        print(
            "  "
            + BG_RED
            + " [FAIL] "
            + RESET
            + " "
            + FG_RED
            + BOLD
            + "Test suite execution failed"
            + RESET
        )
    print()

    # --- 2. Progress Bar ---
    var bar_width = 24
    var pass_ratio = 1.0
    if total_tests > 0:
        pass_ratio = Float64(total_passed) / Float64(total_tests)
    var filled = Int(pass_ratio * Float64(bar_width))
    var empty = bar_width - filled

    var filled_bar = String(py_str("━") * filled)
    var empty_bar = String(py_str("━") * empty)
    var pct_str = String(Int(pass_ratio * 100))

    print(
        "    "
        + FG_MUTED
        + "Progress:  "
        + RESET
        + FG_GREEN
        + filled_bar
        + FG_RED
        + empty_bar
        + RESET
        + " "
        + FG_MUTED
        + pct_str
        + "%"
        + RESET
    )

    # --- 3. Test Suites ---
    print("    " + FG_WHITE + BOLD + "Suites:    " + RESET, end="")
    if passed_files > 0:
        print(
            FG_GREEN + BOLD + String(passed_files) + " passed" + RESET, end=""
        )
        if failed_files_count > 0:
            print(FG_MUTED + ", " + RESET, end="")
    if failed_files_count > 0:
        print(
            FG_RED + BOLD + String(failed_files_count) + " failed" + RESET,
            end="",
        )
    print(FG_MUTED + " (" + String(total_files) + " total)" + RESET)

    # --- 4. Total Tests ---
    print("    " + FG_WHITE + BOLD + "Tests:     " + RESET, end="")
    if total_passed > 0:
        print(
            FG_GREEN + BOLD + String(total_passed) + " passed" + RESET, end=""
        )
        if total_failed > 0 or total_skipped > 0:
            print(FG_MUTED + ", " + RESET, end="")
    if total_failed > 0:
        print(FG_RED + BOLD + String(total_failed) + " failed" + RESET, end="")
        if total_skipped > 0:
            print(FG_MUTED + ", " + RESET, end="")
    if total_skipped > 0:
        print(
            FG_YELLOW + BOLD + String(total_skipped) + " skipped" + RESET,
            end="",
        )
    print(FG_MUTED + " (" + String(total_tests) + " total)" + RESET)

    # --- 5. Duration ---
    var time_str = String(py_str("{:.2f}").format(elapsed_val))
    print(
        "    "
        + FG_WHITE
        + BOLD
        + "Duration:  "
        + RESET
        + FG_CYAN
        + time_str
        + "s"
        + RESET
    )
    print("  " + FG_MUTED + hr + RESET)

    # --- 6. Failure Callouts ---
    if not is_success:
        print("\n    " + FG_RED + BOLD + "FAILED SUITES" + RESET)
        for i in range(failed_files_count):
            print(
                "    "
                + FG_RED
                + "│"
                + RESET
                + "  "
                + FG_RED
                + "✖"
                + RESET
                + " "
                + FG_WHITE
                + String(failed_files[i])
                + RESET
            )
        print("    " + FG_RED + "└" + String(py_str("─") * 42) + RESET + "\n")
        sys.exit(1)
