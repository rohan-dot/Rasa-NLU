grep -iE "apply_edit|read_source|find_ref|test_patch|TestResult|build_ok|crash_gone|tests_pass|iteration|turn" orchestrator.log | head -80

xx

grep -iE "libCRS|rebuild|REBUILD_OUT|mvn|javac|BUILD (SUCCESS|FAILURE)" orchestrator.log | head -40
