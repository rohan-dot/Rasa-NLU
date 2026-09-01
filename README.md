A container run produced this in patch_index.json for every confirmed crash:

  "reason": "agent gave up: No task context was provided: I have no description of the target, the PoV input, the crash location or stack trace, or the path to the scratch copy of the source. Without at least the crashing function/file or the vulnerability details, I cannot search, edit, or validate anything, so no fix can be produced."
  "attempts": 8, "test_calls": 0

The patch agent is being invoked with no task context. test_calls=0 means it never reached test_patch, so the build oracle was never exercised — this blocks everything upstream of the fix I just made.

Find why the initial task message sent to the patch agent is empty or missing its content. The data appears to exist: main._attach_crash_context (main.py:285) back-fills stack_trace, root_cause and cwe onto the UniqueCrash, and the record already carries crash_type, harness, example_input, all_inputs, finding_id. Trace the path from a UniqueCrash record through run_patch_agent into the first LLM message and find where the context is dropped, empty, or never assembled.

Fix it so the agent's opening message always contains, explicitly labelled: the crash type and CWE, the sink/crash location (file:line) and stack trace, the root cause if available, the PoV input path(s), the absolute path of the scratch source copy it may edit, the harness name, and the exact tool names available to it. If any field is genuinely absent on the record, state that plainly in the message rather than omitting it silently.

Add a hard guard: before starting the loop, verify the assembled context is non-empty and contains at minimum a crash location OR a stack trace. If not, fail immediately with a distinct cause (missing_crash_context) logged at WARNING and recorded in patch_index.json — do not burn 8-15 iterations on an agent with nothing to work from.

Add tests: a realistic UniqueCrash record produces an opening message containing each required field; a record missing stack_trace still yields usable context; an empty record trips the missing_crash_context guard immediately with zero iterations spent. Verify with py_compile and the full suite. Update CHANGELOG. Report where the context was being lost. Commit.
