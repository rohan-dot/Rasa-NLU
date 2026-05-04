"""
prompts_thinking.py — Enhanced prompts for reasoning models.

Designed for models with chain-of-thought / thinking capabilities
(Kimi K2.6, o3, o4-mini, Claude with extended thinking, etc.)

Key differences from standard prompts:
1. Multi-step reasoning: trace data flow across function boundaries
2. Backward analysis: work from crash back to public API
3. Fewer, deeper calls: send more context, expect more reasoning
4. Explicit reasoning structure: "First... Then... Therefore..."

Usage:
    from prompts_thinking import SCANNER_PROMPT, HARNESS_PROMPT, ...
    
    # Or swap at runtime:
    if model_supports_thinking:
        from prompts_thinking import *
    else:
        from prompts_standard import *
"""

# ══════════════════════════════════════════════════════════════════
# SCANNER — Multi-function, cross-file reasoning
# ══════════════════════════════════════════════════════════════════

SCANNER_PROMPT = """\
You are an expert security researcher performing a deep code audit.

You will receive multiple functions with their call graph relationships.
Your job is to find REAL vulnerabilities by reasoning about data flow
across function boundaries.

ANALYSIS PROCESS (follow these steps):

Step 1 — ENTRY POINTS: Identify which functions receive external input
(from files, network, user strings, parsed data). These are the attack
surface.

Step 2 — DATA FLOW: For each entry point, trace how input data flows
through the call chain. Track how sizes, lengths, and pointers transform
at each step. Pay attention to:
- Is a size parameter passed without validation?
- Is a length computed from untrusted input without overflow checks?
- Is a pointer assumed non-NULL without a check?
- Is recursion depth bounded?

Step 3 — VULNERABILITY IDENTIFICATION: At each function in the chain,
check if the transformed input can violate the function's assumptions:
- Integer overflow: can size_a + size_b wrap around?
- Buffer overflow: can memcpy length exceed allocation?
- Null deref: can a lookup return NULL that's used without check?
- Stack overflow: can recursion depth be controlled by input?
- Use-after-free: can a free'd pointer be accessed through another path?

Step 4 — EXPLOITABILITY: For each vulnerability, determine if an attacker
can control the input values needed to trigger it through the public API.
This is the critical question — a bug that can't be reached from external
input is a hardening finding, not an exploitable vulnerability.

IMPORTANT: Reason about RECURSIVE call patterns. If function A calls B
which calls A again, check if input can control the recursion depth.
This is how stack overflows work in parsers.

For each vulnerability found, output:
- file, function, bug_type, confidence (0.0-1.0)
- description: the EXACT operation that's vulnerable
- data_flow: complete chain from external input to the bug
- trigger_hint: specific input that triggers it
- reachable_from_api: true/false — can an attacker reach this?

Respond ONLY with a JSON array. Start with [ end with ].
[{"file":"parser.c","function":"parse_value","bug_type":"stack-overflow",
  "confidence":0.95,
  "description":"parse_value calls parse_array which calls parse_value recursively with no depth limit",
  "data_flow":"cJSON_Parse(input) → parse_value → parse_array → parse_value → ... (unbounded recursion)",
  "trigger_hint":"deeply nested JSON arrays: [[[[[[... with 10000+ levels",
  "reachable_from_api":true}]"""


# ══════════════════════════════════════════════════════════════════
# MULTI-FUNCTION SCANNER — Batch analysis
# ══════════════════════════════════════════════════════════════════

BATCH_SCANNER_PROMPT = """\
You are an expert security researcher. You will receive source code for
MULTIPLE related functions from the same codebase, along with their call
graph showing how they relate.

Analyze them TOGETHER, not in isolation. Vulnerabilities often span
multiple functions — a missing check in function A becomes exploitable
because function B passes unvalidated input to it.

For each function group, reason through:

1. Which functions are the PUBLIC API (called by external code)?
2. Which functions are INTERNAL (called only by other functions in this code)?
3. For each internal function, what assumptions does it make about its inputs?
4. Do the callers of each internal function ALWAYS satisfy those assumptions?
5. If not, what input to the public API would violate those assumptions?

Pay special attention to:
- Functions that perform arithmetic on sizes/lengths without overflow checks
- Functions that recurse based on input structure (parsers, tree walkers)
- Functions that free memory in error paths but have other references
- Functions that use sprintf/strcpy into fixed-size buffers

Respond ONLY with a JSON array of vulnerabilities found.
Start with [ end with ]."""


# ══════════════════════════════════════════════════════════════════
# EXPLOITER — Chain-of-thought harness generation
# ══════════════════════════════════════════════════════════════════

HARNESS_PROMPT = """\
You are a security fuzzing engineer. Generate a LibFuzzer harness to
trigger a specific vulnerability.

BEFORE WRITING CODE, reason through these steps:

Step 1 — UNDERSTAND THE FUNCTION:
- What are its parameters and their types?
- Which parameters control the vulnerable operation?
- Is it a public or internal/static function?

Step 2 — PLAN THE APPROACH:
- If the function is static/internal: use #include "{src_file}" to
  access it directly. This is the standard technique for fuzzing
  internal functions.
- If the function requires complex state (contexts, handles): find
  the library's init/create function by reading the source.
- Map fuzz input bytes to parameters: integers via memcpy (4 bytes each),
  strings/buffers by pointing into the fuzz data.

Step 3 — HANDLE THE VULNERABILITY TYPE:
- Integer overflow: pass unclamped integer parameters. Do NOT limit them.
  The whole point is to let the overflow happen.
- Buffer overflow: provide a buffer shorter than the size parameter claims.
- Null deref: set pointer parameters to NULL or create conditions that
  produce NULL (e.g., failed lookup).
- Stack overflow: this needs the PUBLIC API harness, not direct call.
  Call the parser/entry function with deeply nested input.
- Use-after-free: call free, then use the pointer.

Step 4 — WRITE THE CODE:
- Start with #include <stdint.h>, <stddef.h>, <string.h>
- Include the file's own headers (listed below)
- If static function: #include "{src_file}"
- Implement LLVMFuzzerTestOneInput
- Clean up resources
- Return 0

The file {src_file} uses these includes:
{file_includes}

Function signature: {signature}
Bug type: {bug_type}
Description: {description}
Trigger hint: {trigger_hint}

Source code:
```c
{source_code}
```

Now write the harness. Output ONLY C code. No markdown fences.
Start with #include."""


HARNESS_FIX_PROMPT = """\
The harness failed to compile. Reason about what went wrong:

Your code:
```c
{code}
```

Compiler error:
```
{error}
```

Step 1: What does the error message mean?
Step 2: What's the root cause? (missing include, wrong type, redefinition?)
Step 3: What's the fix?

The file {src_file} uses these includes:
{file_includes}

Function signature: {signature}

Output ONLY the fixed C code. No markdown. Start with #include."""


HARNESS_NOCRASH_PROMPT = """\
The harness compiled and ran for {seconds}s ({execs} executions) but
found NO crashes.

Reason about why:

Step 1: Is the harness actually calling the vulnerable function?
Step 2: Are the fuzz-controlled parameters reaching the vulnerable
        operation, or is there validation that blocks them?
Step 3: Is the vulnerability type one that requires going through the
        public API instead of calling directly? (Stack overflows from
        recursion need the parser, not direct function calls.)

The function signature is: {signature}
The bug type is: {bug_type}
Trigger hint: {trigger_hint}

LibFuzzer output:
{fuzzer_output}

Think about what needs to change, then write a COMPLETELY DIFFERENT
harness. Maybe:
- Use the public API instead of direct #include
- Set up the state differently
- Target a different code path to the same function

Output ONLY C code. No markdown. Start with #include."""


# ══════════════════════════════════════════════════════════════════
# REACHABILITY — Backward chain reasoning
# ══════════════════════════════════════════════════════════════════

REACHABILITY_PROMPT = """\
A vulnerability was found in an internal function by calling it directly.
Now determine if an attacker can trigger it through the PUBLIC API.

REASON BACKWARD from the crash:

Step 1 — THE CRASH: The function `{function}` was called with parameter
values that trigger a {bug_type}. Specifically: {description}
The trigger requires: {trigger_hint}

Step 2 — TRACE BACKWARD: The call chain from public API to this function is:
{call_chains}

For each function in the chain (starting from `{function}` going up):
- What parameter values does the caller pass?
- Are those values derived from input, or are they computed internally?
- Is there any validation that would prevent the triggering values?

Step 3 — CONSTRUCT INPUT: If the vulnerability IS reachable, what input
to the public API would cause each function in the chain to pass the
right values down?

Step 4 — GENERATE: Write a Python script that generates the exact input
needed. The script must write to "/tmp/reachability_input".

If you determine the vulnerability is NOT reachable (because some
function in the chain validates or clamps the values), still write a
best-effort script but note in a comment why it likely won't work.

Output ONLY the Python script. No markdown. Start with a comment
explaining your backward reasoning."""


# ══════════════════════════════════════════════════════════════════
# VERIFIER — Structured analysis
# ══════════════════════════════════════════════════════════════════

VERIFIER_PROMPT = """\
Analyze this crash report and classify the vulnerability.

Step 1: What kind of memory safety violation occurred?
  (heap overflow, stack overflow, null deref, use-after-free, etc.)

Step 2: What is the root cause in the source code?
  (missing bounds check, integer overflow, missing null check, etc.)

Step 3: What CWE does this map to?

Step 4: What is the severity?
  Consider: Can an attacker control the crash? Can it lead to code
  execution, or only denial of service?

Do NOT guess exploitability — we test that separately.

Respond ONLY with a JSON object. Start with { end with }.
{"real_vulnerability":true,"cwe":"CWE-122","cwe_name":"Heap Buffer Overflow",
 "severity":"high","root_cause":"one sentence","impact":"one sentence"}"""


# ══════════════════════════════════════════════════════════════════
# CODEBASE MAP — Strategic audit planning
# ══════════════════════════════════════════════════════════════════

CODEBASE_MAP_PROMPT = """\
You are planning a security audit of a C codebase. Given a list of
source files ranked by dangerous-pattern density, identify the most
likely vulnerable functions.

REASONING PROCESS:

Step 1: Which files handle EXTERNAL INPUT? (parsing, reading, decoding)
These are the highest priority.

Step 2: Within those files, which functions perform MEMORY OPERATIONS
(malloc, memcpy, realloc, strcpy) on data derived from external input?

Step 3: For each candidate, what TYPE of vulnerability is most likely?
- Functions with size arithmetic → integer overflow
- Functions with string operations → buffer overflow  
- Recursive functions → stack overflow
- Functions with complex control flow → null deref, use-after-free

Step 4: Rank by LIKELIHOOD and IMPACT. A buffer overflow in a parser
is worse than one in a debug-only function.

Respond with ONLY a JSON array of the TOP 5 most likely vulnerable
functions. Start with [ end with ].
[{"file":"parser.c","function":"parseBuffer","bug_type":"heap-overflow",
  "data_flow":"input → parse → parseBuffer(buf, user_controlled_size)",
  "audit_priority":"critical",
  "reasoning":"This function allocates based on user input without overflow check"}]"""


# ══════════════════════════════════════════════════════════════════
# AUTO-HARNESS — Entry point generation
# ══════════════════════════════════════════════════════════════════

AUTO_HARNESS_PROMPT = """\
Generate a LibFuzzer harness for this library.

REASONING PROCESS:

Step 1: Read the header files. What is the library's main purpose?
(JSON parser? XML parser? Crypto library? Image decoder?)

Step 2: What is the main entry-point function?
(parse, read, decode, process, etc.)

Step 3: What input format does it accept?
- Text/string input (JSON, XML, config) → null-terminate the fuzz data
- Binary input (images, archives, protocols) → pass raw data + size
- Structured input (key-value, multi-part) → may need setup

Step 4: What cleanup is needed after processing?
(free, delete, close, destroy, etc.)

Step 5: What OPTIONS or FLAGS enable more code paths?
Enable all optional features to maximize coverage.

Write a harness that:
1. Includes the necessary headers
2. Converts fuzz input to the right format
3. Calls the main processing function with all options enabled
4. Also calls output/serialization functions if available
   (these often have separate bugs)
5. Cleans up
6. Returns 0

Output ONLY C code. No markdown fences. Start with #include."""
