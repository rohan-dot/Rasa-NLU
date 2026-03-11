Let me read through all the resources you linked before writing anything.

Good — I now have everything I need. Let me write this properly.

---

# CyberGym CRS: Step-by-Step AI Coder Prompt

Here is the full blueprint, broken into **7 sequential prompts**. Give each one to your AI coder in order. Each step produces a concrete module that the next step can import. This prevents context overflow and keeps each generation focused.

---

## 📌 MASTER CONTEXT (paste this at the top of EVERY step prompt)

```
You are building a Cyber Reasoning System (CRS) in Python, evaluated on the CyberGym benchmark,
specifically Task 1 / Level 1.

TASK DEFINITION (Level 1):
  Input:  (1) A pre-patch codebase (repo-vul.tar.gz) + (2) a text description of the vulnerability
  Output: A Proof-of-Concept (PoC) file that, when compiled and executed against the vulnerable
          binary, triggers the vulnerability (crash / sanitizer error)
  Goal:   Bridge a written CVE-style report to a working PoC. No patch file, no error log given.

CONSTRAINTS:
  - Environment: Anaconda (Python 3.10+). No Docker. C/C++ toolchain (gcc/g++/clang) assumed 
    available via conda or system PATH. No guaranteed root.
  - LLM Backend: OpenAI-compatible API (vLLM serving Gemma, or GPT-series). 
    Model is accessed via openai Python SDK pointed at local base_url OR api.openai.com.
  - Fuzzing: Optional module. Use AFL++ or libFuzzer only if they exist in PATH. 
    The system must work WITHOUT a fuzzer and should try fuzzing as a fallback.
  - Languages in dataset: Primarily C and C++.
  - No internet access at inference time (offline codebase only).

PROJECT STRUCTURE (to be built across all steps):
  crs/
  ├── config.py              # Step 1
  ├── data_loader.py         # Step 1
  ├── code_intelligence.py   # Step 2
  ├── llm_router.py          # Step 3
  ├── poc_strategies.py      # Step 4
  ├── build_executor.py      # Step 5
  ├── fuzzer.py              # Step 6
  ├── evaluator.py           # Step 7
  └── main.py                # Step 7
```

---

## 🟢 STEP 1 PROMPT — Data Pipeline & Configuration

```
You are implementing Step 1 of a CRS (Cyber Reasoning System).

Implement TWO files: `crs/config.py` and `crs/data_loader.py`.

=== crs/config.py ===

This file contains all runtime configuration as a dataclass or simple namespace. Include:
  - LLM_BASE_URL: str  (default "http://localhost:8000/v1" for vLLM; overridable via env var OPENAI_BASE_URL)
  - LLM_API_KEY: str   (default "EMPTY" for vLLM; overridable via OPENAI_API_KEY env var)
  - LLM_MODEL: str     (default "google/gemma-3-27b-it"; overridable via LLM_MODEL env var)
  - MAX_TOKENS: int = 4096
  - MAX_RETRIES: int = 3
  - BUILD_TIMEOUT: int = 120   # seconds
  - RUN_TIMEOUT: int = 30      # seconds
  - WORK_DIR: Path             # ~/.crs_workdir by default, created on import
  - USE_SANITIZERS: bool = True
  - FUZZING_ENABLED: bool = True  # will check at runtime if AFL++ or libFuzzer exist in PATH

=== crs/data_loader.py ===

This file handles loading and preparing a CyberGym Level 1 task from the local filesystem.

The CyberGym dataset structure on disk (after downloading via `datasets` lib or manually):
  data/arvo/<task_id>/
      repo-vul.tar.gz   ← the pre-patch codebase
      description.txt   ← text description of the vulnerability

Implement the following:

  class CyberGymTask(dataclass):
      task_id: str
      project_name: str
      project_language: str          # "c" or "c++"
      vulnerability_description: str
      repo_path: Path                # extracted codebase path
      raw_tarball: Path

  def load_task_from_hf(subset_ids: list[str]) -> list[CyberGymTask]:
      """
      Loads a list of tasks by task_id from the HuggingFace `datasets` library.
      Uses: datasets.load_dataset("sunblaze-ucb/cybergym", split="tasks")
      Filters to the requested task IDs.
      For each task, downloads the tar.gz from the HF repo file path given in
      task_difficulty["level1"][0] using huggingface_hub.hf_hub_download().
      Extracts the tarball to WORK_DIR / task_id / "repo" and populates CyberGymTask.
      Downloads description.txt from task_difficulty["level1"][1] similarly.
      Returns list of populated CyberGymTask objects.
      """

  def load_task_from_local(task_dir: Path) -> CyberGymTask:
      """
      Loads a single task from a local directory that already contains:
          repo-vul.tar.gz and description.txt
      Extracts to WORK_DIR / <dir_name> / repo.
      Returns a populated CyberGymTask. 
      Infer project_language from file extensions found in repo 
      (count .c vs .cpp/.cc/.cxx files; majority wins).
      """

  def get_source_files(task: CyberGymTask, max_files: int = 200) -> list[Path]:
      """
      Returns a list of all .c/.cpp/.h/.cc/.cxx files in task.repo_path.
      Truncated to max_files by prioritizing files in src/, lib/, or root 
      over test/ and doc/ directories.
      """

Requirements:
  - Use pathlib throughout
  - Use tarfile for extraction
  - Handle nested tar structures gracefully (sometimes repo is inside a subdirectory in the tar)
  - Print progress messages with print() — no logging framework needed
  - Add a __main__ block that tests load_task_from_local on a hardcoded example path

Do NOT implement any LLM calls, building, or PoC generation here.
```

---

## 🟡 STEP 2 PROMPT — Code Intelligence Module

```
You are implementing Step 2 of a CRS. 
Assume Step 1 is complete: `crs/config.py` and `crs/data_loader.py` exist and work.

Implement `crs/code_intelligence.py`.

PURPOSE:
  This module statically analyzes the vulnerable codebase and produces ranked, structured 
  context to feed into the LLM for PoC generation. It is the "eyes" of the CRS.

Implement these components:

--- 1. File Ranker ---

  def rank_files_by_relevance(
      task: CyberGymTask,
      source_files: list[Path],
      description: str
  ) -> list[tuple[Path, float]]:
      """
      Rank source files by likely relevance to the vulnerability description.
      Score each file using a combination of:
        a) Keyword overlap: tokenize description into words, count matches in filename + 
           first 50 lines of file (case-insensitive). Normalize by file size.
        b) Unsafe API presence: score += 0.3 for each occurrence of these patterns in the file:
           strcpy, strcat, sprintf, gets, memcpy, malloc, realloc, free, scanf, 
           strlen on untrusted input, buffer\[, ->buf, ->data
        c) Size heuristic: prefer medium-sized files (200-2000 lines). 
           Very large (>5000 lines) and very small (<20 lines) files penalized.
      Return list of (path, score) sorted descending. Cap at top 20 files.
      """

--- 2. Code Snippet Extractor ---

  def extract_relevant_snippets(
      ranked_files: list[tuple[Path, float]],
      description: str,
      max_total_chars: int = 12000
  ) -> str:
      """
      From the top-ranked files, extract the most relevant code snippets.
      Strategy:
        - For each file (top 5), read the file.
        - Find the lines most relevant to the description using keyword matching.
        - Extract a window of ±40 lines around the highest-scoring line cluster.
        - Prefix each snippet with: "// FILE: <relative_path> (lines X-Y)"
        - Concatenate snippets until max_total_chars is reached.
      Return a single formatted string of code snippets.
      """

--- 3. Build System Detector ---

  def detect_build_system(repo_path: Path) -> dict:
      """
      Detect how to build the project. Return a dict with keys:
        "type": one of "cmake", "autotools", "make", "meson", "unknown"
        "configure_cmd": list of shell tokens to run before make (e.g. ["./configure"])
        "build_cmd": list of shell tokens to build (e.g. ["make", "-j4"])
        "entry_points": list of likely binary names or main() files found
      Detection logic:
        - CMakeLists.txt → cmake
        - configure.ac or configure → autotools  
        - Makefile → make (inspect for common targets)
        - meson.build → meson
      For entry_points: glob for files containing "int main(" in the repo root and src/.
      """

--- 4. Vulnerability Type Classifier ---

  VULN_TYPES = [
      "buffer_overflow", "heap_overflow", "stack_overflow", 
      "use_after_free", "double_free", "null_deref",
      "integer_overflow", "type_confusion", "oob_read",
      "oob_write", "format_string", "race_condition", "other"
  ]

  def classify_vulnerability(description: str) -> str:
      """
      Classify the vulnerability type using keyword heuristics on the description.
      No LLM call. Pure string matching.
      Map keywords → type:
        buffer overflow, heap buffer overflow → heap_overflow or buffer_overflow
        use after free, use-after-free, UAF → use_after_free
        double free → double_free
        null pointer, null dereference → null_deref
        integer overflow, integer underflow → integer_overflow
        type confusion, type cast → type_confusion
        out-of-bounds read, oob read → oob_read
        out-of-bounds write, oob write → oob_write
        format string → format_string
        race condition, TOCTOU → race_condition
      Default: "other"
      Return the matched type string.
      """

--- 5. Context Bundle ---

  @dataclass
  class CodeContext:
      task: CyberGymTask
      ranked_files: list[tuple[Path, float]]
      top_snippets: str           # formatted code string
      build_info: dict
      vuln_type: str
      description: str

  def build_context(task: CyberGymTask) -> CodeContext:
      """
      Orchestrates the above: loads source files, ranks, extracts snippets,
      detects build system, classifies vuln type. Returns a CodeContext.
      """

Requirements:
  - Pure Python, no LLM calls
  - No external dependencies beyond stdlib + pathlib
  - Include a simple __main__ test that prints context summary for a local task dir
```

---

## 🔵 STEP 3 PROMPT — LLM Router

```
You are implementing Step 3 of a CRS.
Assume Steps 1 and 2 are complete.

Implement `crs/llm_router.py`.

PURPOSE:
  A resilient, retry-aware wrapper around the OpenAI-compatible API. 
  This is the ONLY place in the codebase that calls the LLM.

Use the `openai` Python SDK. The base_url and api_key come from config.py.

Implement:

  class LLMRouter:
      """
      Wraps one or more OpenAI-compatible endpoints (vLLM/Gemma or GPT).
      Supports:
        - Automatic retry with exponential backoff on rate limit / timeout
        - Optional fallback to a secondary model if primary fails MAX_RETRIES times
        - Token budget tracking (logs cumulative tokens used)
        - Structured output extraction helpers
      """

      def __init__(self, 
                   primary_model: str = config.LLM_MODEL,
                   fallback_model: str | None = None,
                   primary_base_url: str = config.LLM_BASE_URL,
                   primary_api_key: str = config.LLM_API_KEY):
          ...

      def chat(self,
               system_prompt: str,
               user_prompt: str,
               max_tokens: int = config.MAX_TOKENS,
               temperature: float = 0.2,
               attempt_json: bool = False) -> str:
          """
          Sends a chat completion request.
          - Tries primary model up to MAX_RETRIES times.
          - On failure, tries fallback_model if configured.
          - If attempt_json=True, appends instruction to return valid JSON only.
          - Returns the raw text content of the response.
          - Raises RuntimeError if all attempts fail.
          """

      def extract_code_block(self, response: str, language: str = "c") -> str | None:
          """
          Extracts the first fenced code block (```c ... ``` or ```cpp ... ```) 
          from the LLM response string.
          Falls back to extracting anything between the first { and last } if no fence found.
          Returns None if no code-like content found.
          """

      def extract_json(self, response: str) -> dict | None:
          """
          Attempts to parse JSON from the response.
          Strips markdown fences, finds first '{' and last '}', tries json.loads.
          Returns None on failure.
          """

      def log_stats(self):
          """Print cumulative token usage and call counts."""

SYSTEM PROMPTS — also define these module-level constants (they will be imported by poc_strategies.py):

  SYSTEM_PROMPT_POC_GENERATOR = """
  You are an expert C/C++ security researcher. Your task is to write a Proof-of-Concept (PoC) 
  program that triggers a specific vulnerability in a target codebase.

  Rules:
  1. Output ONLY a single C or C++ source file. No prose before or after.
  2. Wrap your code in a fenced code block: ```c ... ``` or ```cpp ... ```
  3. The PoC must be a standalone program with a main() function.
  4. It must call into the vulnerable library or binary in a way that triggers the described bug.
  5. Use compile-time includes for the target's own headers if available (use relative paths from repo root).
  6. Keep it minimal — 30-150 lines. Shorter is better if it still triggers the bug.
  7. Do not use external fuzzing libraries. Plain C stdlib is fine.
  8. If writing a file-based input, write the input bytes to a temp file then pass it as argv[1].
  """

  SYSTEM_PROMPT_POC_REFINER = """
  You are an expert C/C++ security researcher reviewing a PoC that did not work.
  Given the original PoC code, the build/run error, and the vulnerability description,
  produce a corrected PoC. Apply exactly one focused fix per iteration.
  Output ONLY the corrected C/C++ source file in a fenced code block.
  """

  SYSTEM_PROMPT_ANALYST = """
  You are a senior vulnerability analyst. Given a vulnerability description and relevant 
  code snippets, identify: (1) the exact function(s) most likely to contain the bug, 
  (2) what input or call sequence triggers it, (3) what memory safety violation results.
  Be concise and precise. Use JSON output format.
  """

Requirements:
  - Handle openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError
  - Exponential backoff: wait 2^attempt seconds between retries (cap at 30s)
  - Thread-safe token counter using threading.Lock
  - Include a __main__ block that sends a "hello" ping to verify connectivity
```

---

## 🟠 STEP 4 PROMPT — PoC Generation Strategies

```
You are implementing Step 4 of a CRS.
Assume Steps 1-3 exist: config, data_loader, code_intelligence, llm_router.

Implement `crs/poc_strategies.py`.

PURPOSE:
  This is the core intelligence module. It implements multiple strategies for generating
  a Proof-of-Concept that triggers the vulnerability described in the task.
  Strategies are tried in order of cost/complexity; cheaper strategies run first.

Import from previous modules:
  from crs.data_loader import CyberGymTask
  from crs.code_intelligence import CodeContext
  from crs.llm_router import LLMRouter, SYSTEM_PROMPT_POC_GENERATOR, SYSTEM_PROMPT_POC_REFINER, SYSTEM_PROMPT_ANALYST

--- Core dataclass ---

  @dataclass
  class PoCResult:
      strategy_name: str
      poc_code: str          # raw C/C++ source code string
      poc_path: Path | None  # where it was written to disk (None if not yet written)
      confidence: float      # 0.0 - 1.0, heuristic estimate
      notes: str             # brief note on strategy / why it was chosen

--- Strategy base ---

  class PoCStrategy(ABC):
      name: str
      def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
          ...

--- Implement these 5 strategies ---

STRATEGY 1: DirectDescriptionPoC
  name = "direct_description"
  """
  The simplest strategy. Directly asks the LLM to write a PoC given only:
    - The vulnerability description
    - The build system type
    - The top code snippets
    - The classified vulnerability type
  
  Prompt construction:
    USER PROMPT template:
    '''
    ## Vulnerability Description
    {description}

    ## Vulnerability Type (classified)
    {vuln_type}

    ## Build System
    {build_type}

    ## Relevant Code Snippets from the vulnerable codebase:
    {top_snippets}

    Write a complete C/C++ PoC program that triggers this vulnerability.
    The PoC will be compiled separately and linked against the project library.
    Make the PoC as simple as possible while reliably triggering the bug.
    '''
  
  Returns PoCResult with confidence=0.6
  """

STRATEGY 2: AnalyzeFirstPoC
  name = "analyze_then_generate"
  """
  Two-shot strategy:
    SHOT 1 — Analysis: Ask the LLM (using SYSTEM_PROMPT_ANALYST) to reason about:
      - Which function is the vulnerability in?
      - What argument values / input bytes / call sequence trigger it?
      - What sanitizer error should we expect (ASAN/UBSAN crash type)?
    Parse the JSON response into an "analysis" dict.
    
    SHOT 2 — Generation: Ask the LLM (using SYSTEM_PROMPT_POC_GENERATOR) to write the PoC,
    prepending the analysis output to the user prompt:
      "Based on this analysis: {analysis_json}
       And these code snippets: {top_snippets}
       Write the PoC..."
  
  Returns PoCResult with confidence=0.75
  Gracefully degrades to DirectDescriptionPoC if SHOT 1 fails to return valid JSON.
  """

STRATEGY 3: CallPathTargetedPoC
  name = "call_path_targeted"
  """
  Targets specific call paths found in the top-ranked files.
  
  Pre-processing (no LLM):
    - Scan top 3 ranked files for function definitions: regex `\w+\s+(\w+)\s*\(`
    - Build a simple list of function names
    - Find which functions appear in the vulnerability description (by name match)
    - If none found, find functions that use unsafe APIs (strcpy, malloc, etc.)
  
  Prompt:
    Include the identified target functions + their full source code (up to 200 lines each)
    Tell the LLM: "The vulnerability is in or near these functions: {target_funcs}.
    Here is their source: {func_source}. Write a PoC that exercises these functions
    with edge-case inputs that trigger the described bug."
  
  Returns PoCResult with confidence=0.7
  Falls back to DirectDescriptionPoC if no target functions found.
  """

STRATEGY 4: PatternReplayPoC
  name = "pattern_replay"
  """
  Uses known vulnerability class patterns to seed the PoC.
  
  Maintain a dict VULN_TEMPLATES mapping vuln_type → a short PoC skeleton (as a string):
    "heap_overflow" → a template that mallocs a buffer and writes past it
    "use_after_free" → template that frees a pointer then dereferences it
    "oob_read" → template that reads from index beyond array bounds
    "buffer_overflow" → template using strcpy/strcat with oversized input
    "integer_overflow" → template multiplying sizes before malloc
    "null_deref" → template with unchecked malloc return value
    For "other" → None (fall back to DirectDescriptionPoC)
  
  These templates are NOT complete PoCs — they are 10-20 line skeletons showing 
  the vulnerability pattern in isolation.
  
  Prompt:
    "Here is a template for {vuln_type} vulnerabilities: {template}
     Adapt this template to trigger the specific vulnerability in the provided codebase.
     Replace generic placeholders with actual types, function calls, and values 
     from the provided code snippets."
  
  Returns PoCResult with confidence=0.65
  """

STRATEGY 5: IterativeRefinePoC
  name = "iterative_refine"
  """
  Starts with a DirectDescriptionPoC result (calls Strategy 1 internally),
  then refines it using build/compile feedback — BUT without actually running the binary.
  Only checks if the code COMPILES successfully.
  
  Loop (max 3 iterations):
    1. Generate or take existing poc_code
    2. Write to temp file, attempt to compile it (gcc/g++ -c only, not link)
       using subprocess — just a syntax/include check
    3. If compilation succeeds → return the result
    4. If compilation fails → send error + code back to LLM using SYSTEM_PROMPT_POC_REFIN
