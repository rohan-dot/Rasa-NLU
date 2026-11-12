# Task 69 — the C/C++ service shape: a project class overriding a library listener whose decoder exists only as a header declaration; `LLVMFuzzerInitialize` for one-time init; generated C/C++ harnesses linked through task 68's step

## Goal
Task 66's fixture matrix covers C/C++ only as raw-bytes shapes (`c-main-file`, `cpp-span-parser`) and as a decoder in
project code (`cpp-nlohmann`). The C++ cousin of `java-listener` / `real-world-1` is missing: a project class derives from
a library's listener type and overrides a virtual that receives an already-decoded payload; the decoder is declared only
in the library's headers (vendored under `third_party/`, `external/`, `deps/`, an `include/<lib>/` without sources, or
`/usr/include`), with no source in the tree. On such a repo today the detector finds no callback, prints `[boundary]
none found` (loud, correct) and fuzzes setter-built stubs — the parser is never reached, exactly the Java failure 66/67
fixed, one language over. Task 67 also left two C/C++ specifics unspecified: where one-time initialisation goes
(`LLVMFuzzerInitialize`, the libFuzzer equivalent of Jazzer's `fuzzerInitialize()` — the hang storm of 2026-11-12 was
init inside the per-input timeout) and how a *generated* C/C++ source becomes a binary (68 links *provided* harness
sources; generated ones must take the same step or they are never run).
Shape, never library name: nothing here may mention DDS, gRPC, UCI or any vendor; the fixtures are invented libraries.

## Precondition — check before editing
`../CAMPAIGN.md` verdicts for 66, 67, 68 (the shell's cwd is the repo root); `src/input_boundary.py` (66), the boundary-harness generation module (67),
`src/local_build.py` and its C/C++ builder (68). If 66 or 67 did not land (file absent / FAIL), do NOT build on nothing:
call `finish()` with no edits and name the missing prerequisite. If 68 did not land, implement everything except the link
step and say so in `finish()`.

## Definitions (additions to 66; C/C++ adapters `src/lang/cpp.py` / `src/lang/c.py` supply the patterns)
- **Callback rule (a) for C++**: a project class publicly derives from a type declared outside the project and overrides
  (`override`, or a matching `virtual` signature) a member whose parameter type is also declared outside the project.
  "Outside the project" = declared in a dependency header root: vendored dirs above, any `include/` root whose headers
  have no corresponding `.cpp/.cc` in the project, include dirs from `compile_commands.json` / CMake
  `target_include_directories` when 68 built the component, system include roots. **C**: a project function passed to a
  library registration call (`<lib>_set_handler(fn)`, `register_*(fn)`, a callback field in a library config struct) whose
  parameter is a library struct — recorded with `confidence: default`.
- **Decoder search for C/C++**: (1) project code reachable from the callback, depth ≤ 3 — unchanged; (2) **declarations
  in dependency headers** (a declaration is enough; no source needed to generate the call): a free function or static
  member shaped `decode|parse|deserialize|unpack|read|from_<fmt>` taking one of `std::string_view`, `const std::string&`,
  `std::span<const std::byte|uint8_t>`, `(const uint8_t*|const char*, size_t)`, `std::istream&`, and producing the
  payload as the return type, `std::optional<P>`, `std::expected<P,…>`, `std::unique_ptr<B>` with B a base of P, or
  `bool|int` with a `P&`/`P*` out-parameter. `template<class T> T decode(...)` matches with `confidence: default` unless a
  concept/`enable_if` bound names a base of P (then `exact`). Instance decoders: record how the instance is obtained
  (default constructor, `static <Codec>& instance()`, a factory in the same header). **Encoder**: the inverse
  (`encode|serialize|pack|write|to_<fmt>`) → `std::string`/`std::vector<std::byte|uint8_t>`/`(uint8_t*, size_t)` out.
- **Transport-to-stub for C++**: the non-project types the enclosing class's constructor takes (by reference, pointer,
  `shared_ptr`). A stub is a subclass overriding every pure virtual with a no-op (nothing that opens a socket, file,
  thread or middleware participant). A transport that is `final`, has a private constructor and no factory, or no virtuals
  → `unconstructible → downstream`, and 67's downstream path applies.

## Fixture matrix — rows added to `tests/fixtures/boundaries/` (tiny invented libraries, headers only, no build)
| fixture | language | callback shape | decoder location | expected record |
|---|---|---|---|---|
| cpp-listener | C++ | `class StatusReader : public lib::MessageListener<lib::StatusMsg> { void onMessage(const lib::StatusMsg&) override; }`, ctor takes `lib::Connection&` | `third_party/lib/{listener.hpp,codec.hpp,connection.hpp}` — declarations only | callback `StatusReader::onMessage`, payload `lib::StatusMsg`, decoder `lib::Codec::decode(std::string_view) -> lib::StatusMsg` (`where: dependency header`, `confidence: exact`, obtain: default ctor), encoder `lib::Codec::encode(const StatusMsg&) -> std::string`, transport `lib::Connection` (pure virtuals `send`, `subscribe`), downstream `Processor::handle(const StatusMsg&)` |
| cpp-listener-multi | C++ | three reader classes over one `lib::Codec`, payloads `StatusMsg`, `ControlMsg`, `HealthMsg` | same | three callbacks grouped on one decoder (×3) — 67 emits ONE harness dispatching on byte 0 |
| cpp-listener-sysinclude | C++ | `cpp-listener` with the headers under an extra include root passed as `--include-dir` (stands in for `/usr/include/<lib>`) | system header | identical record, `where: system header` |
| cpp-outparam | C++ | callback as above; decoder `bool lib::parse(std::span<const std::byte>, lib::StatusMsg&)` | dependency header | decoder with out-param shape; 67 prompt says "no catch — check the return value" |
| c-callback | C | `lib_set_handler(on_packet)`; `void on_packet(const lib_packet_t*)`; `int lib_packet_parse(const uint8_t*, size_t, lib_packet_t*)` in `third_party/lib/lib.h` | dependency header | callback `on_packet`, payload `lib_packet_t`, decoder `lib_packet_parse` (int-with-out-param), no transport, `confidence: default` |
| cpp-not-a-decoder | C++ | `cpp-listener` plus header functions `std::string to_string(const StatusMsg&)` and `void log(std::string_view)` | — | neither is recorded as a decoder (wrong direction / no payload) |
| cpp-project-is-library | C++ | the `lib::` headers have a `src/codec.cpp` in the project | project code | decoder `where: project code`, depth 1 — the dependency rule does not fire for the project's own code |
| real-world-2 (extend) | C++/Rust | `refs/ams-gra-hello-world-sk` components (skipped if absent) | per component | the detector runs on every C++ component without error and prints either records or the 66 warning; no fixed count asserted — record what is found and paste it in the report |

## Change (src/)
1. **Detector (66, `input_boundary.py` + C/C++ adapters).** Rule (a) and the header-declaration decoder search above; a
   header scanner for C/C++ declarations (regex over preprocessed-enough text: strip comments, join continued lines,
   resolve `using`/`typedef` aliases within the same header set; no compiler needed); `--include-dir` on the CLI
   (repeatable). Records carry `where: project code | dependency header <path> | system header <path>`.
2. **Harness skeleton (67, C/C++ adapter).** ALL one-time initialisation — codec/context construction, transport stub
   instance, any `instance()`/factory call — in `extern "C" int LLVMFuzzerInitialize(int*, char***)`; never in
   `LLVMFuzzerTestOneInput`. The per-input function returns 0 below 2 bytes and above a size cap; byte 0 selects the
   payload type when one decoder feeds several callbacks; text decoders get a `std::string_view` of the rest, binary ones
   a `std::span<const std::byte>` / pointer+size. Catch only what the decoder/callback declares or documents; out-param
   decoders → no catch, return 0 on `false`. C harness: same shape in C, `LLVMFuzzerInitialize` emitted only when the
   record has an init step. The prompt states the rule in words ("one-time initialisation belongs in LLVMFuzzerInitialize;
   it is untimed — the per-input function is not") the way 62 states it for `fuzzerInitialize()`.
3. **Acceptor (67).** Add to the catch-all list: `catch (...)` and `catch (const std::exception&)` around the decode.
   Add a check: the decoder's context/codec/stub constructed inside `LLVMFuzzerTestOneInput` →
   `harness-gen: <name>: one-time init must be in LLVMFuzzerInitialize — regenerating` once, then rejected as in 67.
4. **Build (68).** A generated C/C++ harness source is compiled and linked by 68's provided-harness link step for its
   component — same flags, same built libs, `-fsanitize=fuzzer,address`, include dirs from the component build
   (`compile_commands.json` when present, else the header roots 66 found) → `$OUT/<component>/<name>`. Expose that step
   as a function (`local_build.link_harness(component, source, name)`) if 68 wrote it inline. Component not built →
   `harness-gen: <name>: component not built — skipped` (never "fuzzed anyway"); one repair pass via 38 on a compile error.
5. **Seeds (67).** When an encoder is declared: a tiny seed writer in the same language, compiled with the same flags,
   builds a few payloads (aggregate init / public setters) and calls the encoder → `<Harness>_seed_corpus/NN` (byte 0 =
   type index) + `<Harness>.dict`; no encoder → `no encoder — no seeds`, continue.
6. Safety: wrapped per piece; `[cpp_boundary] failed: <error>` → 66's warning path; `[cpp_harness] failed: <error>` →
   today's generation for that candidate.

## Tests (tests/test_cpp_service_shape.py) — fakes only: no compiler, fake LLM, fake 68 build, fake compiler invocation
- every fixture row above: the 66 CLI on the fixture dir yields exactly the expected record (callback, payload, decoder,
  where, confidence, obtain, encoder, transport with the pure virtuals named, downstream); `cpp-listener-multi` grouped ×3;
  `cpp-listener-sysinclude` via `--include-dir`.
- header scanner: comments stripped, continued declarations joined, `using Msg = lib::StatusMsg;` resolved; a declaration
  inside `#if 0` ignored.
- prompt for `cpp-listener`: names `LLVMFuzzerInitialize`, the decoder declaration verbatim, the obtain path, the stub rule
  with `send` and `subscribe` by name, the exceptions to catch; `cpp-listener-multi` → the byte-0 dispatch instruction;
  `cpp-outparam` → "no catch — check the return value"; `c-callback` → a C skeleton, the out-param shape, no `extern "C"`.
- acceptor: reference-shaped harness (init in `LLVMFuzzerInitialize`, calls `lib::Codec::decode` and `onMessage`) →
  accepted; `catch (...)` → rejected; codec constructed per input → the regenerating line once, second miss → rejected,
  no fallback harness registered.
- link step: fake built component (libs + include dirs) → fake compiler called with `-fsanitize=fuzzer,address`, the include
  dirs, the libs, output `$OUT/<component>/<name>`; component not built → the skipped line, compiler and fuzzer never
  invoked.
- golden: the `java-listener` prompt, the Go/Rust/Python fixture prompts and the non-boundary X509 prompt are
  byte-identical to 67's golden files; `tests/test_input_boundary.py` and `tests/test_boundary_harness_generation.py`
  still pass unchanged.

## Acceptance
```
python -m pytest -q tests/test_cpp_service_shape.py tests/test_input_boundary.py tests/test_boundary_harness_generation.py
python3 src/input_boundary.py --root tests/fixtures/boundaries/cpp-listener    # one record, decoder from the header — paste in the report
```

## Do not touch
Gate (16) and static-check (17) logic; sidecar; crash_dedup._get_crash_signature; the Java/Go/Rust/Python records and
prompts (golden); the non-boundary prompt text; 68's build detection; anything outside src/ and tests/.
