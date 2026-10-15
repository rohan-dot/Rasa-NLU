**Title:** DISCVER: LLM-Guided Fuzzing and Evidence-Backed Triage for Autonomous Vulnerability Discovery

Memory-safety vulnerabilities in widely deployed open-source software remain a persistent risk, yet finding and confirming them still depends on scarce expert effort. We present DISCVER, a Cyber Reasoning System (CRS) that combines large language model (LLM) reasoning, static analysis, and coverage-guided fuzzing to autonomously discover, reproduce, and triage vulnerabilities. DISCVER first analyzes target source to map attack surface, using a swarm of role-specialized LLM agents to flag suspicious code paths. It then drives the project's fuzzing harnesses under sanitizer-instrumented builds, seeding each with LLM-generated inputs: the model specifies a structured, format-aware input space that is sampled locally into hundreds of targeted seeds, steering fuzzing toward flagged code. Every crash is replayed to produce a proof-of-vulnerability (PoV), deduplicated by stack signature, and ranked into confidence tiers, so only reproducible, evidence-backed findings are reported. We evaluate DISCVER on a widely used open-source parsing library from the OSS-Fuzz corpus. [DISCVER reached X% more coverage than fuzzing alone, surfaced N unique crashes, and confirmed M as reproducible memory-safety defects.] We discuss how LLM-guided seeding breaks coverage plateaus, how PoV-backed triage suppresses false positives, and lessons for operational deployment of autonomous vulnerability discovery.

---

That's 1,405 characters with the bracketed placeholder, so you have roughly 95 characters of headroom once you fill in real numbers. I don't have cJSON results yet, so the bracketed sentence needs your actual figures. If you have no crashes on cJSON (it's heavily fuzzed), swap it for coverage gain and number of triaged candidates rather than confirmed bugs. "Parsing library" stays accurate for cJSON without naming it.




git clone --depth 1 https://github.com/OwenSanzas/QuartetFuzz /exp/FY26/AIxCC/ro31337/campaign_work/refs/quartetfuzz







nohup python3 campaign_work/autofix.py --work campaign_work --skip-rotate --skip-map > campaign_work/autofix.log 2>&1 &
tail -f campaign_work/autofix.log
