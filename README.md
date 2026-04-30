ls -la output/xml_fuzzer_vuln/povs/
cat output/xml_fuzzer_vuln/bugs/verified-*.json


for poc in output/xml_fuzzer_vuln/povs/exploit-*; do echo "=== $poc ===" && output/xml_fuzzer_vuln/generated_harnesses/exploit_FREE_AND_NULL_* "$poc" 2>&1 | head -5; done
