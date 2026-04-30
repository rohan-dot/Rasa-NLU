ls output/xml_fuzzer_vuln/generated_harnesses/exploit_*


ls output/xml_fuzzer_vuln/povs/


for poc in output/xml_fuzzer_vuln/povs/exploit-crash-*; do echo "=== $poc ===" && $(ls output/xml_fuzzer_vuln/generated_harnesses/exploit_FREE_AND_NULL_* | head -1) "$poc" 2>&1 | tail -10; done


ls -la output/xml_fuzzer_vuln/povs/
cat output/xml_fuzzer_vuln/bugs/verified-*.json


for poc in output/xml_fuzzer_vuln/povs/exploit-*; do echo "=== $poc ===" && output/xml_fuzzer_vuln/generated_harnesses/exploit_FREE_AND_NULL_* "$poc" 2>&1 | head -5; done
