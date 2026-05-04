git clone https://github.com/DaveGamble/cJSON.git cJSON-vuln

cd cJSON-vuln && git checkout d6d5449~1 && cd ..
chmod +x run_standalone.sh && ./run_standalone.sh auto --src-dir /panfs/g52-panfs/exp/FY26/aim/ro31337/AIxCC/cJSON-vuln --output-dir /panfs/g52-panfs/exp/FY26/aim/ro31337/AIxCC/results-cjson --timeout 600 --vllm-model gemma-4-31b-it
