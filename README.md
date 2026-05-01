edit gemma-fuzzer/oss-crs/docker-compose/run-fuzzer.sh and remove this line:
mkdir -p /var/log/gemma-fuzzer



# Persist agent logs
libCRS register-log-dir /var/log/gemma-fuzzer
