gcc -fsanitize=address,undefined -g -O1 \
  /home/ro31337/.crs_workdir/57002/poc_call_path_targeted.c \
  -o /tmp/poc_test \
  -I/home/ro31337/.crs_workdir/57002/repo/src-vul/mosquitto/src \
  -I/home/ro31337/.crs_workdir/57002/repo/src-vul/mosquitto/lib \
  -I/home/ro31337/.crs_workdir/57002/repo/src-vul/mosquitto \
  $(find /home/ro31337/.crs_workdir/57002/repo/src-vul -name "*.c" | grep -v test | grep -v main | head -20) \
  -lm -lpthread 2>&1 | tail -20
