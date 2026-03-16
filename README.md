gcc -fsanitize=address,undefined -g -O1 \
  /home/ro31337/.crs_workdir/57002/poc_iterative_refine.c \
  /home/ro31337/.crs_workdir/57002/repo/src-vul/mosquitto/src/retain.c \
  -o /tmp/poc_real_test \
  -I/home/ro31337/.crs_workdir/57002/repo/src-vul/mosquitto \
  -I/home/ro31337/.crs_workdir/57002/repo/src-vul/mosquitto/src \
  -lm -lpthread 2>&1 | head -30
