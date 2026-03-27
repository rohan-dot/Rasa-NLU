Hey Ashok — re: the build failures on 57002 and the other tasks.

The pipeline is triggering vulns fine (PoCs compile standalone and run), but the project build itself keeps failing because cmake can't find required dev libraries (openssl, cjson, etc.). That means we're not getting sanitizer-linked builds, so crash reports are less detailed (signal_6 instead of heap-buffer-overflow).

I updated build_executor.py to automatically retry cmake with features disabled (e.g. -DWITH_TLS=OFF), but some deps are non-optional.

Fix: run this once on the machine:

sudo apt-get install -y build-essential cmake pkg-config libssl-dev libcjson-dev zlib1g-dev libxml2-dev libcurl4-openssl-dev libsqlite3-dev libyaml-dev libexpat1-dev autoconf automake libtool

Or set AUTO_INSTALL_DEPS = True in config.py to let the pipeline handle it automatically (needs sudo).

Once those libs are there, the project should build with sanitizers and we'll get proper ASAN crash types instead of bare signal_6
