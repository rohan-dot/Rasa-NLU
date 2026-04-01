./configure CFLAGS="-fsanitize=address,undefined -g" \
            LDFLAGS="-fsanitize=address,undefined" \
            --without-xml \
            --without-ftp

make -j4
