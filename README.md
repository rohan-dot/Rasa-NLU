git format-patch 9f5bd21 --stdout > discver-full.patch
ls -lh discver-full.patch



git checkout -b toolcalling-reliability
git am < discver-full.patch

export DISCVER_GUIDED_JSON=1
export DISCVER_SINGLE_ACTION=1
