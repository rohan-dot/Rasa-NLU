cd /exp/FY26/AIxCC/ro31337/discver-patchengineer01
rm -rf send-ashok discver-patch-fix.zip && mkdir -p send-ashok && for f in $(git diff --name-only HEAD~3 -- 'src/*.py' 'tests/*.py'); do mkdir -p "send-ashok/$(dirname "$f")"; cp "$f" "send-ashok/$f"; done && (cd send-ashok && zip -r ../discver-patch-fix.zip .) && unzip -l discver-patch-fix.zip
