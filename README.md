cd /exp/FY26/AIxCC/ro31337/discver_campaign/discver-campaign
mkdir -p refs runs
mv buttercup refs/buttercup
# keep only Atlantis's CRS code as the reference (the rest is k8s/infra noise that slows every grep)
if [ -d atlantis/example-crs-webservice ]; then mv atlantis ../atlantis-full && mv ../atlantis-full/example-crs-webservice refs/atlantis; else mv atlantis refs/atlantis; fi
mv discver-patchengineer02 discver
cp -r /exp/FY26/AIxCC/ro31337/chatgpt/discver results     # the latest run output: orchestrator.log, patches/, report_run_*.md — use a newer one if you have it
git clone --depth 1 https://github.com/o2lab/afc-crs-all-you-need-is-a-fuzzing-brain refs/fuzzing-brain
git clone --depth 1 --filter=blob:none --sparse https://github.com/CodeIntelligenceTesting/jazzer refs/jazzer && (cd refs/jazzer && git sparse-checkout set docs README.md)
ls refs; ls discver/src | head -5; ls results | head -5
