nohup python3 "$BASE/microagent_team.py" \
  --repo "$BASE/discver-patchengineer02" --results "$BASE/discver" \
  --reference "discver-playbook=$BASE/discver-playbook" \
  --reference "buttercup-fuzzer=$BASE/buttercup/fuzzer" \
  --reference "buttercup-seedgen=$BASE/buttercup/seed-gen" \
  --reference "buttercup-patcher=$BASE/buttercup/patcher" \
  --reference "atlantis=$BASE/atlantis" \
  --rounds 5 > "$BASE/run.log" 2>&1 &
tail -f "$BASE/run.log"
