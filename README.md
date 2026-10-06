export AGENT_REASONING_EFFORT=high
nohup python3 campaign.py --work "$PWD" --max-turns 150 --bash-timeout 1800 > campaign.log 2>&1 &
tail -f campaign.log
