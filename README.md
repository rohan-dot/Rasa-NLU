cd /exp/FY26/AIxCC/ro31337/discver_campaign/discver-campaign
export LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1 LITELLM_API_KEY=<key> AGENT_MODEL=claude-opus-4-8 AGENT_TLS_VERIFY=false
python3 microagent.py --verify
(cd discver && python -m pytest -q tests 2>&1 | tail -3)
nohup python3 campaign.py --work "$PWD" > campaign.log 2>&1 &
tail -f campaign.log
