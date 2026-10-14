cd /exp/FY26/AIxCC/ro31337
export LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1 LITELLM_API_KEY=<key> AGENT_TLS_VERIFY=false AGENT_MODEL=claude-opus-5-5
nohup python3 campaign_work/campaign.py --work campaign_work > campaign_work/campaign.log 2>&1 &
tail -f campaign_work/campaign.log
