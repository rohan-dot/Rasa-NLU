cd /exp/FY26/AIxCC/ro31337
bash discver-campaign-v4/setup.sh  campaign_work  discver_2/discver_code  discver_2/results
cd campaign_work/discver && python -m pytest -q tests | tail -8
