
nohup python3 campaign_work/campaign.py --work campaign_work > campaign_work/campaign.log 2>&1 &
tail -f campaign_work/campaign.log
