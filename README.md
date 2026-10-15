nohup python3 campaign_work/autofix.py --work campaign_work --skip-rotate --skip-map > campaign_work/autofix.log 2>&1 &
tail -f campaign_work/autofix.log
