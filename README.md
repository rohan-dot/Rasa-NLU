and discuss what it found: the crashes it surfaced, how triage separated confirmed defects from noise, and how LLM-guided seeding changed coverage. We close with lessons for operational deployment of autonomous vulnerability discovery.










nohup python3 campaign_work/autofix.py --work campaign_work --skip-rotate --skip-map > campaign_work/autofix.log 2>&1 &
tail -f campaign_work/autofix.log
