curl -sS -o /dev/null -w "%{http_code}\n" "$SITE_URL"
curl -sS "$SITE_URL" | head -40

export SITE_USE_PROXY=true
python site_learner.py --mission "..." --task "..." --explore-pages 40 --rounds 4 --headless

