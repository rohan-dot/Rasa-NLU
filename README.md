pip install --user playwright openai httpx
playwright install chromium


export SITE_URL="https://<the site's login page URL>"
export SITE_USER="<your username>"
export SITE_PASS="<your password>"
python site_recon_agent.py --origin KSUU --dest FKKD --stops TJSJ,SBFZ --aircraft C-17 --date 2026-10-20 --headless --allow-calculate --confirm
