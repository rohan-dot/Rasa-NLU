#!/bin/bash
#SBATCH --job-name=discver-campaign
#SBATCH --partition=A100-80
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=g52lambda05,g52lambda03
#SBATCH --output=runs/slurm-%j.out
#SBATCH --signal=B:TERM@300
#
# campaign_job.sh — runs (or continues) the discver campaign as its own Slurm job, so it does not depend on your PC,
# Jupyter or SSH session. Submit from campaign_work:
#     sbatch campaign_job.sh                          continue the latest run
#     sbatch campaign_job.sh --tasks tasks_handoff    start a new run (any campaign.py arguments)
# Watch:  squeue -u $USER      tail -f runs/slurm-<jobid>.out      python3 campaign.py --work . --status
# Stop:   scancel <jobid>      (the running task is put back; submitting again continues where it stopped)
# Needs ~/campaign.env with the LiteLLM settings (bash keepalive.sh writes it).
set -u
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
echo "$(date '+%F %T') job ${SLURM_JOB_ID:-local} on $(hostname) in $(pwd)"

if [ ! -f "$HOME/campaign.env" ]; then
    echo "missing ~/campaign.env — run 'bash keepalive.sh' once (it saves the key), then submit again"; exit 1
fi
# shellcheck disable=SC1090
. "$HOME/campaign.env"
# the same Python setup as other jobs on this cluster (skipped where there are no modules); no CUDA needed
if command -v module >/dev/null 2>&1; then
    module load anaconda/anaconda3/2024.2 >/dev/null 2>&1 && conda activate base >/dev/null 2>&1 || true
fi
PY="${DISCVER_PY:-python3}"
echo "python: $PY ($("$PY" --version 2>&1))"

# never two campaigns on the same tree
if [ -f runs/campaign.pid ]; then
    read -r pid host < runs/campaign.pid || true
    if [ "${host:-}" = "$(uname -n)" ] && tr '\0' ' ' < "/proc/${pid:-0}/cmdline" 2>/dev/null | grep -q campaign.py; then
        echo "a campaign is already running here (pid $pid) — stop it first (kill $pid), then submit again"; exit 1
    fi
    if [ -n "${host:-}" ] && [ "$host" != "$(uname -n)" ] && \
       [ -n "$(find runs \( -path 'runs/2*' -o -name '*.out' \) -type f ! -name "slurm-${SLURM_JOB_ID:-x}.out" -mmin -15 2>/dev/null | head -1)" ]; then
        echo "a campaign on $host wrote to runs/ in the last 15 minutes — stop it there first, then submit again"; exit 1
    fi
fi

"$PY" microagent.py --verify || { echo "gateway check failed on $(hostname) — submit with: sbatch --nodelist=g52lambda02 campaign_job.sh"; exit 1; }

# with arguments: a new run, e.g.  sbatch campaign_job.sh --tasks tasks_handoff   or   ... --tasks tasks_handoff --only 101,102
if [ $# -gt 0 ]; then
    exec "$PY" campaign.py --work . "$@"
fi
# without: continue the latest run
case "$("$PY" campaign.py --work . --status 2>/dev/null | head -1)" in
    *": finished"*) echo "the last campaign run has finished — nothing to do"; exit 0 ;;
    "no campaign state"*|"") echo "no campaign started in this folder — submit with: sbatch campaign_job.sh --tasks tasks_handoff"; exit 1 ;;
esac
exec "$PY" campaign.py --work . --resume
