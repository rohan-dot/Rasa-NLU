#!/bin/bash
#SBATCH --job-name=discver-campaign
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=runs/slurm-%j.out
#SBATCH --signal=B:TERM@300
#
# Runs the discver campaign as a Slurm job, so it keeps going when you disconnect or switch off your PC.
# From campaign_work, in a terminal where the LITELLM_* / AGENT_* variables are exported (sbatch copies them into the job):
#     sbatch campaign_job.sh                          continue the latest run
#     sbatch campaign_job.sh --tasks tasks_handoff    start a new run (any campaign.py arguments)
# Watch: squeue -u $USER ; tail -f runs/slurm-<jobid>.out      Stop: scancel <jobid> (the running task is put back)
set -u
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
echo "$(date '+%F %T') job ${SLURM_JOB_ID:-local} on $(hostname)"

module load anaconda/anaconda3/2024.2 >/dev/null 2>&1 && conda activate base >/dev/null 2>&1

if [ -z "${LITELLM_API_KEY:-}" ] && [ -f "$HOME/campaign.env" ]; then
    # shellcheck disable=SC1090
    . "$HOME/campaign.env"
fi
if [ -z "${LITELLM_API_KEY:-}" ] || [ -z "${AGENT_MODEL:-}" ]; then
    echo "LITELLM_API_KEY / AGENT_MODEL are not set — export them in the terminal, then run sbatch again"; exit 1
fi

# never two campaigns on the same tree
if [ -f runs/campaign.pid ]; then
    read -r pid host < runs/campaign.pid || true
    if [ "${host:-}" = "$(uname -n)" ] && tr '\0' ' ' < "/proc/${pid:-0}/cmdline" 2>/dev/null | grep -q campaign.py; then
        echo "a campaign is already running on this node (pid $pid) — stop it first (kill $pid)"; exit 1
    fi
    if [ -n "${host:-}" ] && [ "$host" != "$(uname -n)" ] && \
       [ -n "$(find runs \( -path 'runs/2*' -o -name '*.out' \) -type f ! -name "slurm-${SLURM_JOB_ID:-x}.out" -mmin -15 2>/dev/null | head -1)" ]; then
        echo "a campaign on $host wrote to runs/ in the last 15 minutes — stop it there first"; exit 1
    fi
fi

python3 microagent.py --verify || { echo "gateway check failed on $(hostname): wrong key/settings, or this node cannot reach LiteLLM"; exit 1; }
[ $# -gt 0 ] && exec python3 campaign.py --work . "$@"
exec python3 campaign.py --work . --resume
