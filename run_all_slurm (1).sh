#!/bin/bash
#SBATCH --job-name=rag_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=10
#SBATCH --time=4-00:00:00
#SBATCH --output=rag_all_%j.log
#
# Everything in one batch job (same as run_all.sh, but it keeps running after you log off).
#
#   cd /exp/FY26/aim/ro31337/AIA
#   sbatch lora_review_agents/run_all_slurm.sh              # the real run: 1000 questions
#   sbatch lora_review_agents/run_all_slurm.sh --limit 5    # five-question test as a job
#
#   squeue -u $USER                 is it running?
#   tail -n 30 rag_all_<jobid>.log  where is it now?
#   scancel <jobid>                 stop it
#
# Submitting it again continues where the last job stopped. Options on the sbatch line override the ones
# above, e.g.  sbatch --exclude=g52lambda08 --time=6-00:00:00 lora_review_agents/run_all_slurm.sh

module load anaconda/anaconda3/2024.2
conda activate base
module load cuda/11.8

WORKDIR="${SLURM_SUBMIT_DIR:-/exp/FY26/aim/ro31337/AIA}"
cd "$WORKDIR" || { echo "cannot cd to $WORKDIR"; exit 1; }
if [ ! -f lora_review_agents/run_all.sh ]; then
  echo "lora_review_agents/run_all.sh not found in $WORKDIR - submit this from /exp/FY26/aim/ro31337/AIA"; exit 1
fi

echo "=== job ${SLURM_JOB_ID:-?} on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

# stop in the first minute if this node cannot reach the runner model (LiteLLM) or lacks the GPU libraries
echo "=== preflight"
if [ "${SKIP_CHECK:-0}" != "1" ] && ! bash lora_review_agents/run_all.sh --check; then
  echo "=== preflight FAILED on $(hostname): see the FAIL lines above. Nothing was run."
  echo "    (to run anyway:  SKIP_CHECK=1 sbatch lora_review_agents/run_all_slurm.sh)"
  exit 1
fi

echo "=== run"
bash lora_review_agents/run_all.sh "$@"
rc=$?
echo "=== job ${SLURM_JOB_ID:-?} finished $(date) with exit code $rc"
exit $rc
