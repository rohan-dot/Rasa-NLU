#!/bin/bash
module load anaconda/anaconda3/2021.11
module load cuda/11.8
cd /exp/FY26/CAITT/ro31337/FCG/FCG_claude
FCG_CHUNK_SIZE=16000 python local_extract.py \
  --model /panfs/g52-panfs/exp/FY26/models/gemma-4-31B-it \
  --checklist checklist_enriched.json --fcg-dir data/fcg \
  --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json \
  --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv \
  --learnings LEARNINGS.md --out all_countries.csv



  sbatch --gres=gpu:2 --partition=A100-80 --time=2-00:00:00 --exclude=g52lambda05,g52lambda03 --output=gemma_run_%j.log gemma_extract.sh
