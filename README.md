python country_route_planner.py "KXXX to KYYY" --aircraft C-17 --auto-stops \
    --date 2026-11-15 --risk --alternatives 3

    # TEST 2: same endpoints, each chose its own stops  -> endpoints_compare.png/.csv
python eval_plots.py --reference "KXXX,KAAA,KBBB,KYYY" --endpoints-only \
    --aircraft C-17 --blocks-file last_run_blocks.json

# TEST 1: human's stops fixed, planner only picks corridors between them -> reference_compare.png/.csv
python eval_plots.py --reference "KXXX,KAAA,KBBB,KYYY" --blocks-file last_run_blocks.json

# plain-English readout + Sol narrative -> eval_results/EXPLANATION.txt
python eval_explain.py --llm


python route_judge.py --reference "KXXX,KAAA,KBBB,KYYY" --aircraft C-17 \
    --blocks-file last_run_blocks.json --runs 3
