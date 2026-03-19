The backslash line continuation is causing issues in the Jupyter terminal. Just put it all on one line:

```bash
python main.py --phase B --test BioASQ-task14bPhaseB-testset1.json --train trainining14b.json
```

The first attempt also had `B/` with a trailing slash — argparse only accepts exactly `A+` or `B`.
