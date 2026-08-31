python -c "
import pandas as pd
a = pd.read_csv('MAIN.csv')                     # the one WITHOUT overflight_raw
b = pd.read_csv('SOURCE.csv')[['airport','overflight_raw']]  # the one WITH it
b = b.drop_duplicates('airport')                # one row per airport, avoids row blow-up
m = a.merge(b, on='airport', how='left')
m.to_csv('MAIN_with_overflight.csv', index=False)
print(f'{len(a)} rows in, {len(m)} rows out; overflight_raw filled for {m.overflight_raw.notna().sum()} rows')
"
