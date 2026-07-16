df = df.dropna(subset=["lat", "lon"])
df = df.drop_duplicates(subset="iso3", keep="first")
df["lat"] = df["lat"].astype(float)
df["lon"] = df["lon"].astype(float)
