import csv

import lm_struct as st

def load_csv(path):
    points = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:
            p = st.TLightning(
                ts=float(row["ts"],
                lat=float(row["lat"]),
                lon=float(row["lng"]))
            )

            points.append(p)

    return points
    