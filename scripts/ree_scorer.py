"""Two-stage REE convergence scorer (loader-derived, merge-asserted)."""
import numpy as np
import pandas as pd

from elemental_society_analysis.utils.config import load_config
from elemental_society_analysis.geochem.loader import load_dataframe

CHOND = {"Eu": 0.0563, "Sm": 0.148, "Gd": 0.199}
E = "La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Y".split()
REE = [f"{e}_ppm" for e in E]


def main():
    df = load_dataframe(load_config().io)
    core = df.dropna(subset=REE).copy()
    core["SREE"] = core[REE].sum(axis=1)
    core["HREE_frac"] = core[[f"{e}_ppm" for e in E[6:]]].sum(axis=1) / core["SREE"]
    core["Eu_Eu*"] = (core["Eu_ppm"] / CHOND["Eu"]) / np.sqrt(
        (core["Sm_ppm"] / CHOND["Sm"]) * (core["Gd_ppm"] / CHOND["Gd"]))

    lab = pd.read_csv("reports/anomaly_labels.csv").dropna(subset=["ID"])
    m = lab.merge(core[["ID", "Name of Area", "SREE", "HREE_frac", "Eu_Eu*"]],
                  on="ID", how="inner")
    assert len(m) == len(lab) == 28, f"join mismatch: labels {len(lab)}, merged {len(m)}"

    m["population"] = np.where(m["ID"].str.startswith(("230621", "240621")),
                               "sinter", "rhyolite")
    print("--- population x flag ---")
    print(pd.crosstab(m["population"], m["anomaly_label"]))

    rh = m[m["population"] == "rhyolite"].sort_values("HREE_frac", ascending=False)
    print("\n--- rhyolites by HREE_frac (the ion-adsorption play) ---")
    print(rh[["ID", "Name of Area", "HREE_frac", "SREE", "Eu_Eu*",
              "anomaly_label", "anomaly_score"]].head(8).round(3).to_string(index=False))

    m["hree_rank"] = m["HREE_frac"].rank(ascending=False).astype(int)
    print("\n--- flagged samples (rank 1 = most HREE-rich of 28) ---")
    print(m[m["anomaly_label"] == -1][
        ["ID", "population", "hree_rank", "HREE_frac", "SREE", "anomaly_score"]
    ].round(3).to_string(index=False))

    m.sort_values("HREE_frac", ascending=False).to_csv(
        "reports/ree_convergence.csv", index=False)
    print("\nwrote reports/ree_convergence.csv")


if __name__ == "__main__":
    main()
