"""
Genres described in Feb 2021 and Oct 2026: the 2021 notebook's view (LetterBoxd-EDA.ipynb: rating and viewers by
genre) carried forward to 2026 and split into what the ratings did and what the rule did (run 06 first).

Genres are the film's 2021 genres, several per film: a film counts in every genre it has, as in the 2021 notebook.
Means are over films (each film counts once). Audience size is the rating count (the 2021 "viewers" field is not
in the panel).

  G1 by genre: films; mean and median displayed score and raw mean in 2021 and 2026, and each one's rank among
     the 19 genres; median and total ratings in each year and median growth; the mean change split into raw-mean
     change (M), change in the old-rule adjustment (G) and the 2026 rule contrast (S); 2026 displayed - raw.
     (T59: films with >= 200 ratings in 2021; T59b: all films with a 2026 score, whose 2021 raw means and M/G
     depend on the Feb-2021 prior mean, see T55b)
  G2 the reshuffle of the genre ranking by mean score: 2021 displayed -> 2026 under the old rule (the ratings'
     part: M + G) -> 2026 displayed (the rule's part: S), with rank moves and Spearman correlations. (T60)

  .venv/bin/python 13_genre_description.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lbx

p = lbx.load_panel()
dec = pd.read_csv(lbx.DATA / "derived" / "panel_decomposition.csv",
                  usecols=["slug", "rating_2021", "rating_count_2021", "raw21", "raw", "n_hist", "shown", "old26",
                           "d_shown", "d_raw", "d_decay", "d_switch"])
d = p[p.clean][["slug", "genre_list"]].merge(dec, on="slug")
d = d[d.shown.notna()].copy()                              # clean panel films with a 2026 score
d["gap26"] = d.shown - d.raw
GENRES = sorted({g for gs in d.genre_list for g in gs})
long = d.explode("genre_list").rename(columns={"genre_list": "genre"}).dropna(subset=["genre"])


def describe(s):
    """One row per genre; ranks are among genres, 1 = highest mean."""
    g = s.groupby("genre")
    t = pd.DataFrame({
        "films": g.size(),
        "displayed 2021: mean": g.rating_2021.mean(), "displayed 2021: median": g.rating_2021.median(),
        "displayed 2026: mean": g.shown.mean(), "displayed 2026: median": g.shown.median(),
        "raw 2021: mean": g.raw21.mean(), "raw 2021: median": g.raw21.median(),
        "raw 2026: mean": g.raw.mean(), "raw 2026: median": g.raw.median(),
        "ratings per film 2021: median": g.rating_count_2021.median(),
        "ratings per film 2026: median": g.n_hist.median(),
        "growth (2026 / 2021 ratings): median": g.apply(lambda x: (x.n_hist / x.rating_count_2021).median()),
        "total ratings 2026 (millions)": g.n_hist.sum() / 1e6,
        "displayed change": g.d_shown.mean(), "M raw-mean change": g.d_raw.mean(),
        "G old-rule adjustment change": g.d_decay.mean(), "S 2026 rule contrast": g.d_switch.mean(),
        "2026 displayed - raw": g.gap26.mean()})
    for col in ["displayed 2021", "displayed 2026", "raw 2021", "raw 2026"]:
        t[f"rank {col}"] = t[f"{col}: mean"].rank(ascending=False, method="min").astype(int)
    t["rank ratings per film 2026"] = t["ratings per film 2026: median"].rank(ascending=False, method="min").astype(int)
    allrow = {"films": len(s.slug.unique())}
    u = s.drop_duplicates("slug")
    allrow.update({"displayed 2021: mean": u.rating_2021.mean(), "displayed 2021: median": u.rating_2021.median(),
                   "displayed 2026: mean": u.shown.mean(), "displayed 2026: median": u.shown.median(),
                   "raw 2021: mean": u.raw21.mean(), "raw 2021: median": u.raw21.median(),
                   "raw 2026: mean": u.raw.mean(), "raw 2026: median": u.raw.median(),
                   "ratings per film 2021: median": u.rating_count_2021.median(),
                   "ratings per film 2026: median": u.n_hist.median(),
                   "growth (2026 / 2021 ratings): median": (u.n_hist / u.rating_count_2021).median(),
                   "total ratings 2026 (millions)": u.n_hist.sum() / 1e6, "displayed change": u.d_shown.mean(),
                   "M raw-mean change": u.d_raw.mean(), "G old-rule adjustment change": u.d_decay.mean(),
                   "S 2026 rule contrast": u.d_switch.mean(), "2026 displayed - raw": u.gap26.mean()})
    t = t.sort_values("rank displayed 2026")
    t.loc["all films with a 2021 genre (each once)"] = pd.Series(allrow)
    return t


big = long[long.rating_count_2021 >= 200]
t59 = describe(big)
t59b = describe(long)
lbx.write_table(t59, "T59_genres_2021_vs_2026_n200", floatfmt=".3f")
lbx.write_table(t59b, "T59b_genres_2021_vs_2026_all_scored", floatfmt=".3f")

# ----------------------------------------------------------------------------- G2 the reshuffle: ratings vs rule
rows, corr = [], {}
for lab, s in [(">= 200 ratings in 2021", big), ("all with a 2026 score", long)]:
    g = s.groupby("genre")
    m = pd.DataFrame({"films": g.size(), "displayed 2021": g.rating_2021.mean(), "old rule 2026": g.old26.mean(),
                      "displayed 2026": g.shown.mean()})
    r = m[["displayed 2021", "old rule 2026", "displayed 2026"]].rank(ascending=False, method="min").astype(int)
    for gen in m.index:
        rows.append({"sample": lab, "genre": gen, "films": m.films[gen],
                     "mean displayed 2021": m.loc[gen, "displayed 2021"],
                     "mean old rule 2026": m.loc[gen, "old rule 2026"],
                     "mean displayed 2026": m.loc[gen, "displayed 2026"],
                     "ratings part (M + G)": m.loc[gen, "old rule 2026"] - m.loc[gen, "displayed 2021"],
                     "rule part (S)": m.loc[gen, "displayed 2026"] - m.loc[gen, "old rule 2026"],
                     "rank 2021": r.loc[gen, "displayed 2021"], "rank 2026, old rule": r.loc[gen, "old rule 2026"],
                     "rank 2026": r.loc[gen, "displayed 2026"],
                     "rank move from ratings (+ = up)": r.loc[gen, "displayed 2021"] - r.loc[gen, "old rule 2026"],
                     "rank move from rule (+ = up)": r.loc[gen, "old rule 2026"] - r.loc[gen, "displayed 2026"]})
    corr[lab] = {"Spearman 2021 vs 2026 displayed": spearmanr(m["displayed 2021"], m["displayed 2026"])[0],
                 "Spearman 2021 vs 2026 old rule": spearmanr(m["displayed 2021"], m["old rule 2026"])[0],
                 "Spearman 2026 old rule vs 2026 displayed": spearmanr(m["old rule 2026"], m["displayed 2026"])[0],
                 "genres changing rank (2021 -> 2026)": int((r["displayed 2021"] != r["displayed 2026"]).sum()),
                 "sum of |rank moves|, ratings": int((r["displayed 2021"] - r["old rule 2026"]).abs().sum()),
                 "sum of |rank moves|, rule": int((r["old rule 2026"] - r["displayed 2026"]).abs().sum())}
t60 = pd.DataFrame(rows).set_index(["sample", "genre"]).sort_values(["sample", "rank 2026"])
lbx.write_table(t60, "T60_genre_ranking_reshuffle", floatfmt=".3f")
t60b = pd.DataFrame(corr).T
lbx.write_table(t60b, "T60b_genre_ranking_correlations", floatfmt=".3f")
print(t59.round(3).to_string())
print(t60.round(3).to_string())
print(t60b.round(3).to_string())
