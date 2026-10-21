"""
Two exploratory tests (run 06 first).

  F1 feedback (anchoring): did films the 2023 rule lifted then attract higher ratings from members? Archive films with
     a pre-change histogram (B), a first post-change histogram (A) and the Oct-2026 histogram (Z).
       treatment  boost at A = displayed - old formula applied to A's histogram (what the new rule added, visible to
                  members), and its hidden part = residual of displayed - raw after flexible controls for A's histogram
       outcome    post: mean of the ratings added A -> 2026, minus A's raw mean (how new raters rated vs the stock)
                  pre (placebo): mean of the ratings added Feb 2021 -> B, minus the 2021 raw mean. The boost did not
                  exist yet, so any association there is audience composition, not feedback.
     Feedback = post coefficient - pre coefficient (stacked, period-interacted controls, director-clustered). (T50)
  F2 canon: Sight and Sound 2022 critics' / directors' poll films vs other films with >= 200 ratings in 2021:
     members' change, formula switch, the hidden adjustment given the 2026 histogram, and audience growth. (T51)

  .venv/bin/python 10_feedback_canon.py
"""
import sqlite3

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import lbx

STARS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
HC = [f"h{int(s * 10):02d}" for s in STARS]
t29 = pd.read_csv(lbx.TAB / "T29_old_formula_fit.csv", index_col=0)
C0 = t29.loc["fitted C (prior weight, in ratings)", "readings"]
M0 = t29.loc["fitted m (prior mean, stars)", "readings"]

p = lbx.load_panel()
clean = p[p.clean].copy()
clean["cluster"] = clean.director1.fillna("film:" + clean.slug)
GEN = sorted({g for gs in clean.genre_list for g in gs})
GCOL = {g: "g_" + g.replace(" ", "_") for g in GEN}
for g in GEN:
    clean[GCOL[g]] = clean.genre_list.map(lambda gs, g=g: g in gs).astype(int)
clean["eng"] = clean.english.astype(int)
clean["coh"] = clean.cohort.astype(str)
GT = " + ".join(GCOL.values())


def hist(df):
    m = df[HC].fillna(0).to_numpy(float)
    return pd.DataFrame(m, index=df.slug.to_numpy(), columns=HC)


def clustered(formula, df):
    return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.cluster)[0]})


ar = pd.read_sql("SELECT * FROM readings WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'archive_hist_pilot.sqlite'}?mode=ro", uri=True))
ar["t"] = pd.to_datetime(ar.served.str[:8])
live = pd.read_sql("SELECT * FROM hist WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'histograms.sqlite'}?mode=ro", uri=True))
Bdf = ar[ar.t < lbx.CHANGE].sort_values("t").groupby("slug").tail(1)
Adf = ar[ar.t >= lbx.CHANGE].sort_values("t").groupby("slug").head(1)
B, A, Z = hist(Bdf), hist(Adf), hist(live)
dispA = Adf.set_index("slug").displayed if "displayed" in Adf else Adf.set_index("slug").shown

# ----------------------------------------------------------------------------- F1 feedback
ids = sorted(set(B.index) & set(A.index) & set(Z.index) & set(clean.slug))
B, A, Z = B.loc[ids], A.loc[ids], Z.loc[ids]
nB, nA, nZ = B.sum(axis=1), A.sum(axis=1), Z.sum(axis=1)
rawA = (A * STARS).sum(axis=1) / nA
f = clean.set_index("slug").loc[ids].copy()
f["rawA"], f["nA"] = rawA, nA
f["sdA"] = np.sqrt(((A.to_numpy() * (STARS - rawA.to_numpy()[:, None]) ** 2).sum(axis=1)) / nA)
f["s5A"], f["sloA"] = A.h50 / nA, (A.h05 + A.h10) / nA
f["dispA"] = dispA.reindex(ids)
f["boost"] = f.dispA - (nA * rawA + C0 * M0) / (nA + C0)
f["lvlA"], f["lnA"] = f.rawA - 3.5, np.log10(f.nA)
f["raw21"] = (f.rating_2021 * (f.rating_count_2021 + C0) - C0 * M0) / f.rating_count_2021
added_post, added_pre = nZ - nA, nB - f.rating_count_2021
f["new_post"] = ((Z - A) * STARS).sum(axis=1) / added_post - f.rawA
f["new_pre"] = (((B * STARS).sum(axis=1)) - f.raw21 * f.rating_count_2021) / added_pre - f.raw21
f["added_post"], f["added_pre"] = added_post, added_pre
f = f[(f.added_post >= 200) & (f.added_pre >= 200) & f.dispA.notna() & (f.rating_count_2021 >= 1000)].reset_index()
FLEXA = "bs(lvlA, df=5) + bs(lnA, df=5) + lvlA:lnA + sdA + s5A + sloA"
f["hidden_boost"] = smf.ols(f"I(dispA - rawA) ~ {FLEXA}", f).fit().resid

# long format: one row per film and period; period-specific starting level and count
post = f.assign(period=1, y=f.new_post, lvl0=f.rawA - 3.5, ln0=np.log10(f.nA))
pre = f.assign(period=0, y=f.new_pre, lvl0=f.raw21 - 3.5, ln0=np.log10(f.rating_count_2021))
long = pd.concat([pre, post], ignore_index=True)
rows = []
for treat, lab in [("boost", "visible boost (new rule minus old formula, at A)"),
                   ("hidden_boost", "hidden part of the boost (given A's histogram)")]:
    ctl = f"lvl0 + I(lvl0**2) + ln0 + {GT} + eng + C(coh)"
    for per, df in [("post (A -> 2026)", post), ("pre placebo (2021 -> B)", pre)]:
        m = clustered(f"y ~ {treat} + {ctl}", df)
        lo, hi = m.conf_int().loc[treat]
        rows.append({"treatment": lab, "estimate": per, "per +0.1 star": 0.1 * m.params[treat], "ci_lo": 0.1 * lo,
                     "ci_hi": 0.1 * hi, "p": m.pvalues[treat], "films": int(m.nobs)})
    m = clustered(f"y ~ period * ({treat} + {ctl})", long)
    term = f"period:{treat}"
    lo, hi = m.conf_int().loc[term]
    rows.append({"treatment": lab, "estimate": "FEEDBACK = post - pre", "per +0.1 star": 0.1 * m.params[term],
                 "ci_lo": 0.1 * lo, "ci_hi": 0.1 * hi, "p": m.pvalues[term], "films": int(m.nobs / 2)})
t50 = pd.DataFrame(rows).set_index(["treatment", "estimate"])
t50.loc[("sample", "median ratings added after A / 2021->B"), "films"] = len(f)
t50.loc[("sample", "median ratings added after A / 2021->B"), "per +0.1 star"] = f.added_post.median()
t50.loc[("sample", "median ratings added after A / 2021->B"), "ci_lo"] = f.added_pre.median()
t50.loc[("sample", "boost at A: mean / sd (stars)"), "per +0.1 star"] = f.boost.mean()
t50.loc[("sample", "boost at A: mean / sd (stars)"), "ci_lo"] = f.boost.std()
lbx.write_table(t50, "T50_feedback_test", floatfmt=".4f")
print(t50.round(4).to_string())

# ----------------------------------------------------------------------------- F2 canon
canon = lbx.load_canon(p)
cs = set(canon.slug.dropna())
dec = pd.read_csv(lbx.DATA / "derived" / "panel_decomposition.csv", usecols=["slug", "raw21", "d_raw", "d_switch", "d_shown",
                                                                              "raw", "n_hist", "sd", "shown"])
d = clean.merge(dec, on="slug")
d = d[d.shown.notna() & (d.rating_count_2021 >= 200)].copy()
M = live.set_index("slug").reindex(d.slug)[HC].fillna(0).to_numpy(float)
P = M / M.sum(1, keepdims=True)
d["s5"], d["slo"] = P[:, 9], P[:, :2].sum(axis=1)
d["canon"] = d.slug.isin(cs).astype(int)
d["lvl21"], d["lvl"], d["ln"] = d.raw21 - 3.5, d.raw - 3.5, np.log10(d.n_hist)
d["wedge"] = d.shown - d.raw
d["log_growth"] = np.log10(d.n_hist / d.rating_count_2021)
rows = []
base = f"lvl21 + I(lvl21**2) + log10_n21 + {GT} + eng + C(coh)"
for y, lab, rhs in [("d_shown", "displayed change", base), ("d_raw", "members (raw change)", base),
                    ("d_switch", "formula switch", base), ("log_growth", "audience growth, log10", base),
                    ("wedge", "hidden adjustment given the 2026 histogram",
                     f"bs(lvl, df=5) + bs(ln, df=5) + lvl:ln + sd + s5 + slo + {GT} + eng + C(coh)")]:
    for samp, df in [("all >= 200", d), ("released before 1990", d[d.year_2021 < 1990])]:
        m = clustered(f"{y} ~ canon + {rhs}", df)
        lo, hi = m.conf_int().loc["canon"]
        rows.append({"outcome": lab, "sample": samp, "canon vs other": m.params.canon, "ci_lo": lo, "ci_hi": hi,
                     "p": m.pvalues.canon, "canon films": int(df.canon.sum()), "films": int(m.nobs)})
t51 = pd.DataFrame(rows).set_index(["outcome", "sample"])
lbx.write_table(t51, "T51_canon_films", floatfmt=".4f")
print(t51.round(4).to_string())
