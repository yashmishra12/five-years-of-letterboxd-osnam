"""
Robustness checks and the panel-representativeness table (run 06 first).

  R1 headline displayed change and its decomposition (members / decay / switch) under alternative samples:
     main (clean matches) | + flagged suspect matches | without TV entries | without 2018-2020 releases. (T52)
  R2 review intensity (pre-change written reviews per rating, archive films) on members' change and the formula
     switch, all films vs without 2018-2020 releases. (T52b; the timing-based version is T19 in 02_jump.py)
  R3 representativeness of the Feb-2021 panel: among films released by 2020 that appear in the 2026 crawl (panel plus
     TMDB top-100k), the share in the clean 2021 panel by audience size, and how well 2026 traits predict being in it
     (logit: pseudo-R2 and AUC). (T53)
  R4 precision of the break date: film-resampling bootstrap (200 draws) of the two-break scan (step date and
     slope-change date both free; monthly step dates 2022-24, quarterly slope-change dates), archive readings 2019+. (T54)

  .venv/bin/python 11_robustness.py
"""
import json
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
live = pd.read_sql("SELECT * FROM hist WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'histograms.sqlite'}?mode=ro", uri=True))
m = live[HC].fillna(0).to_numpy(float)
live = live.assign(n26=m.sum(axis=1), raw26=(m * STARS).sum(axis=1) / m.sum(axis=1))[["slug", "shown", "n26", "raw26"]]

# ----------------------------------------------------------------------------- R1 decomposition under alternative samples
base = p.matched & ~p.dup_slug & ~p.prerelease & ~p.manual_mismatch       # everything except the suspect flag
d = p[base].merge(live, on="slug")
d = d[d.shown.notna()].copy()
d["raw21"] = (d.rating_2021 * (d.rating_count_2021 + C0) - C0 * M0) / d.rating_count_2021
d["old26"] = (d.n26 * d.raw26 + C0 * M0) / (d.n26 + C0)
d["displayed"] = d.shown - d.rating_2021
d["members"] = d.raw26 - d.raw21
d["decay"] = (d.old26 - d.raw26) - (d.rating_2021 - d.raw21)
d["switch"] = d.shown - d.old26
recent = d.year_2021 >= 2018
samples = {"main (clean matches)": d.clean, "+ flagged suspect matches": d.index == d.index,
           "without TV entries": d.clean & ~d.is_tv, "without 2018-2020 releases": d.clean & ~recent}
rows = []
for lab, sel in samples.items():
    for grp, g in [("n2021 >= 200", d.rating_count_2021 >= 200), ("all scored", d.rating_count_2021 > 0)]:
        s = d[sel & g]
        r = {"sample": lab, "group": grp, "films": len(s)}
        for c in ["displayed", "members", "decay", "switch"]:
            ci = lbx.mean_ci(s[c])
            r[c] = ci["mean"]
            if c in ("displayed", "switch"):
                r[f"{c} ci"] = f"[{ci.ci_lo:.3f}, {ci.ci_hi:.3f}]"
        rows.append(r)
t52 = pd.DataFrame(rows).set_index(["group", "sample"]).sort_index()
lbx.write_table(t52, "T52_robustness_decomposition", floatfmt=".4f")
print(t52.round(4).to_string())

# ----------------------------------------------------------------------------- R2 review intensity without recent releases
rv = lbx.archive_tables()[1].query("note == 'ok'")[["slug", "timestamp", "review_count", "rating_count"]]
rv = rv[pd.to_datetime(rv.timestamp.str[:8], errors="coerce") < lbx.CHANGE].sort_values("timestamp")
rv = rv.groupby("slug").tail(1).assign(rr=lambda x: x.review_count / x.rating_count)[["slug", "rr"]]
h = live.merge(pd.read_sql("SELECT * FROM hist WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'histograms.sqlite'}?mode=ro", uri=True))[["slug"] + HC], on="slug")
mm = h[HC].fillna(0).to_numpy(float)
P = mm / mm.sum(axis=1, keepdims=True)
h["sd"] = np.sqrt((P * (STARS - h.raw26.to_numpy()[:, None]) ** 2).sum(axis=1))
h["s5"], h["slo"] = P[:, 9], P[:, :2].sum(axis=1)
r = d[d.clean].merge(rv, on="slug").merge(h[["slug", "sd", "s5", "slo"]], on="slug")
GEN = sorted({g for gs in r.genre_list for g in gs})
for g in GEN:
    r["g_" + g.replace(" ", "_")] = r.genre_list.map(lambda gs, g=g: g in gs).astype(int)
GT = " + ".join("g_" + g.replace(" ", "_") for g in GEN)
r["lvl21"], r["lvl"], r["ln"] = r.raw21 - 3.5, r.raw26 - 3.5, np.log10(r.n26)
r["eng"], r["tv"] = r.english.astype(int), r.is_tv.astype(int)
r["cluster"] = r.director1.fillna("film:" + r.slug)
BASE = f"lvl21 + log10_n21 + eng + tv + {GT}"
HIST = f"bs(lvl, df=5) + bs(ln, df=5) + lvl:ln + sd + s5 + slo + eng + tv + {GT}"
rows = []
for samp, s in [("all archive films", r), ("without 2018-2020 releases", r[r.year_2021 < 2018])]:
    for y, lab, ctl in [("members", "members (n2021 >= 200)", BASE), ("switch", "switch | 2021 baseline", BASE),
                        ("switch", "switch | 2026 histogram", HIST)]:
        f = smf.ols(f"{y} ~ rr + {ctl}", s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.cluster)[0]})
        lo, hi = f.conf_int().loc["rr"]
        rows.append({"sample": samp, "outcome": lab, "per +0.10 review ratio": 0.1 * f.params.rr, "ci_lo": 0.1 * lo,
                     "ci_hi": 0.1 * hi, "p": f.pvalues.rr, "films": int(f.nobs)})
t52b = pd.DataFrame(rows).set_index(["outcome", "sample"]).sort_index()
lbx.write_table(t52b, "T52b_review_intensity_without_recent", floatfmt=".4f")
print(t52b.round(4).to_string())

# ----------------------------------------------------------------------------- R3 representativeness of the 2021 panel
con = sqlite3.connect(f"file:{lbx.DATA / 'letterboxd_2026.sqlite'}?mode=ro", uri=True)
f26 = pd.read_sql("SELECT slug, year, rating, rating_count, runtime, primary_language, genres FROM films", con)
f26 = f26[(f26.year <= 2020) & (f26.rating_count >= 1000) & f26.rating.notna()].drop_duplicates("slug").copy()
f26["in_panel"] = f26.slug.isin(p[p.clean].slug).astype(int)
f26["ln"] = np.log10(f26.rating_count)
f26["eng"] = f26.primary_language.eq("English").astype(int)
f26["runtime"] = f26.runtime.clip(upper=300).fillna(f26.runtime.median())
f26["age"] = 2026 - f26.year
for g in ["drama", "comedy", "horror", "documentary", "animation", "action", "thriller", "romance"]:
    f26[f"g_{g}"] = f26.genres.map(lambda s, g=g: g in (json.loads(s) if isinstance(s, str) else [])).astype(int)
rows = []
for lab, lo, hi in [("1k-10k ratings in 2026", 1e3, 1e4), ("10k-100k", 1e4, 1e5), (">= 100k", 1e5, 1e12), ("all >= 1k", 1e3, 1e12)]:
    s = f26[(f26.rating_count >= lo) & (f26.rating_count < hi)]
    rows.append({"group": lab, "films released by 2020": len(s), "in clean 2021 panel": int(s.in_panel.sum()),
                 "share in panel": s.in_panel.mean(),
                 "mean 2026 rating: in / not": f"{s[s.in_panel == 1].rating.mean():.3f} / {s[s.in_panel == 0].rating.mean():.3f}",
                 "median release year: in / not": f"{s[s.in_panel == 1].year.median():.0f} / {s[s.in_panel == 0].year.median():.0f}"})
gterms = " + ".join(f"g_{g}" for g in ["drama", "comedy", "horror", "documentary", "animation", "action", "thriller", "romance"])
for lab, s in [("all >= 1k", f26), ("10k-100k", f26[(f26.rating_count >= 1e4) & (f26.rating_count < 1e5)])]:
    lg = smf.logit(f"in_panel ~ ln + rating + I(rating**2) + age + I(age**2) + runtime + eng + {gterms}", s).fit(disp=0)
    pr = lg.predict(s)
    yv = s.in_panel.to_numpy()
    order = np.argsort(pr.to_numpy())
    ranks = np.empty(len(pr))
    ranks[order] = np.arange(1, len(pr) + 1)
    auc = (ranks[yv == 1].sum() - yv.sum() * (yv.sum() + 1) / 2) / (yv.sum() * (len(yv) - yv.sum()))
    rows.append({"group": f"logit of being in the panel on 2026 traits ({lab})", "films released by 2020": len(s),
                 "in clean 2021 panel": int(yv.sum()), "share in panel": np.nan,
                 "mean 2026 rating: in / not": f"McFadden pseudo-R2 {lg.prsquared:.3f}",
                 "median release year: in / not": f"AUC {auc:.3f}"})
t53 = pd.DataFrame(rows).set_index("group")
lbx.write_table(t53, "T53_panel_representativeness", floatfmt=".3f")
print(t53.to_string())

# ----------------------------------------------------------------------------- R4 break-date bootstrap
long, _ = lbx.load_archive()
fe = long[long.t >= "2019-01-01"]
fe = fe[fe.groupby("slug").slug.transform("size") >= 3].reset_index(drop=True)
YEAR = 365.25
STEPS = pd.date_range("2022-01-01", "2024-12-01", freq="MS")
KINKS = pd.date_range("2020-07-01", "2024-10-01", freq="QS")
tt = ((fe.t - pd.Timestamp("2021-01-01")).dt.days / YEAR).to_numpy()
days = fe.t.to_numpy()
yv = fe.rating.to_numpy(float)
step_cols = {s_: (days >= np.datetime64(s_)).astype(float) for s_ in STEPS}
kink_cols = {k: np.clip((days - np.datetime64(k)) / np.timedelta64(1, "D") / YEAR, 0, None) for k in KINKS}
film_rows = fe.groupby("slug").indices                       # slug -> row positions
slugs = np.array(list(film_rows))


def best_step(rows, codes):
    """Two-break scan on the given rows (with film codes for the fixed effects); returns the best step date."""
    cnt = np.bincount(codes)
    dm = lambda x: x - (np.bincount(codes, x) / cnt)[codes]
    y, t_ = dm(yv[rows]), dm(tt[rows])
    S = {s_: dm(c[rows]) for s_, c in step_cols.items()}
    K = {k: dm(c[rows]) for k, c in kink_cols.items()}
    best, arg = np.inf, None
    for s_ in STEPS:
        for k in KINKS:
            X = np.column_stack([t_, S[s_], K[k]])
            b, ssr, *_ = np.linalg.lstsq(X, y, rcond=None)
            ssr = ssr[0] if len(ssr) else ((y - X @ b) ** 2).sum()
            if ssr < best:
                best, arg = ssr, s_
    return arg


rng = np.random.default_rng(2026)
all_rows = np.arange(len(fe))
point = best_step(all_rows, pd.factorize(fe.slug)[0])
draws = []
for _ in range(200):
    pick = rng.choice(slugs, len(slugs), replace=True)
    rows = np.concatenate([film_rows[s_] for s_ in pick])
    codes = np.concatenate([np.full(len(film_rows[s_]), i) for i, s_ in enumerate(pick)])
    draws.append(best_step(rows, codes))
dr = pd.Series(pd.to_datetime(draws))
t54 = pd.DataFrame({"value": {
    "films / readings (2019+, >= 3 readings)": f"{fe.slug.nunique():,} / {len(fe):,}",
    "best step date, full sample": point.date().isoformat(),
    "bootstrap draws": len(dr),
    "share of draws with best step in May-Jul 2023": f"{dr.between('2023-05-01', '2023-07-01').mean():.1%}",
    "share of draws with best step exactly at full-sample date": f"{(dr == point).mean():.1%}",
    "2.5th-97.5th percentile of best step date": f"{dr.quantile(0.025).date()} to {dr.quantile(0.975).date()}",
    "most frequent best step dates": ", ".join(f"{d.date()} ({c})" for d, c in dr.value_counts().head(4).items())}})
lbx.write_table(t54, "T54_break_date_bootstrap")
print(t54.to_string())
