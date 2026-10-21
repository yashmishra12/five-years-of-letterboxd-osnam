"""
Rating histograms: the raw mean behind each displayed score, and what that reveals about the formula.

Archive pilot (scrape2026/archive_hist_pilot.py; 100 random archive-sample films): readings of the histogram
fragment just before / after 19 Jun 2023 and near Feb 2021, plus each film's earliest archived page (2016-2020).
  H1 coverage: how many films have each kind of reading
  H2 old formula: displayed = (n*raw + C*m) / (n + C) fitted on pre-change readings from 2021 on (Bayesian average
     toward m, the platform average; m was lower in 2016-20, see T55c)
  H3 measured split of the change across June 2023: displayed change = raw-mean change + gap change,
     with gap = displayed - raw. The gap change is the platform's part, measured rather than inferred.
Live 2026 histograms (scrape2026/histograms_2026.py), when available:
  H4 the 2026 gap by raw level and rating count; films below the 200-rating display threshold

  .venv/bin/python 06_histograms.py
"""
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import lbx

lbx.style()
STARS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
HCOLS = [f"h{int(s * 10):02d}" for s in STARS]


def with_raw(df):
    h = df[HCOLS].fillna(0).to_numpy(float)
    n = h.sum(1)
    df = df.assign(n_hist=n, raw=(h * STARS).sum(1) / np.where(n > 0, n, np.nan))
    df["sd"] = np.sqrt((h * (STARS - df.raw.to_numpy()[:, None]) ** 2).sum(1) / np.where(n > 0, n, np.nan))
    df["gap"] = df.shown - df.raw
    return df


# ----------------------------------------------------------------------------- archive pilot
con = sqlite3.connect(f"file:{lbx.DATA / 'archive_hist_pilot.sqlite'}?mode=ro", uri=True)
done = pd.read_sql("SELECT * FROM done", con)
rd = with_raw(pd.read_sql("SELECT * FROM readings", con))
rd["t"] = pd.to_datetime(rd.served.str[:8], errors="coerce")
ok = rd[(rd.status == "ok") & rd.shown.notna() & (rd.n_hist > 0)].copy()
ok["n_match"] = (ok.n_hist - ok.shown_n).abs() <= 0.001 * ok.shown_n + 1     # histogram sums to the shown count

p = lbx.load_panel()
films = p[p.slug.isin(done.slug)][["slug", "rating_count_2021", "clean"]]
cov = done.merge(films, on="slug", how="left")
kinds = ["frag_near_2021", "frag_before_change", "frag_after_change", "page_early"]
for k in kinds:
    cov[k] = cov.slug.isin(ok[ok.kind == k].slug)
cov["both_sides"] = cov.frag_before_change & cov.frag_after_change
cov["n21_tercile"] = pd.qcut(cov.rating_count_2021, 3, labels=["lowest third", "middle third", "highest third"])
h1 = pd.concat([cov[kinds + ["both_sides"]].mean().to_frame("share of films").T,
                cov.groupby("n21_tercile", observed=True)[kinds + ["both_sides"]].mean()])
h1.insert(0, "films", [len(cov)] + cov.groupby("n21_tercile", observed=True).size().tolist())
lbx.write_table(h1, "T28_histogram_archive_coverage", floatfmt=".2f")

# H2: old formula on pre-change readings
pre = ok[ok.t < lbx.CHANGE]
post = ok[ok.t >= lbx.CHANGE]


def bayes(params, n, raw):
    C, m = params
    return (n * raw + C * m) / (n + C)


# The old rule pulled films toward "our overall platform average" (Letterboxd Journal, 19 Jun 2023), and that average
# moved: with the same C, the 2016-20 film pages fit m = 3.37 and the 2021-23 fragments m = 3.47 (12_revision_checks.py,
# T55c). Both uses below sit in 2021-23 (inverting the Feb-2021 scores; the old rule as last in force, June 2023), so
# the fit uses the readings from 2021 on.
fit_s = pre[pre.t >= "2021-01-01"]
early = pre[pre.t < "2021-01-01"]
fit = least_squares(lambda q: bayes(q, fit_s.shown_n.values, fit_s.raw.values) - fit_s.shown.values, x0=[50, 3.0],
                    bounds=([0, 0.5], [1e5, 5]))
C_hat, m_hat = fit.x
res = {k: v.shown - bayes(fit.x, v.shown_n, v.raw) for k, v in [("fit", fit_s), ("early", early), ("post", post)]}
h2 = pd.DataFrame({
    "readings": [len(fit_s), len(early), len(post)],
    "mean |displayed - raw|": [fit_s.gap.abs().mean(), early.gap.abs().mean(), post.gap.abs().mean()],
    "RMS error, old-formula fit": [np.sqrt((r ** 2).mean()) for r in res.values()],
    "share within ±0.005 (rounding)": [(r.abs() <= 0.0051).mean() for r in res.values()],
    "histogram sums to shown count": [fit_s.n_match.mean(), early.n_match.mean(), post.n_match.mean()],
}, index=["2021 to 18 Jun 2023 (fit sample)", "2016-2020 film pages (same formula; their own m is 3.37, T55c)",
          "after 19 Jun 2023 (same formula applied)"])
h2.loc["fitted C (prior weight, in ratings)", "readings"] = C_hat
h2.loc["fitted m (prior mean, stars)", "readings"] = m_hat
lbx.write_table(h2, "T29_old_formula_fit", floatfmt=".4f")

# H3: measured split across the change, for films with fragments on both sides
# side of the change by the capture actually served (Wayback can serve a nearby capture on the other side)
b = ok[(ok.kind == "frag_before_change") & (ok.t < lbx.CHANGE)].set_index("slug")
a = ok[(ok.kind == "frag_after_change") & (ok.t >= lbx.CHANGE)].set_index("slug")
both = b[["t", "shown", "raw", "gap", "shown_n", "sd"]].join(a[["t", "shown", "raw", "gap", "shown_n"]],
                                                             lsuffix="_b", rsuffix="_a", how="inner")
both["days"] = (both.t_a - both.t_b).dt.days
both["d_shown"], both["d_raw"], both["d_gap"] = both.shown_a - both.shown_b, both.raw_a - both.raw_b, both.gap_a - both.gap_b
h3 = both[["d_shown", "d_raw", "d_gap"]].apply(lbx.mean_ci).T
h3["median days between readings"] = both.days.median()
for lab, sub in [("gap <= 180 days", both[both.days <= 180])]:
    h3.loc[f"d_gap ({lab})"] = lbx.mean_ci(sub.d_gap)
lbx.write_table(h3, "T30_measured_split_across_change", floatfmt=".4f")
(lbx.DATA / "derived").mkdir(exist_ok=True)
both.to_csv(lbx.DATA / "derived" / "pilot_split.csv")

# ----------------------------------------------------------------------------- live 2026 histograms
live_db = lbx.DATA / "histograms.sqlite"
live = None
if live_db.exists():
    live = with_raw(pd.read_sql("SELECT * FROM hist WHERE status = 'ok'", sqlite3.connect(
        f"file:{live_db}?mode=ro", uri=True)))
    lv = live[live.shown.notna()].copy()
    lv["n_band"] = pd.cut(lv.shown_n, [199, 999, 9999, 99999, 1e9], labels=["200-999", "1k-10k", "10k-100k", ">=100k"])
    lv["raw_band"] = pd.cut(lv.raw, [0, 2, 2.5, 3, 3.5, 4, 5])
    h4 = lv.pivot_table(index="raw_band", columns="n_band", values="gap", aggfunc="mean", observed=True)
    h4["films"] = lv.groupby("raw_band", observed=True).size()
    lbx.write_table(h4, "T31_live_gap_by_level_and_count", floatfmt=".3f")
    hidden = live[live.shown.isna()]
    print(f"live histograms: {len(live):,} films ({len(lv):,} with a displayed score, {len(hidden):,} below the "
          f"threshold; hidden films' median count {hidden.n_hist.median():.0f}, max {hidden.n_hist.max():.0f})")

    # H5: measured decomposition of each panel film's 2021 -> 2026 change.
    # 2021 raw mean: invert the old formula (fitted above) on the Feb-2021 displayed score and count;
    # 2026 raw mean: from the live histogram. displayed change = raw change (members) + gap change, and the gap
    # change splits further, using the old formula at 2026 counts as the counterfactual:
    #   shrinkage decay = old-formula gap at 2026 count - 2021 gap (the prior's pull fades as ratings pile up;
    #                     would have happened with no formula change)
    #   formula switch  = 2026 displayed - old formula at 2026 count and raw (the platform's change itself)
    pan = lbx.load_panel()
    pan = pan[pan.clean].merge(live[["slug", "raw", "n_hist", "sd", "shown", "shown_n"]], on="slug")
    pan["raw21"] = (pan.rating_2021 * (pan.rating_count_2021 + C_hat) - C_hat * m_hat) / pan.rating_count_2021
    pan["gap21"] = pan.rating_2021 - pan.raw21
    pan["d_raw"] = pan.raw - pan.raw21
    pan["gap26"] = pan.shown - pan.raw
    pan["d_gap"] = pan.gap26 - pan.gap21
    pan["d_shown"] = pan.shown - pan.rating_2021
    pan["old26"] = bayes(fit.x, pan.n_hist, pan.raw)
    pan["d_decay"] = (pan.old26 - pan.raw) - pan.gap21
    pan["d_switch"] = pan.shown - pan.old26
    shownboth = pan[pan.shown.notna()]
    n21 = shownboth.rating_count_2021
    rows = {"all with a 2026 score": shownboth,
            "n2021 >= 200": shownboth[n21 >= 200],
            "n2021 200-999": shownboth[(n21 >= 200) & (n21 < 1000)],
            "n2021 1,000-9,999": shownboth[(n21 >= 1000) & (n21 < 10000)],
            "n2021 >= 10,000": shownboth[n21 >= 10000],
            "n2021 >= 1,000": shownboth[n21 >= 1000],
            "n2021 30-199": shownboth[n21 < 200]}
    h5 = pd.DataFrame({k: {"films": len(v), "displayed change": v.d_shown.mean(), "members (raw change)": v.d_raw.mean(),
                           "formula (gap change)": v.d_gap.mean(),
                           "formula share of change": v.d_gap.mean() / v.d_shown.mean() if len(v) else np.nan,
                           "of which: shrinkage decay": v.d_decay.mean(),
                           "of which: formula switch": v.d_switch.mean(),
                           "switch share of change": v.d_switch.mean() / v.d_shown.mean() if len(v) else np.nan,
                           "switch ci_lo": lbx.mean_ci(v.d_switch)["ci_lo"] if len(v) > 2 else np.nan,
                           "switch ci_hi": lbx.mean_ci(v.d_switch)["ci_hi"] if len(v) > 2 else np.nan}
                       for k, v in rows.items()}).T
    hid = pan[pan.shown.isna()]
    h5.loc["below 200 in 2026 (no score shown)"] = [len(hid), np.nan, hid.d_raw.mean()] + [np.nan] * 7
    lbx.write_table(h5, "T32_measured_decomposition_panel", floatfmt=".4f")
    pan.to_csv(lbx.DATA / "derived" / "panel_decomposition.csv", index=False)
    print(h5.round(4).to_string())

print(h1.round(2).to_string())
print(h2.round(4).to_string())
print(h3.round(4).to_string())
print(both[["days", "shown_b", "raw_b", "gap_b", "shown_a", "raw_a", "gap_a", "d_shown", "d_raw", "d_gap"]]
      .round(3).head(12).to_string())
