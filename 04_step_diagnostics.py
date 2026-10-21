"""
Is the June-2023 change a clean step? Direct tests of the "+0.043 jump on 19 Jun 2023" hypothesis.
  D1  prediction test: what a clean +0.043 step (plus the measured drift) predicts for each film's
      before/after difference, compared with what is observed, by snapshot gap
  D2  timing traps: rollout before 19 Jun, or archived pages lagging the change
  D3  step vs ramp: film fixed-effects fits where the change phases in linearly over L months (L=0 is a step)
  D4  model-free trajectory: post-change readings minus the film's own last reading before the change,
      for films whose pre-change reading is within 4 months of the change
Tables T15-T18, figure figS3.

  .venv/bin/python 04_step_diagnostics.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import lbx

lbx.style()
CHG, YEAR = lbx.CHANGE, 365.25
H0_STEP = 0.043          # the handoff's jump
long, films = lbx.load_archive()
br = pd.read_csv(lbx.DATA / "derived" / "jump_per_film.csv", parse_dates=["t_before", "t_after"])
br = br[br.slug.isin(long.slug)]


def se_mean(x):
    return x.std(ddof=1) / np.sqrt(len(x))


# ----------------------------------------------------------------------------- D3 first (gives drift rates)
x = long[long.t >= "2019-01-01"]
x = x[x.groupby("slug").slug.transform("size") >= 3].copy()
tyr = (x.t - pd.Timestamp("2021-01-01")).dt.days / YEAR
since = (x.t - CHG).dt.days / YEAR
groups = x.slug.astype("category").cat.codes


def fe(X):
    Xd = X - X.groupby(x.slug.values).transform("mean")
    yd = x.rating - x.groupby("slug").rating.transform("mean")
    return sm.OLS(yd.values, Xd).fit(cov_type="cluster", cov_kwds={"groups": groups})


ramp_rows = []
for L in [0, 1, 2, 3, 4, 6, 9, 12, 18, 24]:
    Ly = L / 12
    ramp = (since >= 0).astype(float) if L == 0 else (since / Ly).clip(0, 1)
    X = pd.DataFrame({"pre slope": tyr, "change": ramp, "slope change after ramp": (since - Ly).clip(lower=0)})
    f = fe(X)
    lo, hi = f.conf_int().loc["change"]
    ramp_rows.append({"ramp length (months)": L, "SSR": f.ssr, "size of change": f.params["change"], "ci_lo": lo,
                      "ci_hi": hi, "slope before (/yr)": f.params["pre slope"],
                      "slope after ramp (/yr)": f.params["pre slope"] + f.params["slope change after ramp"]})
t17 = pd.DataFrame(ramp_rows).set_index("ramp length (months)")
sigma2 = t17.SSR.min() / (len(x) - x.slug.nunique() - 3)
t17["dSSR/sigma2 vs best"] = (t17.SSR - t17.SSR.min()) / sigma2
lbx.write_table(t17, "T17_step_vs_ramp", floatfmt=".4f")
pre_rate = t17.loc[0, "slope before (/yr)"]
post_rate = t17.loc[0, "slope after ramp (/yr)"]

# ----------------------------------------------------------------------------- D1 prediction test
pre_yr = (CHG - br.t_before).dt.days / YEAR
post_yr = (br.t_after - CHG).dt.days / YEAR
br["pred_h0"] = H0_STEP + pre_rate * pre_yr + post_rate * post_yr        # clean step + measured drift
br["pred_nostep"] = pre_rate * pre_yr + post_rate * post_yr             # same drift, no step
br["gap_m"] = br.gap_yr * 12
bins = [0, 3, 6, 9, 12, 18, 24, 120]
labels = ["0-3", "3-6", "6-9", "9-12", "12-18", "18-24", ">24"]
br["gap_bin"] = pd.cut(br.gap_m, bins, labels=labels)
d1 = br.groupby("gap_bin", observed=True).agg(films=("jump_raw", "size"), observed=("jump_raw", "mean"),
                                              observed_se=("jump_raw", se_mean),
                                              predicted_clean_step=("pred_h0", "mean"),
                                              predicted_drift_only=("pred_nostep", "mean"),
                                              median_ratings_before=("n_before", "median"))
d1["z (observed - clean step)"] = (d1.observed - d1.predicted_clean_step) / d1.observed_se
lbx.write_table(d1, "T15_clean_step_prediction_by_gap", floatfmt=".4f")

# ----------------------------------------------------------------------------- D2 timing traps
def window(lab, b0, b1, a0, a1):
    s = br[(br.t_before >= b0) & (br.t_before <= b1) & (br.t_after >= a0) & (br.t_after <= a1)]
    if len(s) < 5:
        return {"window": lab, "films": len(s)}
    m = s.jump_raw.mean()
    return {"window": lab, "films": len(s), "observed": m, "ci_lo": m - 1.96 * se_mean(s.jump_raw),
            "ci_hi": m + 1.96 * se_mean(s.jump_raw), "predicted_clean_step": s.pred_h0.mean(),
            "median gap (days)": s.gap_yr.median() * YEAR, "median ratings before": s.n_before.median(),
            "mean rating before": s.r_before.mean()}


d2 = pd.DataFrame([
    window("before Apr 1-Jun 18, after Jun 19-Sep 30 (tight)", "2023-04-01", "2023-06-18", "2023-06-19", "2023-09-30"),
    window("before Mar 1-May 15 (pre-rollout), after Jun 19-Sep 30", "2023-03-01", "2023-05-15", "2023-06-19", "2023-09-30"),
    window("before May 16-Jun 18, after Jun 19-Sep 30", "2023-05-16", "2023-06-18", "2023-06-19", "2023-09-30"),
    window("before Apr 1-Jun 18, after Jun 19-Jul 31 (early after)", "2023-04-01", "2023-06-18", "2023-06-19", "2023-07-31"),
    window("before Apr 1-Jun 18, after Aug 1-Oct 31 (later after)", "2023-04-01", "2023-06-18", "2023-08-01", "2023-10-31"),
    window("before Jan 1-Jun 18, after Jun 19-Dec 31 (wider)", "2023-01-01", "2023-06-18", "2023-06-19", "2023-12-31"),
]).set_index("window")
# the two pre-rollout/late windows differ sharply in popularity; compare them adjusting for count, level, gap
import statsmodels.formula.api as smf
g = br[(br.t_before >= "2023-03-01") & (br.t_before <= "2023-06-18") & (br.t_after >= "2023-06-19")
       & (br.t_after <= "2023-09-30")].copy()
g["late"] = (g.t_before >= "2023-05-16").astype(int)
fl = smf.ols("jump_raw ~ late + np.log10(n_before) + r_before + gap_yr", g).fit(cov_type="HC3")
d2.loc["late (May 16-Jun 18) minus early (Mar 1-May 15) baseline, adjusted for count, level, gap"] = {
    "films": len(g), "observed": fl.params.late, "ci_lo": fl.conf_int().loc["late", 0],
    "ci_hi": fl.conf_int().loc["late", 1]}
lbx.write_table(d2, "T16_timing_windows", floatfmt=".4f")

# ----------------------------------------------------------------------------- D4 model-free trajectory
close = br[br.t_before >= CHG - pd.Timedelta(days=120)][["slug", "r_before", "t_before"]]
traj = long.merge(close, on="slug")
traj = traj[traj.t >= "2019-01-01"].copy()
traj["rel"] = traj.rating - traj.r_before
traj["m"] = (traj.t - CHG).dt.days / (YEAR / 12)
edges = [-30, -18, -12, -6, -0.0001, 3, 6, 9, 12, 18, 24, 30, 36, 41]
traj["bin"] = pd.cut(traj.m, edges)
traj = traj[~((traj.t == traj.t_before))]            # the baseline reading itself is 0 by construction
d4 = traj.groupby("bin", observed=True).rel.apply(lbx.mean_ci).unstack()[["n", "mean", "ci_lo", "ci_hi"]]
d4["films"] = traj.groupby("bin", observed=True).slug.nunique()
lbx.write_table(d4, "T18_trajectory_close_baseline", floatfmt=".4f")

# ----------------------------------------------------------------------------- figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
mid = [1.5, 4.5, 7.5, 10.5, 15, 21, 30]
mids = [mid[labels.index(k)] for k in d1.index]
ax1.plot(mids, d1.predicted_clean_step, color=lbx.S2, lw=1.8, label=f"predicted: clean +{H0_STEP} step + drift")
ax1.plot(mids, d1.predicted_drift_only, color=lbx.MUTED, lw=1.4, label="predicted: drift only, no step")
ax1.errorbar(mids, d1.observed, yerr=1.96 * d1.observed_se, color=lbx.S1, marker="o", ms=4.5, lw=1.6,
             elinewidth=0.9, capsize=0, mec="white", mew=0.8, label="observed")
ax1.axhline(0, color=lbx.INK2, lw=0.8)
ax1.set_xlabel("Months between last pre- and first post-change snapshot")
ax1.set_ylabel("Rating after − before (stars)")
ax1.set_title("A  Short-gap films (mostly very popular) show small steps", loc="left")
ax1.legend(loc="upper left")
mm = [iv.mid for iv in d4.index]
pre, post = d4[[m < 0 for m in mm]], d4[[m > 0 for m in mm]]
for part in (pre, post):
    pm = [iv.mid for iv in part.index]
    ax2.errorbar(pm, part["mean"], yerr=[part["mean"] - part.ci_lo, part.ci_hi - part["mean"]], color=lbx.S1,
                 marker="o", ms=4.5, lw=1.6, elinewidth=0.9, capsize=0, mec="white", mew=0.8)
ax2.axvline(0, color=lbx.INK2, lw=0.8)
ax2.axhline(0, color=lbx.GRID, lw=0.8)
ax2.axhline(H0_STEP, color=lbx.S2, lw=1.2)
ax2.text(-26, H0_STEP - 0.004, f"+{H0_STEP}: where a clean step would put every post-change point", color=lbx.INK2, fontsize=7.5, va="top")
ax2.set_xlabel("Months relative to 19 Jun 2023")
ax2.set_ylabel("Rating minus film's last pre-change reading")
ax2.set_title(f"B  Gradual rise, {close.slug.nunique()} films with a baseline ≤4 months before", loc="left")
fig.tight_layout()
lbx.save(fig, "figS3_step_diagnostics")
plt.close(fig)

print(f"films {len(br)}; FE drift used: {pre_rate:+.4f}/yr before, {post_rate:+.4f}/yr after")
print(d1.round(4).to_string()); print(d2.round(4).to_string()); print(t17.round(4).to_string()); print(d4.round(4).to_string())
