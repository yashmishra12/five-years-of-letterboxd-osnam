"""
Selection from Letterboxd's higher minimum-ratings threshold (2021: ~30 ratings to show an average; 2026: 200).
Which 2021-rated films lost their displayed average, and does that bias the 2021->2026 change?
Tables: T11 retention by 2021 count/rating/cohort, T12 logit of losing the average, T13 bias check.
Figure: figS2 retention vs 2021 rating count.

  .venv/bin/python 03_selection.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import lbx

lbx.style()
p = lbx.load_panel()
c = p[p.clean].copy()           # clean matches, with or without a 2026 average
c["lost"] = (~c.rated26).astype(int)
films26 = pd.read_csv(lbx.DATA / "films_2026.csv", usecols=["rating", "rating_count"])
counts = films26.rating_count.dropna()

# threshold evidence
thr = pd.DataFrame({
    "value": [p.rating_count_2021.min(), counts.min(), int((counts.between(200, 209)).sum()),
              int(counts.between(190, 199).sum()), int(films26.rating.isna().sum()), len(films26)]},
    index=["2021: smallest count with an average", "2026: smallest count with an average",
           "2026: films with 200-209 ratings", "2026: films with 190-199 ratings (none expected)",
           "2026 crawl: films without an average", "2026 crawl: films fetched"])

# T11 retention
def ret(col, label):
    g = c.groupby(col, observed=True).agg(films=("lost", "size"), lost_average=("lost", "mean"))
    g.index = pd.MultiIndex.from_product([[label], g.index.astype(str)])
    return g


c["n21_band"] = pd.cut(c.rating_count_2021, [29, 49, 99, 149, 199, 299, 999, 1e8],
                       labels=["30-49", "50-99", "100-149", "150-199", "200-299", "300-999", ">=1000"])
c["r21_band"] = pd.cut(c.rating_2021, [0, 2.5, 3, 3.5, 4, 5], labels=["<=2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", ">4.0"])
t11 = pd.concat([ret("n21_band", "2021 rating count"), ret("r21_band", "2021 rating"), ret("cohort", "Release year"),
                 ret(np.where(c.english, "English", "Non-English"), "Language")])
lbx.write_table(t11, "T11_retention_by_strata")

# T12 logit: lost ~ 2021 count, rating, cohort, language
fit = smf.logit("lost ~ np.log2(rating_count_2021) + I(rating_2021 - 3.2) + C(cohort, Treatment('2000-09')) + english + is_tv",
                c).fit(disp=0, cov_type="HC1")
t12 = pd.DataFrame({"odds ratio": np.exp(fit.params), "ci_lo": np.exp(fit.conf_int()[0]),
                    "ci_hi": np.exp(fit.conf_int()[1]), "p": fit.pvalues})
t12.loc["N"] = [fit.nobs, np.nan, np.nan, np.nan]
lbx.write_table(t12, "T12_logit_lost_average")

# T13 bias check: does restricting to films that kept their average change the headline estimates?
a = lbx.analysis_panel(p)
s = a[a.rating_count_2021 >= lbx.MIN_2026]
rows = {}
for label, df in [("all films that kept an average", a), ("films with >=200 ratings in 2021 (99.8% kept)", s)]:
    f = smf.ols("d ~ I(rating_2021 - 3.2) + log10_n21 + C(cohort) + english + is_tv", df).fit(cov_type="HC3")
    rows[label] = {"N": len(df), "mean change": df.d.mean(), "slope on 2021 rating": f.params["I(rating_2021 - 3.2)"],
                   "retention in this group": c[c.rating_count_2021 >= (200 if "200" in label else 30)].rated26.mean()}
t13 = pd.DataFrame(rows).T
lbx.write_table(t13, "T13_selection_bias_check")
lbx.write_table(thr, "T14_threshold_evidence", floatfmt=",.0f")

# figS2: share losing the average by 2021 count, for 2021 rating terciles
fig, ax = plt.subplots(figsize=(5.2, 3.0))
c["r_t"] = pd.qcut(c.rating_2021, 3, labels=["lowest third", "middle third", "highest third"])
edges = [30, 40, 50, 60, 80, 100, 125, 150, 200, 300, 500, 1000, 1e8]
for (lab, g), col in zip(c.groupby("r_t", observed=True), [lbx.S1, lbx.S2, lbx.S3]):
    b = g.groupby(pd.cut(g.rating_count_2021, edges, right=False), observed=True).lost.mean()
    mids = [np.sqrt(iv.left * min(iv.right, 3000)) for iv in b.index]
    ax.plot(mids, b.values, color=col, marker="o", ms=4, mec="white", mew=0.6, lw=1.6,
            label=f"2021 rating: {lab}")
ax.set_xscale("log")
ax.set_xticks([30, 50, 100, 200, 500, 1000], ["30", "50", "100", "200", "500", "1,000"])
ax.minorticks_off()
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Ratings in Feb 2021 (log scale)")
ax.set_ylabel("Share with no 2026 average")
ax.set_title("Films below ~150 ratings in 2021 often lost their average", loc="left")
ax.legend(loc="upper right")
lbx.save(fig, "figS2_lost_average")
plt.close(fig)

print(thr.to_string())
print(t11.round(3).to_string())
print(t13.round(4).to_string())
