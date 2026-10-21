"""
Cross-platform check against IMDb (public non-commercial dataset title.ratings.tsv.gz, downloaded 20 Oct 2026 into
scrape2026/data/imdb_ratings/; joined on imdb_id). IMDb's displayed average is itself an undisclosed weighted vote.

  X1 coverage and agreement: Letterboxd 2026 raw mean vs IMDb rating (on a 5-point scale: IMDb / 2). (T41)
  X2 does Letterboxd's adjustment move toward the wider consensus? Outcome displayed - raw for scored films, with
     flexible controls for the Letterboxd histogram (as switch (b) in 07): given the same Letterboxd histogram, does
     the IMDb rating, or Letterboxd's audience size relative to IMDb's, predict the adjustment? By size band, and
     whether the genre / trait effects of T35-T36 survive the IMDb controls. Director-clustered SEs. (T42, T43)

  .venv/bin/python 08_imdb_crossplatform.py     (run 06 first)
"""
import sqlite3

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import lbx

STARS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
HCOLS = [f"h{int(s * 10):02d}" for s in STARS]
MIN_VOTES = 100

imdb = pd.read_csv(lbx.DATA / "imdb_ratings" / "title.ratings.tsv.gz", sep="\t",
                   dtype={"tconst": str, "averageRating": float, "numVotes": int})
h = pd.read_sql("SELECT * FROM hist WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'histograms.sqlite'}?mode=ro", uri=True))
M = h[HCOLS].fillna(0).to_numpy(float)
n = M.sum(1)
P = M / n[:, None]
raw = P @ STARS
h = h.assign(n_hist=n, raw=raw, sd=np.sqrt((P * (STARS - raw[:, None]) ** 2).sum(1)), share5=P[:, 9],
             share_lo=P[:, :2].sum(1))

p = lbx.load_panel()
d = p[p.clean].merge(h[["slug", "shown", "n_hist", "raw", "sd", "share5", "share_lo"]], on="slug")
d = d.merge(imdb, left_on="imdb_id", right_on="tconst", how="left")
d = d.merge(lbx.load_leads(p).drop_duplicates("slug"), on="slug", how="left")
d["imdb5"] = d.averageRating / 2
d["gap"] = d.raw - d.imdb5                         # Letterboxd members' raw mean minus IMDb (5-point scale)
d["lb_vs_imdb"] = np.log10(d.n_hist / d.numVotes)   # Letterboxd audience relative to IMDb's
d["wedge"] = d.shown - d.raw
d["lvl26"] = d.raw - 3.5
d["ln26"] = np.log10(d.n_hist)
d["english"] = d.english.astype(int)
d["is_tv"] = d.is_tv.astype(int)
d["cluster"] = d.director1.fillna("film:" + d.slug)
GENRES = sorted({g for gs in d.genre_list for g in gs})
COL = {g: "g_" + g.replace(" ", "_") for g in GENRES}
for g in GENRES:
    d[COL[g]] = d.genre_list.map(lambda gs, g=g: g in gs).astype(int)
GT = " + ".join(COL.values())
HIST = "bs(lvl26, df=5) + bs(ln26, df=5) + lvl26:ln26 + sd + share5 + share_lo"

# ----------------------------------------------------------------------------- X1 coverage and agreement
rows = {}
for lab, s in [("clean panel films with a histogram", d), ("... with IMDb >= 100 votes", d[d.numVotes >= MIN_VOTES]),
               ("... and a 2026 Letterboxd score", d[(d.numVotes >= MIN_VOTES) & d.shown.notna()]),
               ("... and >= 10k Letterboxd ratings", d[(d.numVotes >= MIN_VOTES) & d.shown.notna() & (d.n_hist >= 10000)])]:
    has = s.imdb5.notna()
    rows[lab] = {"films": len(s), "with IMDb rating": int(has.sum()),
                 "corr(LB raw, IMDb)": s.raw.corr(s.imdb5), "corr(LB displayed, IMDb)": s.shown.corr(s.imdb5),
                 "mean LB raw - IMDb/2": s.gap.mean(), "sd of gap": s.gap.std(),
                 "median LB ratings / IMDb votes": (s.n_hist / s.numVotes).median()}
t41 = pd.DataFrame(rows).T
lbx.write_table(t41, "T41_imdb_coverage_agreement", floatfmt=".3f")
print(t41.round(3).to_string())

# ----------------------------------------------------------------------------- X2 adjustment vs IMDb
sc = d[d.shown.notna() & (d.numVotes >= MIN_VOTES)].dropna(subset=["imdb5", "lb_vs_imdb"]).copy()


def fit(df, rhs):
    return smf.ols(f"wedge ~ {rhs}", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.cluster)[0]})


rows = []
for lab, s in [("all scored", sc), ("200-999", sc[sc.n_hist < 1000]), ("1k-10k", sc[(sc.n_hist >= 1000) & (sc.n_hist < 10000)]),
               (">=10k", sc[sc.n_hist >= 10000])]:
    for spec, rhs in [("IMDb rating", f"imdb5 + {HIST}"),
                      ("IMDb rating + LB/IMDb audience + genres, English, TV", f"imdb5 + lb_vs_imdb + {HIST} + {GT} + english + is_tv")]:
        f = fit(s, rhs)
        r = {"band": lab, "spec": spec, "films": int(f.nobs)}
        for v, scale in [("imdb5", 1), ("lb_vs_imdb", 1)]:
            if v in f.params:
                lo, hi = f.conf_int().loc[v]
                r.update({f"{v} coef": scale * f.params[v], f"{v} ci_lo": scale * lo, f"{v} ci_hi": scale * hi,
                          f"{v} p": f.pvalues[v]})
        r["R2"] = f.rsquared
        rows.append(r)
t42 = pd.DataFrame(rows).set_index(["band", "spec"])
lbx.write_table(t42, "T42_imdb_predicts_adjustment", floatfmt=".4f")
print(t42.round(4).to_string())

# do the genre / trait effects on the adjustment survive IMDb controls? (all scored films with IMDb)
rows = []
for trait in ["g_horror", "g_documentary", "g_animation", "g_family", "english", "lead_female"]:
    s = sc.dropna(subset=["lead_female"]) if trait == "lead_female" else sc
    extra = " + lead_female" if trait == "lead_female" else ""
    for spec, rhs in [("histogram controls", f"{HIST} + {GT} + english + is_tv{extra}"),
                      ("+ IMDb rating, LB/IMDb audience", f"imdb5 + lb_vs_imdb + {HIST} + {GT} + english + is_tv{extra}")]:
        f = fit(s, rhs)
        lo, hi = f.conf_int().loc[trait]
        rows.append({"trait": trait.replace("g_", ""), "spec": spec, "coef": f.params[trait], "ci_lo": lo, "ci_hi": hi,
                     "p": f.pvalues[trait], "films": int(f.nobs)})
t43 = pd.DataFrame(rows).set_index(["trait", "spec"])
lbx.write_table(t43, "T43_traits_with_imdb_controls", floatfmt=".4f")
print(t43.round(4).to_string())
