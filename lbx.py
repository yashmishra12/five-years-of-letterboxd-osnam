"""Shared loading, cleaning and plotting helpers for the Letterboxd five-year panel analysis."""
import json
import re
import os
import sqlite3
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# LBX_DATA points at a folder of the shared CSVs (data package); default is the project's own crawl databases
DATA = Path(os.environ["LBX_DATA"]).resolve() if os.environ.get("LBX_DATA") else ROOT / "scrape2026" / "data"
ARCHIVE_DB = DATA / "archive_history.sqlite"
FILMS_LISTED_2021 = 94_051            # rows in the raw Feb-2021 scrape (feb2021.json), for the sample-flow table
OUT = Path(__file__).resolve().parent
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

T2021 = pd.Timestamp("2021-02-21")    # git commit date of feb2021.json (scrape taken on/just before)
CHANGE = pd.Timestamp("2023-06-19")   # Letterboxd Journal: "The Score: we've updated our weighted-rating calculations"
MIN_2026 = 200                        # smallest rating count with a displayed average in 2026

# Same title and year, but the 2026 page is a different film from the 2021 record. These pass the automatic
# runtime/genre checks, so they are excluded by hand. Found by: non-English 2021 record -> English 2026 page with
# >10x count growth, and archived pre-2021 counts above the 2021 count.
KNOWN_MISMATCHES = {
    "black-rain": "2021: Japanese drama/war (Imamura's film); 2026 page: Ridley Scott's film; "
                  "archived pre-2021 count 1,965 exceeds the 2021 count 1,399",
    "leap-year": "2021: Spanish, 94 min, 755 ratings (likely Ano bisiesto); 2026 page: Anand Tucker's film, "
                 "165,724 ratings",
    "scared-stiff-1987": "2021: Cantonese, 93 min, crime/comedy/horror/thriller; 2026 page: Richard Friedman's "
                         "US horror, 83 min",
    # excluded after manual review
    "from-the-depths-2020": "2021: Swedish, no runtime, drama/thriller/horror/mystery; 2026 page: English, 85 min, "
                            "Jose Montesinos; 35 -> 616 ratings",
    "last-gasp": "2021: German, 93 min, horror/thriller; 2026 page: English, 93 min, Scott McGinnis; "
                 "63 -> 1,083 ratings",
}
# Possible mismatches awaiting the author's review; kept in the analysis until decided
IDENTITY_REVIEW = {}

# ----------------------------------------------------------------------------- panel

def _genres(s):
    return set(json.loads(s)) if isinstance(s, str) else set()


def load_panel():
    """All 2021 rated films with match diagnostics. Columns added:
    matched, dup_slug, prerelease, runtime_ratio, genre_disjoint, count_shrank, suspect, clean, rated26,
    d (2026 - 2021 rating), growth (count ratio), lang, cohort."""
    p = pd.read_csv(DATA / "panel_2021_2026.csv")
    p["matched"] = p.match_status.str.startswith("ok")
    p["dup_slug"] = p.matched & p.slug.duplicated(keep=False)
    p["prerelease"] = p.year_2021 >= 2021
    p["runtime_ratio"] = p.duration_2021 / p.runtime
    g21, g26 = p.genres_2021.map(_genres), p.genres.map(_genres)
    p["genre_disjoint"] = [bool(a) and bool(b) and not (a & b) for a, b in zip(g21, g26)]
    p["count_shrank"] = p.rating_count < 0.9 * p.rating_count_2021
    rr = p.runtime_ratio
    p["runtime_mismatch"] = rr.notna() & ((rr < 2 / 3) | (rr > 1.5))
    p["suspect"] = p.runtime_mismatch | p.genre_disjoint | p.count_shrank
    p["manual_mismatch"] = p.matched & p.slug.isin(KNOWN_MISMATCHES)
    p["identity_review"] = p.matched & p.slug.isin(IDENTITY_REVIEW)
    p["clean"] = p.matched & ~p.dup_slug & ~p.prerelease & ~p.suspect & ~p.manual_mismatch
    p["rated26"] = p.rating.notna()
    p["d"] = p.rating - p.rating_2021
    p["growth"] = p.rating_count / p.rating_count_2021
    p["log_growth"] = np.log(p.growth)
    p["log10_n21"] = np.log10(p.rating_count_2021)
    p["is_tv"] = p.tmdb_type.eq("tv")
    # baseline (2021) covariates: 2026 metadata is post-treatment (22% of exact matches changed genres)
    p["lang"] = p.language_2021.fillna(p.primary_language)
    p["english"] = p.lang.eq("English")
    p["director1"] = p.directors.map(lambda x: (json.loads(x) or [None])[0] if isinstance(x, str) else None)
    p["cohort"] = pd.cut(p.year_2021, [0, 1959, 1979, 1999, 2009, 2014, 2017, 2018, 2019, 2020],
                         labels=["<1960", "1960-79", "1980-99", "2000-09", "2010-14", "2015-17",
                                 "2018", "2019", "2020"])
    p["genre_list"] = g21.map(sorted)          # baseline genres; use for genre analyses
    p["genre_list_2026"] = g26.map(sorted)     # sensitivity only
    return p


def analysis_panel(p=None, min_n21=None):
    """Clean matches that still show a 2026 average. min_n21=MIN_2026 gives the selection-free subsample
    (films that already had >= 200 ratings in 2021 keep their average ~99% of the time)."""
    p = load_panel() if p is None else p
    a = p[p.clean & p.rated26].copy()
    if min_n21:
        a = a[a.rating_count_2021 >= min_n21]
    return a

# ----------------------------------------------------------------------------- external joins

def _norm(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s.replace("&", "and")).strip()


def load_canon(p=None):
    """Sight and Sound 2022 poll membership per panel slug (critics' 250, directors' 100). Matched on normalised
    title (2026 title, original title or 2021 title) and year +-1; else on year +-1 and a shared director surname."""
    p = load_panel() if p is None else p
    c = pd.read_csv(DATA / "canon_sight_and_sound_2022.csv")
    m = p[p.matched][["slug", "title", "original_title", "title_2021", "year_2021", "directors"]].copy()
    m["dir_surnames"] = m.directors.map(lambda x: {d.split()[-1].lower() for d in json.loads(x)} if isinstance(x, str) else set())
    keys = {}
    for r in m.itertuples():
        for t in (r.title, r.original_title, r.title_2021):
            if isinstance(t, str):
                for dy in (-1, 0, 1):
                    keys.setdefault((_norm(t), r.year_2021 + dy), r.slug)
    out = []
    for r in c.itertuples():
        slug = keys.get((_norm(r.title), r.year))
        how = "title+year"
        if slug is None:
            surn = {d.strip().split()[-1].lower() for d in str(r.directors).split(",") if d.strip()}
            cand = m[(m.year_2021 - r.year).abs() <= 1]
            cand = cand[cand.dir_surnames.map(lambda s: bool(s & surn))]
            slug, how = (cand.slug.iloc[0], "director+year") if len(cand) == 1 else (None, "unmatched")
        out.append({"poll": r.poll, "rank": r.rank, "bfi_title": r.title, "year": r.year, "slug": slug, "match": how})
    return pd.DataFrame(out)


def load_leads(p=None):
    """Lead gender per panel film from IMDb's billed cast (scrape2026/imdb_leads.py): lead_female = the
    first-billed acting credit is "actress"; top3_female = share of "actress" among the top three billed.
    Films without acting credits (most documentaries) get NaN."""
    p = load_panel() if p is None else p
    c = pd.read_csv(DATA / "imdb_leads.csv")
    g = c.groupby("imdb_id").agg(lead_female=("category", lambda s: float(s.iloc[0] == "actress")),
                                 top3_female=("category", lambda s: (s == "actress").mean()),
                                 n_billed=("category", "size"))
    return p[["slug", "imdb_id"]].merge(g, left_on="imdb_id", right_index=True, how="left").drop(columns="imdb_id")


# ----------------------------------------------------------------------------- archive

def archive_tables():
    """(queue, readings) from the crawl database, or from archive_queue.csv / archive_readings.csv."""
    if ARCHIVE_DB.exists():
        con = sqlite3.connect(f"file:{ARCHIVE_DB}?mode=ro", uri=True, timeout=60)
        return (pd.read_sql("SELECT slug, position, status FROM queue", con),
                pd.read_sql("SELECT slug, timestamp, rating, rating_count, review_count, note FROM readings", con))
    return (pd.read_csv(DATA / "archive_queue.csv", usecols=["slug", "position", "status"]),
            pd.read_csv(DATA / "archive_readings.csv", dtype={"timestamp": str, "requested": str},
                        usecols=["slug", "timestamp", "rating", "rating_count", "review_count", "note"]))


def load_archive():
    """Long table of dated readings per archive film: Feb-2021 scrape, archive snapshots, Oct-2026 crawl.
    Only films whose processing finished and whose panel match is clean."""
    q, r = archive_tables()
    r = r[r.rating.notna()][["slug", "timestamp", "rating", "rating_count"]]
    q = q[q.status == "done"]
    p = load_panel()
    p = p[p.clean & p.rated26 & p.slug.isin(q.slug)]
    r = r[r.slug.isin(p.slug)].assign(t=lambda x: pd.to_datetime(x.timestamp.str[:8]), source="archive")
    first = p[["slug", "rating_2021", "rating_count_2021"]].rename(
        columns={"rating_2021": "rating", "rating_count_2021": "rating_count"}).assign(t=T2021, source="scrape2021")
    last = p[["slug", "rating", "rating_count", "fetched_at"]].assign(
        t=lambda x: pd.to_datetime(x.fetched_at.str[:10]), source="crawl2026").drop(columns="fetched_at")
    long = pd.concat([first, r[["slug", "rating", "rating_count", "t", "source"]], last], ignore_index=True)
    long = long.merge(q[["slug", "position"]], on="slug")
    long["detailed"] = long.position < 100
    # one reading per film-day: on a day with both, keep our own crawl's reading over the archive's
    long["_prio"] = long.source.eq("archive").astype(int)
    long = (long.sort_values(["slug", "t", "_prio"]).drop_duplicates(["slug", "t"], keep="first")
            .drop(columns="_prio").reset_index(drop=True))
    return long, p

# ----------------------------------------------------------------------------- stats & plotting

def mean_ci(x):
    x = pd.Series(x).dropna()
    n, m = len(x), x.mean()
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return pd.Series({"n": n, "mean": m, "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se, "median": x.median(),
                      "share_up": (x > 0).mean(), "share_down": (x < 0).mean()})


INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"   # categorical slots 1-3 (+violet, used alone)


def style():
    mpl.rcParams.update({
        "font.family": ["Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2, "lines.markersize": 5, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")


def write_table(df, name, floatfmt=".3f", index=True):
    df.to_csv(TAB / f"{name}.csv", index=index)
    (TAB / f"{name}.md").write_text(df.to_markdown(floatfmt=floatfmt, index=index) + "\n")
