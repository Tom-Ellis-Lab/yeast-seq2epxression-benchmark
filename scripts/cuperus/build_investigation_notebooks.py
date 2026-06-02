"""Generate notebooks/cuperus_translation_features.ipynb.

Screens candidate "translation-only" 5'-UTR features (from Cuperus 2017) for
their correlation with growth_rate, to decide what goes into metric 2's `f`.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))

md(r"""# Cuperus 5′-UTR — candidate translation-only features for metric 2

**Question.** Metric 2 of the Cuperus benchmark credits the model for mRNA-channel
signal *beyond* a set of hand-crafted **translation-only** features `f`. For `f` to be
valid it must (a) carry real signal (otherwise metric 2 collapses to metric 1) and
(b) be *translation-only* — i.e. act through initiation efficiency, **not** through
mRNA abundance (uORF→NMD, structure→stability), or it steals the model's credit.

This notebook screens candidate features from the Cuperus 2017 feature analysis
(Fig 1; *Effects of 5′ UTR features*):

| feature | mechanism | channel | candidate for `f`? |
|---|---|---|---|
| **Kozak −3 = A** / −5…−1 context | translation initiation efficiency | translation-only | **yes** |
| **in-frame uAUG extension** (uAUG, no stop before main ATG) | adds N-terminal residues; initiation | translation-ish | maybe |
| uORF / out-of-frame uAUG | competes + triggers **NMD** | mRNA-mediated | **no** (model's to claim) |
| # uAUGs | uORF dosage | mRNA-mediated | no |
| MFE / secondary structure | scanning + mRNA **stability** | mixed/mRNA | no |
| GC / A-content | composition baseline | — | no |

We compute each on the full random library, correlate with `growth_rate` (overall and
on the clean depth bucket `t0 ≥ 101`), check collinearity (is Kozak orthogonal to the
mRNA-channel features?), and do an incremental-R² preview of metric 2.
""")

code('''from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
sns.set_theme(style="whitegrid", context="notebook")

def find_data_dir():
    p = Path.cwd()
    for _ in range(6):
        c = p / "archive" / "cuperus"
        if c.exists():
            return c
        p = p.parent
    raise FileNotFoundError("archive/cuperus not found from %s" % Path.cwd())

DATA = find_data_dir()
rand = pd.read_csv(DATA / "GSM2793752_Random_UTRs.csv", index_col=0)
print("random library:", rand.shape)
rand.head()''')

md(r"""## 1. Feature definitions

Coordinates: the 50 bp UTR sits immediately 5′ of the `HIS3` ATG, so the **main ATG
starts at index `L = len(UTR)`** (just past the UTR). Position −k (Kozak notation) is
index `L − k`; −1 is the last base of the UTR.

- **Kozak**: nucleotides at −1…−5 (UTR indices L−1 … L−5); `kozak_A_m3` = (−3 is A);
  `kozak_purine_m3` = (−3 is A or G).
- **uAUG classification** (per ATG at index `s`, frame = `(L − s) mod 3`):
  - *in-frame, no stop before main ATG* → **in-frame extension** (translation).
  - *in-frame with an in-frame stop before main ATG* → **uORF** (terminates upstream → NMD).
  - *out-of-frame* → **uORF** (reads in another frame, terminates somewhere → NMD).
    Caveat: out-of-frame stops past the main ATG would need the CDS; we treat any
    out-of-frame uAUG as a uORF, which matches Cuperus's definition.
- **MFE**: ViennaRNA minimum free energy of the UTR (the paper used −50..+70 incl. 70 nt
  of `HIS3` CDS; we use the UTR alone as a proxy and compute it on a subsample for speed).
- **GC / A-content** of the UTR.""")

code('''STOPS = {"TAA", "TAG", "TGA"}

def uaug_features(utr):
    L = len(utr); main = L
    starts = [k for k in range(L - 2) if utr[k:k+3] == "ATG"]
    n = len(starts)
    has_ext = has_uorf = False; n_oof = 0
    for s in starts:
        if (main - s) % 3 == 0:               # in frame with the main ATG
            stop_before = any(utr[j:j+3] in STOPS for j in range(s, main, 3))
            if stop_before:
                has_uorf = True               # in-frame uORF (terminates upstream)
            else:
                has_ext = True                # N-terminal extension (no stop)
        else:                                  # out of frame
            n_oof += 1; has_uorf = True
    return n, int(has_ext), int(has_uorf), n_oof

def add_features(df):
    u = df["UTR"]; L = u.str.len()
    df["len"] = L
    for k in range(1, 6):
        df[f"m{k}"] = [s[-k] if len(s) >= k else None for s in u]
    df["kozak_A_m3"] = (df["m3"] == "A").astype(int)
    df["kozak_purine_m3"] = df["m3"].isin(["A", "G"]).astype(int)
    feats = np.array([uaug_features(s) for s in u])
    df["n_uAUG"] = feats[:, 0]
    df["has_inframe_ext"] = feats[:, 1]
    df["has_uORF"] = feats[:, 2]
    df["n_oof_uAUG"] = feats[:, 3]
    df["has_uAUG"] = (df["n_uAUG"] > 0).astype(int)
    df["gc"] = u.str.count("[GC]") / L
    df["frac_A"] = u.str.count("A") / L
    return df

%time rand = add_features(rand)
rand[["UTR","growth_rate","t0","kozak_A_m3","n_uAUG","has_uORF","has_inframe_ext","gc"]].head()''')

code('''# MFE on a reproducible subsample (ViennaRNA is ~1 ms/seq; full library would be slow)
try:
    import RNA
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(rand.index.values, size=min(40000, len(rand)), replace=False)
    mfe = {i: RNA.fold(rand.at[i, "UTR"])[1] for i in sub_idx}
    rand["mfe"] = pd.Series(mfe)
    print("MFE computed on", len(mfe), "sequences; describe:")
    print(rand["mfe"].describe())
except ModuleNotFoundError:
    rand["mfe"] = np.nan
    print("ViennaRNA not available — MFE skipped")''')

md(r"""## 2. Univariate correlation with `growth_rate`

For continuous features: Spearman ρ (rank, robust) and Pearson r. For binary features:
point-biserial r (= Pearson with 0/1) and the group-mean gap (a direct effect size in
nats). Computed on the **full library** and on the **clean depth bucket** (`t0 ≥ 101`,
least measurement noise) — the gap between the two columns is the depth-attenuation we
expect.""")

code('''clean = rand[rand["t0"] >= 101]
print(f"full N={len(rand):,}   clean (t0>=101) N={len(clean):,}")

def corr_row(df, col, gr="growth_rate"):
    d = df[[col, gr]].dropna()
    x, y = d[col].values, d[gr].values
    rho = stats.spearmanr(x, y).statistic
    r = stats.pearsonr(x, y).statistic if len(set(x)) > 1 else np.nan
    out = {"n": len(d), "spearman": rho, "pearson": r}
    if set(np.unique(x)) <= {0, 1}:               # binary -> group-mean gap
        out["gap_nats"] = y[x == 1].mean() - y[x == 0].mean()
        out["frac_pos"] = x.mean()
    return out

CANDIDATE = ["kozak_A_m3", "kozak_purine_m3", "has_inframe_ext"]
CONTRAST  = ["has_uORF", "has_uAUG", "n_uAUG", "n_oof_uAUG", "mfe", "gc", "frac_A"]
rows = []
for col in CANDIDATE + CONTRAST:
    full = corr_row(rand, col); cl = corr_row(clean, col)
    rows.append({"feature": col,
                 "class": "translation?" if col in CANDIDATE else "mRNA/contrast",
                 "spearman_full": full["spearman"], "spearman_clean": cl["spearman"],
                 "pearson_clean": cl["pearson"],
                 "gap_nats_clean": cl.get("gap_nats", np.nan),
                 "frac_pos": cl.get("frac_pos", np.nan)})
summary = pd.DataFrame(rows).set_index("feature").round(3)
summary''')

md("""## 3. Kozak per-position effect (recover the −3 = A preference)

Mean `growth_rate` by nucleotide at each Kozak position −1…−5, on the clean bucket.""")

code('''fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
for ax, k in zip(axes, range(1, 6)):
    g = clean.groupby(f"m{k}")["growth_rate"].mean().reindex(list("ACGT"))
    g.plot(kind="bar", ax=ax, color="steelblue")
    ax.set(title=f"position -{k}", xlabel="", ylabel="mean growth_rate" if k == 1 else "")
fig.suptitle("Kozak per-position effect (clean bucket, t0>=101)", y=1.04)
fig.tight_layout(); plt.show()''')

md("""## 4. Collinearity — is Kozak orthogonal to the mRNA-channel features?

If the candidate Kozak feature is largely independent of `has_uORF` / `n_uAUG` / `mfe`,
then using Kozak as `f` and leaving uORF/structure to the model double-counts nothing.""")

code('''fcols = ["kozak_A_m3", "has_inframe_ext", "has_uORF", "n_uAUG", "mfe", "gc", "growth_rate"]
cm = clean[fcols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
ax.set_title("Spearman correlation among features (clean bucket)")
plt.show()''')

code('''# effect-size view: growth_rate by uORF status, and MFE vs growth_rate
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.violinplot(data=clean, x="has_uORF", y="growth_rate", ax=ax[0], inner="quartile")
ax[0].set(title="uORF presence vs growth_rate (clean)", xlabel="has_uORF")
sub = clean.dropna(subset=["mfe"])
hb = ax[1].hexbin(sub["mfe"], sub["growth_rate"], gridsize=40, cmap="viridis", mincnt=1)
ax[1].set(title="MFE vs growth_rate (clean, subsample)", xlabel="UTR MFE (kcal/mol)", ylabel="growth_rate")
fig.colorbar(hb, ax=ax[1]); fig.tight_layout(); plt.show()''')

md(r"""## 5. Incremental-R² preview of metric 2

Metric 2 reports the model's partial signal over `f`. Before any model, this previews
how much variance each block explains and how additive they are. OLS `growth_rate ~ X`
on the clean bucket (Kozak one-hot of −1…−5; uORF block; MFE):
""")

code('''import numpy as np
def r2(df, Xcols, cat=()):
    d = df.dropna(subset=list(Xcols) + ["growth_rate"]).copy()
    parts = []
    for c in Xcols:
        if c in cat:
            parts.append(pd.get_dummies(d[c], prefix=c, drop_first=True).astype(float))
        else:
            parts.append(d[[c]].astype(float))
    X = pd.concat(parts, axis=1).values
    X = np.column_stack([np.ones(len(X)), X])
    y = d["growth_rate"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, len(d)

kozak = ["m1", "m2", "m3", "m4", "m5"]
uorf  = ["has_uORF", "n_uAUG", "has_inframe_ext"]
blocks = {
    "Kozak (-1..-5 one-hot)":   r2(clean, kozak, cat=kozak),
    "uORF block":               r2(clean, uorf),
    "MFE":                      r2(clean.dropna(subset=["mfe"]), ["mfe"]),
    "Kozak + uORF":             r2(clean, kozak + uorf, cat=kozak),
    "Kozak + uORF + MFE":       r2(clean.dropna(subset=["mfe"]), kozak + uorf + ["mfe"], cat=kozak),
}
pd.DataFrame({k: {"R2": round(v[0], 4), "n": v[1]} for k, v in blocks.items()}).T''')

md("""## 6. Native library — same screen

Re-run the univariate correlations on the 11,856 native fragments (variable length;
`main ATG` index = fragment length, so the feature functions generalize).""")

code('''nat = pd.read_csv(DATA / "GSM2793754_Native_UTRs.csv", index_col=0)
nat = add_features(nat)
nat_clean = nat[nat["t0"] >= 101]
rows = []
for col in ["kozak_A_m3", "kozak_purine_m3", "has_inframe_ext", "has_uORF", "has_uAUG", "n_uAUG", "gc"]:
    full = corr_row(nat, col); cl = corr_row(nat_clean, col)
    rows.append({"feature": col, "spearman_full": full["spearman"],
                 "spearman_clean(t0>=101)": cl["spearman"],
                 "gap_nats_clean": cl.get("gap_nats", np.nan)})
print(f"native full N={len(nat):,}  clean N={len(nat_clean):,}")
pd.DataFrame(rows).set_index("feature").round(3)''')

md(r"""## 7. Takeaways

Filled in from the numbers above (regenerate to refresh):

1. **Kozak (−3 = A / −1…−5 context)** — is it the cleanest `f`? Check: nonzero correlation
   with `growth_rate`, near-zero collinearity with `has_uORF` / `mfe`, and a small-but-real
   incremental R² so metric 2 doesn't collapse to metric 1.
2. **uORF / # uAUG** — expect the strongest correlation (the dominant feature) → confirms it
   belongs to the **model's** mRNA channel, excluded from `f`.
3. **MFE** — expect weak (paper R²≈0.078) and partly mRNA-mediated → excluded.
4. Whether `has_inframe_ext` is worth adding to `f` depends on its size and orthogonality here.

The decision for the spec's `f` follows directly from §2 (signal), §4 (orthogonality), and §5
(incremental R²).""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
out = Path("notebooks/cuperus_translation_features.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out, "with", len(cells), "cells")
