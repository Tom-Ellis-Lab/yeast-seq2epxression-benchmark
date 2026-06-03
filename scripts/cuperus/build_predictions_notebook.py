"""Generate notebooks/cuperus_predictions.ipynb — inspect the v1 Cuperus
Shorkie/Yorzoi predictions: uORF mechanism, clean-bucket scatters, the native
Kozak-vs-mRNA contrast (why Yorzoi's native ρ collapses under the partial correlation), and
worked examples. Reads results/default/{shorkie,yorzoi}__cuperus_utr/. Not part
of scoring; investigation only (see ROADMAP pre-release cleanup).
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md(r"""# Cuperus 5′-UTR — Shorkie / Yorzoi prediction inspection (v1)

Headline (clean bucket `t0 ≥ 101`): Shorkie ρ ≈ 0.28, Yorzoi ρ ≈ 0.11. This
notebook looks under those numbers:

1. **uORF mechanism** — uORFs trigger NMD → less mRNA; do the models predict
   lower `HIS3` coverage for uORF-containing UTRs? (the mRNA channel they *can* see)
2. **Clean-bucket scatters** — predicted coverage vs `growth_rate`, per model.
3. **Native Kozak contrast** — why Yorzoi's native ρ (0.164) collapses to ~0
   under the partial correlation (it's Kozak context, not mRNA signal) while Shorkie's survives.
4. **Worked examples.**
""")

code('''from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from yeastbench.benchmarks._cuperus_metrics import kozak_design, partial_correlation
sns.set_theme(style="whitegrid", context="notebook")


def find_results():
    p = Path.cwd()
    for _ in range(6):
        c = p / "results" / "default"
        if (c / "shorkie__cuperus_utr" / "random.tsv").exists():
            return c
        p = p.parent
    raise FileNotFoundError("results/default/shorkie__cuperus_utr/random.tsv not found")


RES = find_results()

def load(model, lib):
    df = pd.read_csv(RES / f"{model}__cuperus_utr" / f"{lib}.tsv", sep="\\t")
    return df.rename(columns={"score": f"g_{model}"})

# merge the two models on UTR (row-aligned, but merge to be safe)
rand = load("shorkie", "random").merge(
    load("yorzoi", "random")[["UTR", "g_yorzoi"]], on="UTR")
nat = load("shorkie", "native").merge(
    load("yorzoi", "native")[["UTR", "g_yorzoi"]], on="UTR")
print("random:", rand.shape, "| native:", nat.shape)
for m in ("shorkie", "yorzoi"):
    s = json.load(open(RES / f"{m}__cuperus_utr" / "summary.json"))
    print(f"{m}: random ρ={s['random_spearman']:.3f} (clean {s['random_bucket5_spearman']:.3f}), "
          f"native ρ={s['native_spearman']:.3f}, partial-corr ρ "
          f"rand={s['random_partial_spearman']:.3f} nat={s['native_partial_spearman']:.3f}")
rand.head()''')

md(r"""## 1. uORF mechanism

uORFs (upstream ORFs) trigger NMD → lower mRNA → lower `growth_rate`. An RNA-seq
model *can* see this (mRNA channel), unlike pure translation effects. Detect a
uORF in each 50 bp UTR (an upstream ATG that either reads out of frame with the
HIS3 ATG, or in-frame with a stop before it), then ask: does the measured label
drop with uORFs (ground truth), and do the models' predicted `g` drop too?""")

code('''STOPS = {"TAA", "TAG", "TGA"}

def has_uorf(utr):
    L = len(utr); main = L
    for s in range(L - 2):
        if utr[s:s+3] != "ATG":
            continue
        if (main - s) % 3 == 0:                       # in frame with HIS3 ATG
            if any(utr[j:j+3] in STOPS for j in range(s, main, 3)):
                return True                            # in-frame stop before ATG -> uORF
        else:
            return True                                # out-of-frame uAUG -> uORF
    return False

clean = rand[rand.t0 >= 101].copy()
clean["uORF"] = clean.UTR.map(has_uorf)
print("clean-bucket UTRs with a uORF: %.1f%%" % (100 * clean.uORF.mean()))

def gap(col, by="uORF"):
    a = clean.loc[clean[by], col]; b = clean.loc[~clean[by], col]
    return b.mean(), a.mean(), a.mean() - b.mean()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for axi, (col, lab) in zip(ax, [("growth_rate", "measured growth_rate"),
                                ("g_shorkie", "Shorkie predicted g"),
                                ("g_yorzoi", "Yorzoi predicted g")]):
    sns.violinplot(data=clean, x="uORF", y=col, ax=axi, inner="quartile", cut=0)
    no, yes, d = gap(col)
    axi.set_title(f"{lab}\\nno-uORF {no:.3g} → uORF {yes:.3g}  (Δ {d:+.3g})", fontsize=10)
fig.suptitle("uORF effect (clean bucket): measured truth vs each model's prediction", y=1.03)
fig.tight_layout(); plt.show()''')

md("""Both `growth_rate` (truth) and the model `g` should drop for uORF-containing
UTRs if the model captures the NMD channel. Shorkie's drop tracks the measured
drop more than Yorzoi's.""")

md("""## 2. Clean-bucket scatter — predicted coverage vs growth_rate""")

code('''fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for axi, m in zip(ax, ["shorkie", "yorzoi"]):
    x = clean.growth_rate.values
    y = np.log(np.clip(clean[f"g_{m}"].values, 1e-9, None))
    hb = axi.hexbin(x, y, gridsize=50, cmap="viridis", mincnt=1)
    rho = stats.spearmanr(clean.growth_rate, clean[f"g_{m}"]).statistic
    axi.set(title=f"{m}  (clean bucket, ρ={rho:.3f})",
            xlabel="growth_rate", ylabel=f"log predicted {m} HIS3 coverage")
    fig.colorbar(hb, ax=axi, label="count")
fig.tight_layout(); plt.show()''')

md(r"""## 3. Native Kozak contrast — what the partial correlation catches

On native, Yorzoi's overall ρ (0.164) is decent but its partial-correlation ρ ≈ 0:
its correlation is the **Kozak start-context**, not mRNA signal. Residualize each
model's native `g` and `growth_rate` on the Kozak features `f` (cross-validated)
and recompute the correlation — Yorzoi's drops to ~0, Shorkie's survives.""")

code('''nat_clean = nat[nat.t0 >= 101].copy()
f = kozak_design(nat_clean.UTR.tolist())
E = nat_clean.growth_rate.values
rows = []
for m in ("shorkie", "yorzoi"):
    g = nat_clean[f"g_{m}"].values
    raw = stats.spearmanr(g, E).statistic       # rank, scale-free
    m2 = partial_correlation(np.log(g), E, f)   # log g, matching the benchmark
    rows.append({"model": m, "native raw ρ (clean)": round(raw, 3),
                 "partial-corr ρ": round(m2["partial_spearman"], 3),
                 "partial-corr incr R²": round(m2["incremental_r2"], 3)})
pd.DataFrame(rows).set_index("model")''')

md("""Yorzoi: raw ρ collapses to ~0 once Kozak is removed → its native signal *is*
Kozak. Shorkie: survives → genuine mRNA-channel signal beyond translation context.""")

md("""## 4. Worked examples — extreme growth_rate UTRs (clean bucket)""")

code('''cols = ["UTR", "growth_rate", "uORF", "g_shorkie", "g_yorzoi"]
ex = clean.sort_values("growth_rate")
print("=== 6 LOWEST growth_rate ===")
print(ex[cols].head(6).to_string(index=False))
print("\\n=== 6 HIGHEST growth_rate ===")
print(ex[cols].tail(6).to_string(index=False))''')

md("""## Takeaways (regenerate to refresh)

- Shorkie tracks the uORF→NMD drop and correlates ~2.5× better than Yorzoi.
- Yorzoi's native correlation is Kozak start-context, not mRNA signal (the partial correlation → 0).
- Both signs positive; the achievable ceiling is the RNA-visible fraction of a
  protein-level assay (CNN R²=0.62 on the clean split; Shorkie r²≈0.078).""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
out = Path("notebooks/cuperus_predictions.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out, "with", len(cells), "cells")
