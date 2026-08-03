"""Build all benchmark explanatory banners (editable .drawio.svg + plain _banner.svg).

Run:  python scripts/figures/make_banners.py [name ...]
With no args, builds every figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from banner_lib import Canvas, YEAST_CHROMS  # noqa: E402

IMG = Path(__file__).resolve().parents[2] / "img"


# ======================================================================= HONG
def build_hong():
    c = Canvas(2080, 700)
    c.text(12, 6, 1400, 20, 'Hong IGR insertions \u2014 one constant '
           'TDH3p\u00b7mCherry\u00b7ADH1t cassette integrated at 150 sites across all '
           '16 chromosomes (position effect)', bold=True, size=14)

    # ---- karyotype: 150 sites span every chromosome
    c.text(20, 60, 600, 16, '150 mCherry-cassette integration sites (red) '
           '\u2014 spread over all 16 chromosomes', gray=True, size=11)
    c.karyotype(20, 96, YEAST_CHROMS, col_w=530, row_h=40, n_rows=8,
                accent_sites=150, seed=7, label_w=40, max_bar=470)

    # ---- zoom: one IGR insertion
    zy = 470
    c.text(20, 436, 600, 16, 'Zoom \u2014 one intergenic insertion '
           '(cassette identical at every site)', bold=True, size=12)
    c.line(40, zy, 1060, zy)
    c.gene(60, zy - 10, 150, 'gray')                       # upstream native gene
    c.tri(372, zy, 7, up=False, fill='white')             # Cas9 cut
    c.text(330, zy + 12, 90, 14, 'Cas9 cut', gray=True, size=10, align='center')
    c.arrowhead(382, zy - 10, 86, 20, 'white', value='TDH3p')
    c.rect(468, zy - 10, 120, 20, 'red', value='mCherry')  # accent = the readout
    c.rect(588, zy - 10, 50, 20, 'white', value='ADH1t')
    c.gene(700, zy - 10, 150, 'gray')                      # downstream native gene
    c.text(382, zy - 34, 256, 14, 'constant reporter cassette', gray=True,
           size=10, align='center')
    # site-dependent fluorescence chips
    c.text(880, zy - 40, 180, 14, 'measured F (site-dependent)', gray=True, size=10)
    for k, (v, yy) in enumerate([(1.18, zy - 22), (1.00, zy - 4), (0.71, zy + 14)]):
        c.rect(880, yy, v * 70, 12, 'red')
        c.text(880 + v * 70 + 4, yy - 2, 44, 14, f'{v:.2f}', gray=True, size=10)

    # ---- divider + model panel
    c.line(1100, 30, 1100, 660, dashed=True, stroke='gray')
    c.text(1140, 60, 800, 16, 'Predict expression from the cassette-centered '
           'window', bold=True, size=12)
    wy = 120
    c.rect(1150, wy - 30, 820, 60, fillnone=True, dashed=True, stroke='gray')
    c.text(1154, wy - 48, 320, 14, 'model window (cassette-centered)', gray=True, size=10)
    c.line(1160, wy, 1960, wy)
    c.gene(1175, wy - 10, 120, 'gray')
    c.tri(1452, wy, 7, up=False, fill='white')
    c.arrowhead(1460, wy - 10, 80, 20, 'white', value='TDH3p')
    c.rect(1540, wy - 10, 120, 20, 'red', value='mCherry')
    c.rect(1660, wy - 10, 48, 20, 'white', value='ADH1t')
    c.gene(1740, wy - 10, 120, 'gray')
    c.line(1540, wy + 12, 1540, wy + 18)
    c.line(1540, wy + 18, 1660, wy + 18)
    c.line(1660, wy + 12, 1660, wy + 18)
    c.text(1540, wy + 20, 120, 14, 'readout: mCherry CDS', gray=True, size=10, align='center')
    c.arrow(1600, wy + 46, 1600, 200)
    c.trapezoid(1360, 200, 520, 34, 'white', value='Model')
    c.arrow(1600, 234, 1600, 262)
    # coverage track
    TB = 336
    c.line(1360, TB, 1900, TB)
    hts = [12, 18, 30, 46, 64, 72, 56, 32, 16]
    for i, h in enumerate(hts):
        bx = 1360 + i * 60
        c.looplimit(bx, TB - h, 60, h, 'red' if i in (4, 5) else 'gray', size=12)
    c.line(1600, 270, 1600, TB, dashed=True, stroke='gray')
    c.line(1720, 270, 1720, TB, dashed=True, stroke='gray')
    c.text(1600, TB + 4, 120, 14, 'mCherry CDS', gray=True, size=10, align='center')
    c.text(1372, 268, 230, 14, 'predicted expression = \u03a3 coverage over CDS',
           size=10)

    # metric box + scatter
    c.rect(1140, 440, 470, 214, 'gray', rounded=True)
    c.text(1158, 452, 300, 16, 'Scoring', bold=True, size=13)
    c.text(1158, 482, 400, 14, 'fluorescence_norm = F(site) / F(IntTrain92)')
    c.text(1158, 516, 300, 14, 'Primary metric', bold=True)
    c.text(1158, 538, 420, 14, 'Spearman \u03c1 ( predicted , measured )')
    c.text(1158, 558, 420, 14, 'over IntProp ( n = 52 )')
    c.text(1158, 592, 300, 14, 'Published ceiling', bold=True)
    c.text(1158, 614, 420, 14, 'SPCC(mRNA, fluo) = 0.847')
    _scatter(c, 1680, 470, 1960, 640, '\u03c1 > 0')

    c.emit(str(IMG / 'hong_igr'))


def _scatter(c, x0, y0, x1, y1, note='\u03c1 > 0', xlab='predicted', ylab='measured'):
    c.line(x0, y1, x1, y1)
    c.line(x0, y0, x0, y1)
    c.text((x0 + x1) / 2 - 70, y1 + 6, 140, 14, xlab, align='center', gray=True, size=10)
    c.text(x0 - 30, (y0 + y1) / 2 - 8, 80, 14, ylab, align='center', gray=True, size=10, rot=90)
    c.line(x0 + 18, y1 - 20, x1 - 14, y0 + 16, stroke='gray')
    pts = [(0.12, 0.20), (0.22, 0.12), (0.30, 0.40), (0.40, 0.30), (0.50, 0.55),
           (0.58, 0.42), (0.66, 0.70), (0.74, 0.58), (0.84, 0.82), (0.92, 0.74),
           (0.46, 0.66), (0.70, 0.48)]
    for px, py in pts:
        cx = x0 + 16 + px * (x1 - x0 - 30)
        cy = y1 - 14 - py * (y1 - y0 - 28)
        c.ellipse(cx, cy, 4, 'accent')
    c.text(x1 - 64, y0 + 2, 60, 16, note, align='right', bold=True, size=12)


def _divider(c, x, y0=30, y1=None):
    c.line(x, y0, x, y1 or (c.H - 30), dashed=True, stroke='gray')


def _modelbox(c, x, y, w, label='Model'):
    c.trapezoid(x, y, w, 30, 'white', value=label)


def _track(c, x0, x1, baseline, heights, accent_idx=(), bw=None):
    n = len(heights)
    bw = bw or (x1 - x0) / n
    c.line(x0, baseline, x1, baseline)
    for i, h in enumerate(heights):
        c.looplimit(x0 + i * bw, baseline - h, bw, h,
                    'red' if i in accent_idx else 'gray', size=min(12, bw / 2))


# ====================================================================== RAFI
def build_rafi():
    c = Canvas(2080, 560)
    c.text(12, 6, 1500, 20, 'Rafi / DREAM random-promoter MPRA \u2014 71,103 random '
           '80 bp promoters; two scoring modes, ranked per stratum', bold=True, size=14)

    # ---- construct
    c.text(40, 58, 500, 14, '110 bp oligo (constant flanks, variable insert)', gray=True, size=11)
    c.rect(40, 80, 95, 22, 'white', value='17 bp adapter', size=9)
    c.rect(135, 80, 250, 22, 'red', value='80 bp random insert', size=10)
    c.rect(385, 80, 70, 22, 'white', value='tail', size=9)

    _divider(c, 40 + 0, 0, 0)  # noop guard removed below
    # Mode 1 — marginalized foundation models
    c.text(40, 130, 720, 14, 'Mode 1 \u2014 zero-shot foundation models: logSED in native '
           'context', bold=True, size=12)
    by = 196
    c.line(60, by, 470, by)
    c.rect(78, by - 10, 64, 20, 'red', value='insert', size=9)
    c.text(150, by - 26, 130, 12, '180 bp upstream', gray=True, size=9, align='center')
    c.arrowhead(214, by - 10, 60, 20, 'white', value='prom', dx=10)
    c.tri(280, by, 6, up=False, fill='white')
    c.gene(290, by - 10, 150, 'gray', value='host gene')
    c.arrow(478, by, 540, by)
    _modelbox(c, 540, by - 16, 150, 'Model')
    c.arrow(690, by, 740, by)
    c.text(742, by - 18, 230, 14, 'mean logSED\nover 22 host loci', size=10)

    # Mode 2 — supervised DREAM-RNN
    c.text(40, 270, 720, 14, 'Mode 2 \u2014 supervised DREAM-RNN: predicts reporter '
           'expression', bold=True, size=12)
    dy = 330
    c.rect(78, dy - 10, 110, 20, 'white', value='plasmid ctx', size=9)
    c.rect(188, dy - 10, 64, 20, 'red', value='insert', size=9)
    c.rect(252, dy - 10, 90, 20, 'white', value='YFP', size=9)
    c.arrow(350, dy, 540, dy)
    _modelbox(c, 540, dy - 16, 150, 'DREAM-RNN')
    c.arrow(690, dy, 740, dy)
    c.text(742, dy - 10, 230, 14, 'predicted reporter expression', size=10)
    c.text(40, dy + 36, 900, 14, 'Both emit one scalar per sequence, fed its native '
           'substrate; compared by ranking only.', gray=True, size=10)

    # ---- metric: per-stratum bars
    _divider(c, 1020)
    c.text(1050, 58, 800, 14, 'Per-stratum ranking metric', bold=True, size=13)
    strata = ['high', 'low', 'yeast', 'random', 'chal', 'SNVs', 'motif\nperturb', 'motif\ntile']
    vals = [0.62, 0.55, 0.71, 0.78, 0.60, 0.50, 0.57, 0.66]
    bx0, baseY, bw = 1070, 320, 86
    c.line(bx0 - 10, baseY, bx0 + len(strata) * bw, baseY)
    c.line(bx0 - 10, 110, bx0 - 10, baseY)
    c.text(bx0 - 46, 100, 40, 14, '1.0', gray=True, size=9, align='right')
    for i, (s, v) in enumerate(zip(strata, vals)):
        h = v * 190
        acc = i >= 5
        c.rect(bx0 + i * bw, baseY - h, bw - 22, h, 'red' if acc else 'gray')
        c.text(bx0 + i * bw - 4, baseY + 6, bw, 26, s, gray=True, size=9, align='center')
    c.text(bx0, baseY + 44, 900, 14, 'Pearson r + Spearman \u03c1 per stratum; the three '
           'pair strata (red) also score \u0394(alt\u2212ref).', gray=True, size=10)
    c.text(bx0, baseY + 64, 900, 14, 'DREAM-RNN sets the in-distribution reference '
           '(overall r \u2248 0.97); zero-shot models rank lower.', gray=True, size=10)
    c.emit(str(IMG / 'rafi_promoter'))


# ==================================================================== SHALEM
def build_shalem():
    c = Canvas(2080, 560)
    c.text(12, 6, 1600, 20, 'Shalem 3\u2032-end / terminator MPRA \u2014 ~14k designed '
           'terminators, marginalized logSED across 22 host genes', bold=True, size=14)

    # oligo
    c.text(40, 58, 500, 14, '150 bp oligo', gray=True, size=11)
    c.rect(40, 80, 70, 22, 'white', value='P5', size=9)
    c.rect(110, 80, 250, 22, 'red', value='102 bp 3\u2032-end element', size=10)
    c.rect(360, 80, 60, 22, 'white', value='BC', size=9)
    c.rect(420, 80, 60, 22, 'white', value='P3', size=9)

    # construct per host gene
    c.text(40, 132, 900, 14, 'Per host gene: replace 450 bp downstream of the stop with '
           'insert + no-termination filler', bold=True, size=12)
    by = 210
    c.line(40, by, 520, by)
    c.gene(60, by - 10, 150, 'gray', value='host CDS')
    c.tri(214, by, 6, up=False, fill='white')
    c.text(214, by + 12, 60, 12, 'stop', gray=True, size=9, align='center')
    c.rect(224, by - 10, 90, 20, 'red', value='insert', size=9)
    c.rect(314, by - 10, 150, 20, 'white', value='300 bp no-term filler', size=9)
    c.text(224, by - 28, 240, 12, '450 bp replacement', gray=True, size=9, align='center')
    c.text(470, by + 4, 120, 12, 'native resumes', gray=True, size=9)

    c.arrow(540, by, 600, by)
    _modelbox(c, 600, by - 16, 160, 'Model')
    c.arrow(760, by, 820, by)
    c.text(822, by - 18, 260, 14, 'logSED over host exon bins\n\u2192 mean over 22 hosts',
           size=10)
    c.text(40, 300, 900, 14, 'REF = native 3\u2032-UTR (no insert), cached per host; '
           'ALT = insert replacement. logSED = log2(alt/ref).', gray=True, size=10)
    c.text(40, 322, 900, 14, 'Higher logSED = stronger terminator = more YFP.', gray=True, size=10)

    # metric scatter
    _divider(c, 1020)
    c.text(1050, 58, 800, 14, 'Overall correlation', bold=True, size=13)
    _scatter(c, 1120, 110, 1560, 420, '+ r', xlab='mean logSED (pred)', ylab='Expression (YFP)')
    c.text(1120, 440, 900, 14, 'Primary: overall Pearson r on (pred, Expression) over '
           '14,172 oligos; Spearman \u03c1 alongside.', gray=True, size=11)
    c.text(1120, 462, 900, 14, 'Technical noise floor \u2248 13 % median RSD '
           '(barcode duplicates).', gray=True, size=11)
    c.emit(str(IMG / 'shalem_terminator'))


# ====================================================================== KITA
def build_kita():
    c = Canvas(2080, 560)
    c.text(12, 6, 1600, 20, 'Kita cis-eQTL classification \u2014 eQTL vs distance-matched '
           'control, scored by predicted variant effect', bold=True, size=14)

    # locus with positive + matched negative
    c.text(40, 60, 900, 14, 'Positive: eQTL within 8 kb of TSS (Promoter / UTR5 / UTR3 / '
           'ORF). Negative: distance-matched control.', bold=True, size=12)
    by = 170
    c.line(40, by, 700, by)
    c.tri(120, by, 7, up=True, fill='white')
    c.text(80, by + 12, 80, 12, 'TSS', gray=True, size=9, align='center')
    c.gene(130, by - 10, 200, 'gray', value='regulated gene')
    c.line(120, by, 120, by - 40, stroke='gray')
    c.diamond(232, by - 44, 16, 18, 'accent')                 # positive variant
    c.text(150, by - 64, 200, 12, 'eQTL variant (\u2264 8 kb)', gray=True, size=9)

    by2 = 280
    c.line(40, by2, 700, by2)
    c.tri(120, by2, 7, up=True, fill='white')
    c.gene(130, by2 - 10, 200, 'gray', value='random gene')
    c.diamond(232, by2 - 44, 16, 18, 'white')                 # matched negative
    c.line(120, by2, 120, by2 - 40, stroke='gray')
    c.text(150, by2 - 64, 300, 12, 'matched control (same dist, MAF, REF/ALT)',
           gray=True, size=9)

    c.arrow(710, 225, 770, 225)
    _modelbox(c, 770, 209, 170, 'Model')
    c.text(772, 250, 220, 14, 'score = effect(ALT vs REF)', size=10)
    c.text(40, 360, 920, 14, 'VariantEffectScorer: REF vs ALT alleles \u2192 model \u2192 '
           'one scalar per variant.', gray=True, size=11)

    # ROC + PR
    _divider(c, 1020)
    c.text(1050, 58, 400, 14, 'AUROC (mean \u00b1 SEM, 4 iters)', bold=True, size=12)
    _roc(c, 1070, 110, 1380, 410, pr=False)
    c.text(1500, 58, 400, 14, 'AUPRC', bold=True, size=12)
    _roc(c, 1520, 110, 1830, 410, pr=True)
    c.text(1070, 440, 1000, 14, 'No class balancing; random + perfect baselines drawn; '
           'secondary report stratified by distance-to-TSS.', gray=True, size=11)
    c.emit(str(IMG / 'kita_eqtl'))


def _roc(c, x0, y0, x1, y1, pr=False):
    c.line(x0, y1, x1, y1)
    c.line(x0, y0, x0, y1)
    if pr:
        c.line(x0, y1 - (y1 - y0) * 0.25, x1, y1 - (y1 - y0) * 0.25, dashed=True, stroke='gray')
        curve = [(0.0, 0.95), (0.3, 0.9), (0.55, 0.8), (0.75, 0.62), (0.9, 0.4), (1.0, 0.25)]
        c.text((x0 + x1) / 2 - 40, y1 + 6, 80, 14, 'recall', align='center', gray=True, size=10)
        c.text(x0 - 30, (y0 + y1) / 2 - 8, 80, 14, 'precision', align='center', gray=True, size=10, rot=90)
    else:
        c.line(x0, y1, x1, y0, dashed=True, stroke='gray')        # random diagonal
        curve = [(0.0, 0.0), (0.1, 0.55), (0.25, 0.78), (0.5, 0.9), (0.75, 0.96), (1.0, 1.0)]
        c.text((x0 + x1) / 2 - 30, y1 + 6, 60, 14, 'FPR', align='center', gray=True, size=10)
        c.text(x0 - 30, (y0 + y1) / 2 - 8, 80, 14, 'TPR', align='center', gray=True, size=10, rot=90)
    pts = [(x0 + px * (x1 - x0), y1 - py * (y1 - y0)) for px, py in curve]
    for a, b in zip(pts, pts[1:]):
        c.line(a[0], a[1], b[0], b[1], stroke='accent')


# ===================================================================== MENEU
def build_meneu():
    c = Canvas(2080, 560)
    c.text(12, 6, 1700, 20, 'Meneu foreign-DNA coverage \u2014 whole bacterial chromosomes '
           'in yeast; tile end-to-end, predict RNA-seq zero-shot', bold=True, size=14)

    # chimeric contig
    c.text(40, 58, 800, 14, 'Integrated chimeric contig (per bacterial chromosome)',
           bold=True, size=12)
    cy = 120
    c.line(40, cy, 980, cy)
    c.tri(56, cy, 8, up=False, fill='white')
    c.tri(964, cy, 8, up=False, fill='white')
    c.rect(70, cy - 9, 380, 18, 'gray', value='bacterial arm')
    c.rect(450, cy - 9, 110, 18, 'red', value='CEN6/ARS-HIS3')
    c.rect(560, cy - 9, 380, 18, 'gray', value='bacterial arm')
    c.text(40, cy + 16, 940, 12, 'telomere \u2014 [Mpneumo ~818 kb, 40% GC  |  Mmmyco '
           '~1.22 Mb, 24% GC] \u2014 telomere', gray=True, size=9)

    # tiling -> model
    c.text(40, 178, 900, 14, 'Tile to the receptive field, predict each window, stitch',
           bold=True, size=12)
    ty = 232
    c.line(40, ty, 980, ty)
    for i in range(6):
        wx = 60 + i * 150
        c.rect(wx, ty - 12, 150, 24, 'white', dashed=True, stroke='gray')
    c.arrow(510, 260, 510, 300)
    _modelbox(c, 360, 300, 300, 'Model (per window)')
    c.arrow(510, 330, 510, 360)

    # predicted vs measured tracks
    import random
    rng = random.Random(3)
    base = [max(0, rng.random() ** 2) for _ in range(20)]
    meas = [v * 70 for v in base]
    pred = [max(2, v * 70 + rng.uniform(-12, 12)) for v in base]
    c.text(40, 372, 300, 12, 'measured RNA-seq', gray=True, size=10)
    _track(c, 360, 980, 392 + 0, meas, accent_idx=range(20))   # measured (accent)
    c.text(40, 420, 300, 12, 'predicted', gray=True, size=10)
    _track(c, 360, 980, 470, pred)                              # predicted (gray)

    # metrics
    _divider(c, 1040)
    c.text(1070, 60, 800, 14, 'Per-chromosome metrics (never pooled)', bold=True, size=13)
    c.text(1070, 100, 900, 14, 'Over non-overlapping 5 kb windows, 1 bp resolution:', gray=True, size=11)
    rows = [('Shape \u2014 co-variation', 'median per-window Pearson'),
            ('Shape \u2014 mass placement', 'median JS divergence (bits)'),
            ('Magnitude', 'per-window fold-change error,\ndepth-normalized')]
    yy = 140
    for a, b in rows:
        c.rect(1070, yy, 360, 56, 'gray', rounded=True)
        c.text(1086, yy + 8, 340, 14, a, bold=True, size=11)
        c.text(1086, yy + 28, 340, 14, b, gray=True, size=10)
        yy += 70
    c.text(1070, yy + 6, 980, 14, 'GC gradient reported (Mmmyco 24% \u2192 Mpneumo 40% '
           '\u2248 yeast); cis zero-shot fraction, read vs ExoShorkie refs.',
           gray=True, size=10)
    c.emit(str(IMG / 'meneu_foreign_dna'))


# ======================================================================== WU
def build_wu():
    c = Canvas(2080, 640)
    c.text(12, 6, 1700, 20, 'Wu RFP insertions \u2014 one fixed cassette swapped into the '
           'kanMX locus of 1044 ORF-deletion strains across all 16 chromosomes',
           bold=True, size=14)

    c.text(20, 58, 700, 14, '1044 integration loci (red) \u2014 uniformly scattered over '
           'all 16 chromosomes', gray=True, size=11)
    c.karyotype(20, 94, YEAST_CHROMS, col_w=520, row_h=38, n_rows=8,
                accent_sites=170, seed=11, label_w=40, max_bar=460)

    # zoom cassette
    zy = 470
    c.text(20, 432, 800, 14, 'Zoom \u2014 RFP transcription unit replaces kanMX at the '
           'ORF-deletion boundary (constant at every locus)', bold=True, size=12)
    c.line(30, zy, 1060, zy)
    c.gene(50, zy - 10, 110, 'gray', value='native')
    segs = [('U1\u00b7tag\u00b7U2', 70, 'white'), ('tCYC1', 56, 'white'),
            ('pURA3', 64, 'white'), ('mCherry', 110, 'red'), ('tADH1', 56, 'white'),
            ('pLEU2\u00b7LEU2\u00b7tLEU2', 130, 'white'), ('D2\u00b7tag\u00b7D1', 70, 'white')]
    x = 200
    for lab, w, col in segs:
        c.rect(x, zy - 10, w, 20, col, value=lab, size=8)
        x += w
    c.gene(x + 20, zy - 10, 110, 'gray', value='native')
    c.text(420, zy - 30, 110, 12, 'readout', gray=True, size=9, align='center')
    c.line(420, zy - 14, 470, zy - 14, stroke='gray')

    # model + metric
    _divider(c, 1110, 50, 620)
    c.text(1140, 60, 800, 14, 'Predict RFP/OD600 from the mCherry-centered window',
           bold=True, size=12)
    c.arrow(1300, 100, 1300, 138)
    _modelbox(c, 1170, 138, 260, 'Model')
    c.arrow(1300, 168, 1300, 200)
    _track(c, 1150, 1620, 300, [16, 24, 40, 58, 70, 60, 38, 22, 14], accent_idx=(3, 4, 5))
    c.text(1150, 232, 360, 14, 'readout: \u03a3 coverage over mCherry CDS', gray=True, size=10)

    _scatter(c, 1700, 100, 2000, 300, '+ r', xlab='predicted', ylab='RFP/OD600')

    # histogram of 5 classes
    c.text(1150, 360, 600, 14, 'Measured distribution (paper\u2019s 5 classes) + tail '
           'detection', bold=True, size=12)
    hx0, hbase = 1170, 560
    counts = [71, 311, 410, 161, 91]
    labels = ['xlow\n<5', 'low\n5\u20136', 'mod\n6\u20137', 'high\n7\u20138', 'xhigh\n\u22658']
    bw = 90
    c.line(hx0 - 10, hbase, hx0 + len(counts) * bw, hbase)
    for i, (n, lab) in enumerate(zip(counts, labels)):
        h = n / 410 * 150
        acc = i in (0, 4)
        c.rect(hx0 + i * bw, hbase - h, bw - 22, h, 'red' if acc else 'gray', value=str(n), size=9)
        c.text(hx0 + i * bw - 6, hbase + 6, bw, 28, lab, gray=True, size=9, align='center')
    c.text(hx0, hbase + 44, 900, 14, 'Primary: Pearson r + Spearman \u03c1 vs RFP/OD600. '
           'Secondary: extreme-low / extreme-high tail AUROC + AUPRC.', gray=True, size=10)
    c.emit(str(IMG / 'wu_rfpins'))


FIGS = {'hong': build_hong, 'rafi': build_rafi, 'shalem': build_shalem,
        'kita': build_kita, 'meneu': build_meneu, 'wu': build_wu}


def main():
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        cells = FIGS[n]()
        print(f'built {n}')


if __name__ == '__main__':
    main()
