"""Shared library for the benchmark explanatory banners.

Each drawing primitive is emitted twice: once as a draw.io ``mxCell``
(editable model) and once as the matching rendered SVG element (preview),
so ``<name>.drawio.svg`` (with the model embedded in ``content=``) and the
plain ``<name>_banner.svg`` always agree. Visual vocabulary mirrors
``img/scramble.drawio.svg``.

Design rule (per review feedback): use colour sparingly. Structural
elements are neutral (white / light-grey, black outlines); a single red
accent tracks the one thing the benchmark actually measures or perturbs.
"""
from __future__ import annotations

import math

# (fill_light, fill_dark, stroke_light, stroke_dark) ; hex = light values
PAL = {
    'white':  ('#ffffff', 'var(--ge-dark-color, #121212)', '#000000', 'rgb(255,255,255)'),
    'gray':   ('#f5f5f5', 'rgb(26,26,26)',  '#666666', 'rgb(149,149,149)'),
    'dgray':  ('#e8e8e8', 'rgb(40,40,40)',  '#999999', 'rgb(120,120,120)'),
    'red':    ('#f8cecc', 'rgb(60,24,22)',  '#b85450', 'rgb(184,100,96)'),
    'accent': ('#e51400', 'rgb(255,120,100)', '#b20000', 'rgb(255,150,140)'),
}
S_CRES = 16384  # shorkie receptive field (ref only)


def _esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def _attresc(s):
    return _esc(s).replace('\n', '&#10;')


class Canvas:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.cells = []
        self.body = []
        self._id = 2

    # ---- model + body emit ------------------------------------------------
    def _vid(self):
        self._id += 1
        return self._id

    def _mvertex(self, style, x, y, w, h, value=''):
        i = self._vid()
        self.cells.append(
            f'        <mxCell id="{i}" parent="1" vertex="1" value="{_esc(value)}" '
            f'style="{style}">\n'
            f'          <mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" as="geometry" />\n        </mxCell>')

    def _medge(self, style, x1, y1, x2, y2, waypoint=None):
        i = self._vid()
        wp = ''
        if waypoint:
            wp = ('\n            <Array as="points">'
                  f'<mxPoint x="{waypoint[0]:.2f}" y="{waypoint[1]:.2f}" /></Array>')
        self.cells.append(
            f'        <mxCell id="{i}" parent="1" edge="1" value="" style="{style}">\n'
            f'          <mxGeometry relative="1" as="geometry">{wp}\n'
            f'            <mxPoint x="{x1:.2f}" y="{y1:.2f}" as="sourcePoint" />\n'
            f'            <mxPoint x="{x2:.2f}" y="{y2:.2f}" as="targetPoint" />\n'
            f'          </mxGeometry>\n        </mxCell>')

    # ---- style helpers ----------------------------------------------------
    def _fs(self, fill, stroke=None, fillnone=False):
        if fillnone:
            fa, fcss = 'fill="none"', 'fill: none;'
        else:
            fl, fd = PAL[fill][0], PAL[fill][1]
            fa, fcss = f'fill="{fl}"', f'fill: light-dark({fl}, {fd});'
        sk = stroke or fill
        sl, sd = PAL[sk][2], PAL[sk][3]
        return f'{fa} stroke="{sl}" style="{fcss} stroke: light-dark({sl}, {sd});"'

    # ---- text -------------------------------------------------------------
    def text(self, x, y, w, h, value, bold=False, size=12, align='left',
             valign='top', gray=False, rot=0):
        st = ('text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;'
              f'align={align};verticalAlign={valign};rounded=0;')
        if bold:
            st += 'fontStyle=1;'
        if size != 12:
            st += f'fontSize={size};'
        if gray:
            st += 'fontColor=#333333;'
        if rot:
            st += f'rotation={rot};'
        self._mvertex(st, x, y, w, h, value)
        tx = x if align == 'left' else (x + w / 2 if align == 'center' else x + w)
        anchor = {'left': 'start', 'center': 'middle', 'right': 'end'}[align]
        ty = (y + size if valign == 'top'
              else y + h / 2 + size * 0.34 if valign == 'middle' else y + h)
        rarg = None
        if rot:
            tx, ty = x + w / 2, y + h / 2 + size * 0.34
            anchor, rarg = 'middle', -rot
        self._btext(tx, ty, value, anchor, bold, size, gray, rarg)

    def _btext(self, x, y, s, anchor='start', bold=False, size=12, gray=False, rot=None):
        fw = ' font-weight="bold"' if bold else ''
        if gray:
            fill, extra = '#333333', ' style="fill: light-dark(#333333, #c1c1c1);"'
        else:
            fill, extra = 'light-dark(#000000, #ffffff)', ''
        tr = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot is not None else ''
        for ln, part in enumerate(str(s).split('\n')):
            dy = ln * (size * 1.2)
            self.body.append(
                f'<text x="{x:.1f}" y="{y + dy:.1f}" fill="{fill}"{extra} '
                f'font-family="Helvetica, Arial, sans-serif" font-size="{size}px" '
                f'text-anchor="{anchor}"{fw}{tr}>{_esc(part)}</text>')

    def _label(self, x, y, w, h, value, bold=False, size=12, gray=False):
        if value:
            self._btext(x + w / 2, y + h / 2 + size * 0.34, value, 'middle',
                        bold, size, gray)

    # ---- shapes -----------------------------------------------------------
    def rect(self, x, y, w, h, fill='white', value='', bold=False, size=12,
             gray=False, rounded=False, dashed=False, fillnone=False, stroke=None):
        st = ('rounded=1;arcSize=10;' if rounded else 'rounded=0;') + 'whiteSpace=wrap;html=1;'
        if dashed:
            st += 'dashed=1;dashPattern=6 4;'
        st += f'fillColor={"none" if fillnone else PAL[fill][0]};strokeColor={PAL[stroke or fill][2]};'
        if gray:
            st += 'fontColor=#333333;'
        if bold:
            st += 'fontStyle=1;'
        if size != 12:
            st += f'fontSize={size};'
        self._mvertex(st, x, y, w, h, value)
        rxy = ' rx="6" ry="6"' if rounded else ''
        da = ' stroke-dasharray="6 4"' if dashed else ''
        self.body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
                         f'height="{h:.2f}"{rxy}{da} {self._fs(fill, stroke, fillnone)}/>')
        self._label(x, y, w, h, value, bold, size, gray)

    def arrowhead(self, x, y, w, h, fill='white', west=False, value='', dx=10):
        st = ('html=1;shadow=0;dashed=0;align=center;verticalAlign=middle;'
              f'shape=mxgraph.arrows2.arrow;dy=0;dx={dx};notch=0;')
        if west:
            st += 'rotation=-180;'
        st += f'fillColor={PAL[fill][0]};strokeColor={PAL[fill][2]};'
        self._mvertex(st, x, y, w, h, value)
        cy = y + h / 2
        if west:
            p = f'M {x+dx} {y} L {x+w} {y} L {x+w} {y+h} L {x+dx} {y+h} L {x} {cy} Z'
        else:
            p = f'M {x} {y} L {x+w-dx} {y} L {x+w} {cy} L {x+w-dx} {y+h} L {x} {y+h} Z'
        self.body.append(f'<path d="{p}" stroke-miterlimit="10" {self._fs(fill)}/>')
        self._label(x, y, w, h, value, size=min(size_for(w), 11))

    def gene(self, x, y, bodyw, fill='gray', west=False, value='', headw=26, h=20):
        if west:
            self.arrowhead(x, y, headw, h, fill, west=True)
            self.rect(x + headw, y, bodyw, h, fill, value)
        else:
            self.rect(x, y, bodyw, h, fill, value)
            self.arrowhead(x + bodyw, y, headw, h, fill)

    def diamond(self, x, y, w, h, fill='white'):
        self._mvertex(f'rhombus;whiteSpace=wrap;html=1;fillColor={PAL[fill][0]};'
                      f'strokeColor={PAL[fill][2]};', x, y, w, h)
        cx, cy = x + w / 2, y + h / 2
        self.body.append(f'<path d="M {cx} {y} L {x+w} {cy} L {cx} {y+h} L {x} {cy} Z" '
                         f'stroke-miterlimit="10" {self._fs(fill)}/>')

    def tri(self, cx, cy, s, up=True, fill='white'):
        """small triangle marker (e.g. Cas9 cut)."""
        self._mvertex(f'triangle;whiteSpace=wrap;html=1;direction={"north" if up else "south"};'
                      f'fillColor={PAL[fill][0]};strokeColor={PAL[fill][2]};',
                      cx - s, cy - s, 2 * s, 2 * s)
        if up:
            p = f'M {cx} {cy-s} L {cx+s} {cy+s} L {cx-s} {cy+s} Z'
        else:
            p = f'M {cx-s} {cy-s} L {cx+s} {cy-s} L {cx} {cy+s} Z'
        self.body.append(f'<path d="{p}" stroke-miterlimit="10" {self._fs(fill)}/>')

    def looplimit(self, x, y, w, h, fill='gray', value='', size=12, gray=True):
        st = (f'shape=loopLimit;whiteSpace=wrap;html=1;size={size};'
              f'fillColor={PAL[fill][0]};strokeColor={PAL[fill][2]};')
        if gray:
            st += 'fontColor=#333333;'
        self._mvertex(st, x, y, w, h, value)
        v = min(size * 0.8, h * 0.5)
        s2 = min(size, w / 2)
        p = (f'M {x+s2} {y} L {x+w-s2} {y} L {x+w} {y+v} L {x+w} {y+h} '
             f'L {x} {y+h} L {x} {y+v} Z')
        self.body.append(f'<path d="{p}" stroke-miterlimit="10" {self._fs(fill)}/>')
        self._label(x, y, w, h, value, size=11, gray=gray)

    def trapezoid(self, x, y, w, h, fill='white', value='', size=40):
        self._mvertex(f'shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;'
                      f'html=1;fixedSize=1;size={size};fillColor={PAL[fill][0]};'
                      f'strokeColor={PAL[fill][2]};', x, y, w, h, value)
        self.body.append(f'<path d="M {x} {y+h} L {x+size} {y} L {x+w-size} {y} '
                         f'L {x+w} {y+h} Z" stroke-miterlimit="10" {self._fs(fill)}/>')
        self._label(x, y, w, h, value)

    def ellipse(self, cx, cy, r, fill='accent'):
        self._mvertex(f'ellipse;whiteSpace=wrap;html=1;fillColor={PAL[fill][0]};'
                      f'strokeColor={PAL[fill][2]};', cx - r, cy - r, 2 * r, 2 * r)
        self.body.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{r}" ry="{r}" '
                         f'{self._fs(fill)}/>')

    # ---- edges ------------------------------------------------------------
    def line(self, x1, y1, x2, y2, dashed=False, stroke='black', waypoint=None):
        sc = '#000000' if stroke == 'black' else PAL[stroke][2]
        sd = 'rgb(255,255,255)' if stroke == 'black' else PAL[stroke][3]
        st = 'endArrow=none;html=1;rounded=0;'
        if stroke != 'black':
            st += f'strokeColor={sc};'
        if dashed:
            st += 'dashed=1;dashPattern=6 4;'
        self._medge(st, x1, y1, x2, y2, waypoint)
        da = ' stroke-dasharray="6 4"' if dashed else ''
        pts = f'M {x1:.2f} {y1:.2f} ' + (f'L {waypoint[0]:.2f} {waypoint[1]:.2f} ' if waypoint else '') + f'L {x2:.2f} {y2:.2f}'
        self.body.append(f'<path d="{pts}" fill="none" stroke="{sc}"{da} '
                         f'stroke-miterlimit="10" style="stroke: light-dark({sc}, {sd});"/>')

    def arrow(self, x1, y1, x2, y2, waypoint=None):
        self._medge('endArrow=classic;html=1;rounded=0;', x1, y1, x2, y2, waypoint)
        self.line(x1, y1, x2, y2, waypoint=waypoint) if False else None
        sx, sy = (waypoint if waypoint else (x1, y1))
        da_pts = f'M {x1:.2f} {y1:.2f} ' + (f'L {waypoint[0]:.2f} {waypoint[1]:.2f} ' if waypoint else '') + f'L {x2:.2f} {y2:.2f}'
        self.body.append(f'<path d="{da_pts}" fill="none" stroke="#000000" '
                         f'stroke-miterlimit="10" style="stroke: light-dark(rgb(0,0,0), rgb(255,255,255));"/>')
        ang = math.atan2(y2 - sy, x2 - sx)
        L, ww = 10, 4
        bx, by = x2 - L * math.cos(ang), y2 - L * math.sin(ang)
        px, py = -math.sin(ang) * ww, math.cos(ang) * ww
        self.body.append(
            f'<path d="M {x2:.1f} {y2:.1f} L {bx+px:.1f} {by+py:.1f} '
            f'L {bx-px:.1f} {by-py:.1f} Z" fill="#000000" stroke="#000000" '
            f'style="fill: light-dark(rgb(0,0,0), rgb(255,255,255)); '
            f'stroke: light-dark(rgb(0,0,0), rgb(255,255,255));"/>')

    # ---- composite: coverage wiggle (decorative; not individually editable)
    def wiggle(self, x0, baseline, xs_ys, fill='gray', stroke=None):
        """filled coverage profile; one mxCell (group-ish) + svg polygon.
        xs_ys: list of (x,y_top)."""
        x1 = xs_ys[-1][0]
        pts = ' '.join(f'L {x:.1f} {y:.1f}' for x, y in xs_ys)
        d = f'M {x0:.1f} {baseline:.1f} ' + pts.replace('L', 'L', 1) + f' L {x1:.1f} {baseline:.1f} Z'
        # editable: a polygon via shape=mxgraph.basic.... fallback to a generic polyline cell
        # store as an mxgraph generic shape is complex; use a 'shape=mxgraph...' not available,
        # so represent as the bounding rect placeholder + svg path (visual only).
        self.body.append(f'<path d="{d}" stroke-miterlimit="10" {self._fs(fill, stroke)}/>')

    def karyotype(self, x0, y0, names_lengths, col_w, row_h, n_rows, accent_sites,
                  seed=0, label_w=34, max_bar=None, tick_color='accent'):
        """Two-/multi-column karyotype. names_lengths: list of (name,length).
        accent_sites: int total sites distributed proportional to length; ticks
        drawn per chromosome (capped ~per_cap each for cell economy)."""
        import random
        rng = random.Random(seed)
        maxlen = max(l for _, l in names_lengths)
        max_bar = max_bar or col_w - label_w - 14
        scale = max_bar / maxlen
        total = sum(l for _, l in names_lengths)
        for i, (nm, L) in enumerate(names_lengths):
            col = i // n_rows
            row = i % n_rows
            bx = x0 + col * col_w + label_w
            by = y0 + row * row_h
            bw = L * scale
            self.text(x0 + col * col_w - 2, by - 8, label_w, 16, nm, size=10,
                      align='right', gray=True)
            self.rect(bx, by - 3, bw, 6, 'dgray', rounded=True)
            # ticks
            n = max(2, round(accent_sites * L / total))
            n = min(n, 11)
            for _ in range(n):
                tx = bx + rng.random() * bw
                self.line(tx, by - 9, tx, by + 9, stroke=tick_color)

    # ---- output -----------------------------------------------------------
    def _model_xml(self):
        return (
            '<mxfile host="app.diagrams.net" agent="cursor" version="24.0.0">\n'
            '  <diagram name="figure" id="benchmark-banner">\n'
            f'    <mxGraphModel dx="1400" dy="800" grid="1" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{self.W}" pageHeight="{self.H}" math="0" shadow="0">\n'
            '      <root>\n        <mxCell id="0" />\n        <mxCell id="1" parent="0" />\n'
            + '\n'.join(self.cells) + '\n      </root>\n    </mxGraphModel>\n'
            '  </diagram>\n</mxfile>')

    def emit(self, base_path):
        head = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
                '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n')
        svg_open = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
            f'width="{self.W}px" height="{self.H}px" viewBox="0 0 {self.W} {self.H}" '
            f'style="background: #ffffff; background-color: light-dark(#ffffff, '
            f'var(--ge-dark-color, #121212)); color-scheme: light dark;"{{c}}>')
        bg = ('<rect fill="#ffffff" width="100%" height="100%" x="0" y="0" '
              'style="fill: light-dark(#ffffff, var(--ge-dark-color, #121212));"/>')
        bstr = '\n'.join(self.body)
        drawio = (head + '<!-- Do not edit this file with editors other than draw.io -->\n'
                  + svg_open.format(c=f' content="{_attresc(self._model_xml())}"')
                  + '\n' + bg + '\n<g>' + bstr + '</g>\n</svg>\n')
        plain = head + svg_open.format(c='') + '\n' + bg + '\n<g>' + bstr + '</g>\n</svg>\n'
        open(base_path + '.drawio.svg', 'w').write(drawio)
        open(base_path + '_banner.svg', 'w').write(plain)
        return len(self.cells)


def size_for(w):
    return 9


YEAST_CHROMS = [
    ('I', 230218), ('II', 813184), ('III', 316620), ('IV', 1531933),
    ('V', 576874), ('VI', 270161), ('VII', 1090940), ('VIII', 562643),
    ('IX', 439888), ('X', 745751), ('XI', 666816), ('XII', 1078177),
    ('XIII', 924431), ('XIV', 784333), ('XV', 1091291), ('XVI', 948066),
]
