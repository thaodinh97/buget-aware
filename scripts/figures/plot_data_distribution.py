"""Figures of how the train split is distributed over the 20 FL clients and the 4 shards.

    python scripts/figures/plot_data_distribution.py          # writes docs/figures/{shard,client}_distribution.{png,svg,csv}

Partition: Dirichlet alpha = 0.5 over the binary label, partition seed 42, 20 clients (src.data.federated.partition, as in
every shard and DT experiment). Shards: assign_shards(4, 20) (permutation seed 7). Rows: the train split of
data/processed/connected_v3_auxfit_log1p_seed42 (114,284 rows). Colors: slots 1-2 of the dataviz reference palette,
validated with its validate_palette.js (light mode, all checks pass).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

from src.data.federated import partition
from src.fu.shard_exact import assign_shards

OUT = ROOT/'docs/figures'
DPI = 200
CSS = DPI/96                      # one CSS px in device px
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, BASELINE = '#e1e0d9', '#c3c2b7'
NORMAL, ATTACK = '#2a78d6', '#eb6834'    # categorical slots 1, 2
FONT = ['Segoe UI', 'DejaVu Sans']


def vn(n):
    return f'{int(round(n)):,}'.replace(',', '.')


def pct(x, digits=1):
    return f'{100*x:.{digits}f}'.replace('.', ',') + ' %'


def data(S):
    d = np.load(ROOT/'data/processed/connected_v3_auxfit_log1p_seed42/train.npz')
    y = d['y']
    parts = partition(y, 42, .5, 20)
    rows = []
    for j, clients in enumerate(assign_shards(S, 20)):
        for c in clients:
            p = parts[c]
            rows.append(dict(shard=j, client=c, rows=len(p), normal=int((y[p] == 0).sum()), attack=int((y[p] == 1).sum())))
    return pd.DataFrame(rows)


class Bars:
    """Thin bars, 2px surface gap between stacked segments, 4px rounded data-end (square at the baseline)."""

    def __init__(self, ax):
        self.ax = ax
        bbox = ax.get_window_extent()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        self.dx, self.dy = (x1-x0)/bbox.width, (y1-y0)/bbox.height    # data units per device px

    def width(self, band):
        return min(band*.62, 24*CSS*self.dx)

    def segment(self, x, bottom, top, width, color, round_top):
        gap = 2*CSS*self.dy
        if bottom > 0:
            bottom += gap/2
        if not round_top:
            top -= gap/2
        if top <= bottom:
            return
        x0, x1 = x-width/2, x+width/2
        rx, ry = min(4*CSS*self.dx, width/2), min(4*CSS*self.dy, (top-bottom))
        if round_top:
            verts = [(x0, bottom), (x1, bottom), (x1, top-ry), (x1, top), (x1-rx, top), (x0+rx, top), (x0, top),
                     (x0, top-ry), (x0, bottom)]
            codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3,
                     MPath.CURVE3, MPath.CLOSEPOLY]
            self.ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor='none', zorder=3))
        else:
            self.ax.add_patch(Rectangle((x0, bottom), width, top-bottom, facecolor=color, edgecolor='none', zorder=3))

    def stack(self, x, values, colors, width):
        bottom = 0.
        for i, (v, c) in enumerate(zip(values, colors)):
            self.segment(x, bottom, bottom+v, width, c, round_top=i == len(values)-1)
            bottom += v


def style(ax, ymax, ticks, fmt):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, ymax)
    ax.set_yticks(ticks)
    ax.set_yticklabels([fmt(t) for t in ticks], color=MUTED, fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(axis='both', length=0)


def legend(fig, y):
    handles = [Rectangle((0, 0), 1, 1, facecolor=ATTACK), Rectangle((0, 0), 1, 1, facecolor=NORMAL)]
    fig.legend(handles, ['Attack', 'Normal'], loc='upper left', bbox_to_anchor=(0.075, y), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK2, handlelength=1, handleheight=1, columnspacing=1.6)


def shard_figure(df):
    s = df.groupby('shard').agg(rows=('rows', 'sum'), normal=('normal', 'sum'), attack=('attack', 'sum'),
                                clients=('client', lambda c: ', '.join(map(str, sorted(c))))).reset_index()
    total, S = s.rows.sum(), len(s)
    fig = plt.figure(figsize=(1.6*S+0.8, 4.6), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes([0.075, 0.2, 0.9, 0.58])
    ax.set_xlim(-0.6, len(s)-0.4)
    ymax = 1.14*s.rows.max()
    style(ax, ymax, np.arange(0, ymax, 20000), vn)
    bars = Bars(ax)
    w = bars.width(1)
    for i, r in s.iterrows():
        bars.stack(i, [r.normal, r.attack], [NORMAL, ATTACK], w)
        ax.text(i, r.rows+1400, f'{vn(r.rows)} dòng · {pct(r.rows/total)}', ha='center', va='bottom', fontsize=9,
                color=INK, fontweight='semibold' if r.rows == s.rows.max() else 'normal')
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([f'Shard {r.shard}\nclient {r.clients}\nattack {pct(r.attack/r.rows, 0)}' for _, r in s.iterrows()],
                       fontsize=8.5, color=INK2)
    sizes = sorted(set(df.groupby('shard').size()))
    per = f'{sizes[0]}' if len(sizes) == 1 else f'{sizes[0]}–{sizes[-1]}'
    fig.text(0.075, 0.94, f'Phân bố dữ liệu train giữa {S} shard (S = {S})', fontsize=12.5, color=INK, fontweight='semibold')
    fig.text(0.075, 0.895, f'{vn(total)} dòng; mỗi shard gồm {per} client (gán bằng hoán vị seed 7)', fontsize=9, color=INK2)
    legend(fig, 0.865)
    return fig, s


def client_figure(df):
    total = df.rows.sum()
    n = len(df)
    fig = plt.figure(figsize=(10, 6.4), dpi=DPI, facecolor=SURFACE)
    top = fig.add_axes([0.07, 0.47, 0.83, 0.33])
    low = fig.add_axes([0.07, 0.14, 0.83, 0.24])
    # 20 clients grouped by shard, with one empty band between groups
    xs, x, prev = [], 0., None
    for sh in df.shard:
        if prev is not None and sh != prev:
            x += 0.8
        xs.append(x)
        x += 1
        prev = sh
    xs = np.array(xs)
    for ax in (top, low):
        ax.set_xlim(xs.min()-0.7, xs.max()+0.7)
    style(top, 32000, np.arange(0, 30001, 10000), vn)
    style(low, 1.0, [0, .25, .5, .75, 1.], lambda t: f'{int(round(100*t))} %')
    bt, bl = Bars(top), Bars(low)
    w = min(bt.width(1), bl.width(1))
    for xi, (_, r) in zip(xs, df.iterrows()):
        bt.stack(xi, [r.normal, r.attack], [NORMAL, ATTACK], w)
        bl.stack(xi, [r.normal/r.rows, r.attack/r.rows], [NORMAL, ATTACK], w)
    # selective labels: the largest and the smallest client
    for idx in (df.rows.idxmax(), df.rows.idxmin()):
        r = df.loc[idx]
        top.text(xs[df.index.get_loc(idx)], r.rows+700, f'{vn(r.rows)}', ha='center', va='bottom', fontsize=8.5,
                 color=INK, fontweight='semibold')
    share = df.normal.sum()/total
    low.axhline(share, color=INK2, linewidth=.8, linestyle=(0, (3, 2)), zorder=4)
    low.text(xs.max()+0.75, share, f' toàn tập:\n {pct(share, 0)} normal', va='center', ha='left', fontsize=8, color=INK2)
    top.set_xticks([])
    low.set_xticks(xs)
    low.set_xticklabels([str(c) for c in df.client], fontsize=8.5, color=INK2)
    for sh in sorted(df.shard.unique()):
        g = xs[(df.shard == sh).to_numpy()]
        low.text(g.mean(), -0.2, f'Shard {sh}', ha='center', va='top', fontsize=9, color=INK, transform=low.get_xaxis_transform())
    top.set_ylabel('Số dòng', fontsize=9, color=INK2)
    low.set_ylabel('Tỉ lệ', fontsize=9, color=INK2)
    fig.text(0.07, 0.035, f'Số dưới cột: mã client. Client được xếp theo shard chứa nó (S = {df.shard.nunique()}).',
             fontsize=8, color=MUTED)
    fig.text(0.07, 0.95, 'Phân bố dữ liệu train giữa 20 client', fontsize=12.5, color=INK, fontweight='semibold')
    fig.text(0.07, 0.915, f'{vn(total)} dòng, chia theo Dirichlet α = 0,5 trên nhãn (seed 42). Trên: số dòng mỗi client; '
             'dưới: tỉ lệ normal / attack', fontsize=9, color=INK2)
    legend(fig, 0.89)
    return fig


def main():
    plt.rcParams['font.family'] = FONT
    OUT.mkdir(parents=True, exist_ok=True)
    for S in (4, 5):                     # S = 4: the DT / FU experiments; S = 5: also in shard v2 / v3
        suffix = '' if S == 4 else f'_s{S}'
        df = data(S)
        fig, s = shard_figure(df)
        for ext in ('png', 'svg'):
            fig.savefig(OUT/f'shard_distribution{suffix}.{ext}', dpi=DPI, facecolor=SURFACE)
        s.assign(share=s.rows/s.rows.sum(), attack_share=s.attack/s.rows).to_csv(OUT/f'shard_distribution{suffix}.csv',
                                                                                 index=False)
        fig = client_figure(df)
        for ext in ('png', 'svg'):
            fig.savefig(OUT/f'client_distribution{suffix}.{ext}', dpi=DPI, facecolor=SURFACE)
        df.assign(share=df.rows/df.rows.sum(), attack_share=df.attack/df.rows).to_csv(
            OUT/f'client_distribution{suffix}.csv', index=False)
        plt.close('all')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
