import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# Result file prefix -> label shown on the charts.
MODELS = {
    "8M": "potion-base-8M",
    "32M": "potion-retrieval-32M",
    "128M": "potion-multilingual-128M",
    "constella": "constella-zero",
}
MODES = ("dense", "hybrid")
METRICS = ("ndcg@10", "mrr@10", "recall@10")

# Qdrant brand palette
WASH = "#E2E7F5"
WHITE = "#FFFFFF"
N30, N50, N60, N90 = "#28324D", "#576280", "#717C99", "#D4D9E6"
BLUE, TEAL, ERROR = "#2F6FF0", "#038585", "#E02828"
MODE_COLORS = {"dense": BLUE, "hybrid": TEAL}

FONT = 'ui-monospace, "JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace'

# Canvas geometry. The viewBox starts left of 0 to leave room for the rotated y title.
VB_X, VB_Y, VB_W, VB_H = -70, 0, 1170, 620
PLOT_L, PLOT_R, PLOT_T, PLOT_B = 80, 1080, 70, 500


@dataclass(frozen=True)
class Scale:
    """Linear map from a data range to pixel coordinates along an axis."""

    d0: float
    d1: float
    p0: float
    p1: float

    def __call__(self, v: float) -> float:
        return self.p0 + (v - self.d0) / (self.d1 - self.d0) * (self.p1 - self.p0)


Scores = dict[str, dict[str, dict[str, float]]]  # mode -> model -> metric -> mean


def load_scores(results_dir: Path) -> Scores:
    scores: Scores = {mode: {} for mode in MODES}
    for mode in MODES:
        for model in MODELS:
            path = results_dir / f"{mode}_{model}_fiqa_evaluation.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing results file: {path}")
            with open(path) as f:
                data = json.load(f)
            scores[mode][model] = {m: data[m]["mean"] for m in METRICS}
    return scores


def _text(
    x: float, y: float, s: str, size: int, fill: str, anchor: str = "middle"
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}">{s}</text>'
    )


def _model_label(cx: float, model: str) -> str:
    """Two-line category label, split at the first hyphen so four models fit."""
    first, second = MODELS[model].split("-", 1)
    return _text(cx, PLOT_B + 36, first, 20, N50) + _text(
        cx, PLOT_B + 62, second, 20, N50
    )


def _page(svg_body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FiQA chart</title>
<style>
  html, body {{ margin: 0; }}
  body {{ background: {WASH}; font-family: {FONT}; }}
  .frame {{ display: inline-block; padding: 48px; }}
  .canvas {{ background: {WHITE}; padding: 24px; }}
  svg {{ display: block; font-family: {FONT}; }}
</style>
</head>
<body>
<div class="frame"><div class="canvas">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="{VB_X} {VB_Y} {VB_W} {VB_H}" width="{VB_W}" height="{VB_H}">
{svg_body}
</svg>
</div></div>
</body>
</html>
"""


def _axes(
    y: Scale,
    ticks: list[float],
    y_title: str,
    x_title: str,
    fmt: str,
) -> list[str]:
    out = []
    for t in ticks:
        py = y(t)
        out.append(
            f'<line x1="{PLOT_L}" y1="{py:.1f}" x2="{PLOT_R}" y2="{py:.1f}" '
            f'stroke="{N30 if t == 0 else N90}" stroke-width="{2 if t == 0 else 1}"/>'
        )
        out.append(_text(PLOT_L - 14, py + 6, format(t, fmt), 18, N60, "end"))
    cy = (PLOT_T + PLOT_B) / 2
    out.append(
        f'<text x="{VB_X + 22}" y="{cy:.1f}" font-size="20" fill="{N50}" '
        f'text-anchor="middle" transform="rotate(-90 {VB_X + 22} {cy:.1f})">{y_title}</text>'
    )
    out.append(_text((PLOT_L + PLOT_R) / 2, VB_H - 18, x_title, 20, N50))
    return out


def _legend(items: list[tuple[str, str]]) -> str:
    x = PLOT_L
    out = []
    for label, color in items:
        out.append(f'<rect x="{x}" y="22" width="20" height="20" fill="{color}"/>')
        out.append(_text(x + 30, 39, label, 20, N50, "start"))
        x += 30 + 12 * len(label) + 40
    return "\n".join(out)


def _category_centers(n: int) -> tuple[list[float], float]:
    slot = (PLOT_R - PLOT_L) / n
    return [PLOT_L + slot * (i + 0.5) for i in range(n)], slot


def metric_chart(scores: Scores, metric: str, y_max: float = 0.5) -> str:
    """Grouped bars: one group per model, a dense and a hybrid bar in each."""
    y = Scale(0, y_max, PLOT_B, PLOT_T)
    ticks = [round(i * 0.1, 1) for i in range(round(y_max / 0.1) + 1)]
    out = _axes(y, ticks, metric, "embedding model", ".1f")
    out.append(_legend([(m, MODE_COLORS[m]) for m in MODES]))

    centers, slot = _category_centers(len(MODELS))
    bar_w, gap = slot * 0.27, slot * 0.04
    for cx, model in zip(centers, MODELS):
        for j, mode in enumerate(MODES):
            v = scores[mode][model][metric]
            x0 = cx - gap / 2 - bar_w if j == 0 else cx + gap / 2
            top = y(v)
            out.append(
                f'<rect x="{x0:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
                f'height="{PLOT_B - top:.1f}" fill="{MODE_COLORS[mode]}"/>'
            )
            out.append(_text(x0 + bar_w / 2, top - 12, f"{v:.3f}", 22, N30))
        out.append(_model_label(cx, model))
    return _page("\n".join(out))


def delta_chart(scores: Scores, metric: str = "ndcg@10") -> str:
    """One bar per model: hybrid minus dense, above or below zero."""
    deltas = {
        m: scores["hybrid"][m][metric] - scores["dense"][m][metric] for m in MODELS
    }
    lo, hi = -0.05, 0.10
    y = Scale(lo, hi, PLOT_B, PLOT_T)
    ticks = [-0.05, 0.0, 0.05, 0.10]
    out = _axes(y, ticks, f"hybrid minus dense, {metric}", "embedding model", "+.2f")

    centers, slot = _category_centers(len(MODELS))
    bar_w = slot * 0.4
    for cx, model in zip(centers, MODELS):
        d = deltas[model]
        top, bottom = sorted((y(d), y(0)))
        out.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{bottom - top:.1f}" fill="{TEAL if d >= 0 else ERROR}"/>'
        )
        label_y = top - 12 if d >= 0 else bottom + 28
        out.append(_text(cx, label_y, f"{d:+.3f}", 22, N30))
        out.append(_model_label(cx, model))
    return _page("\n".join(out))


def index_page(charts: list[tuple[str, str, str]]) -> str:
    """Landing page linking to every chart file (relative links)."""
    items = "\n".join(
        f'  <li><a href="{name}">{title}</a><span>{description}</span></li>'
        for name, title, description in charts
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FiQA charts</title>
<style>
  html, body {{ margin: 0; }}
  body {{ background: {WASH}; font-family: {FONT}; color: {N30}; }}
  .frame {{ padding: 48px; }}
  .canvas {{ background: {WHITE}; padding: 48px; max-width: 760px; }}
  h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: normal; color: {N30}; }}
  p {{ margin: 0 0 32px; font-size: 18px; color: {N60}; }}
  ul {{ list-style: none; margin: 0; padding: 0; }}
  li {{ display: flex; flex-direction: column; gap: 4px; padding: 16px 0; border-top: 1px solid {N90}; }}
  a {{ font-size: 22px; color: {BLUE}; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  span {{ font-size: 18px; color: {N50}; }}
</style>
</head>
<body>
<div class="frame"><div class="canvas">
<h1>FiQA charts</h1>
<p>Embedding models, dense vs hybrid search</p>
<ul>
{items}
</ul>
</div></div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plot-fiqa")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="charts")
    args = parser.parse_args(argv)

    scores = load_scores(Path(args.results_dir))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # (file name, link title, description, html)
    charts: list[tuple[str, str, str, str]] = [
        (
            f"fiqa_{m.replace('@', 'at')}.html",
            m,
            f"Dense vs hybrid {m} for each embedding model",
            metric_chart(scores, m),
        )
        for m in METRICS
    ]
    charts.append(
        (
            "fiqa_hybrid_gain_ndcgat10.html",
            "hybrid gain, ndcg@10",
            "Hybrid minus dense nDCG@10 for each embedding model",
            delta_chart(scores),
        )
    )
    for name, _, _, html in charts:
        if html.count("<svg") != html.count("</svg>") or html.count(
            "<text"
        ) != html.count("</text>"):
            raise RuntimeError(f"Unbalanced markup in {name}")
        (out / name).write_text(html)
        print(f"Wrote {out / name}")

    (out / "index.html").write_text(index_page([c[:3] for c in charts]))
    print(f"Wrote {out / 'index.html'}")
