import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MODES = ("dense", "hybrid")
METRICS = ("ndcg@10", "mrr@10", "recall@10")

# Qdrant brand palette
WASH = "#E2E7F5"
WHITE = "#FFFFFF"
N30, N50, N60, N80, N90 = "#28324D", "#576280", "#717C99", "#B4BACC", "#D4D9E6"
BLUE, TEAL, ERROR = "#2F6FF0", "#038585", "#E02828"
MODE_COLORS = {"dense": BLUE, "hybrid": TEAL}

FONT = 'ui-monospace, "JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace'

# Canvas geometry. The viewBox starts left of 0 to leave room for the rotated y title.
VB_X, VB_Y, VB_W, VB_H = -70, 0, 1170, 620
PLOT_L, PLOT_R, PLOT_T, PLOT_B = 80, 1080, 70, 500

FIQA_DOCS = 57_638  # documents in the FiQA corpus, all embedded in every load


@dataclass(frozen=True)
class Dataset:
    key: str  # prefix of the chart files
    title: str
    models: dict[str, tuple[str, str]]  # file prefix -> two-line chart label
    eval_file: str  # format string over `mode` and `model`
    y_max: float  # shared y axis maximum of the metric charts


# The stella pipeline embeds documents with the full stella_en_400M_v5 transformer
# and queries with constella-zero, a static lookup table into stella's document space.
FIQA = Dataset(
    key="fiqa",
    title="FiQA",
    models={
        "8M": ("potion", "base-8M"),
        "32M": ("potion", "retrieval-32M"),
        "128M": ("potion", "multilingual-128M"),
        "constella": ("stella-400M-v5", "+ constella-zero"),
    },
    eval_file="{mode}_{model}_fiqa_evaluation.json",
    y_max=0.5,
)
MIRACL = Dataset(
    key="miracl",
    title="MIRACL English",
    models={
        "8M": ("potion", "base-8M"),
        "128M": ("potion", "multilingual-128M"),
    },
    eval_file="{mode}_{model}_eval.json",
    y_max=0.4,
)


@dataclass(frozen=True)
class Scale:
    """Map from a data range to pixel coordinates along an axis (linear or log10)."""

    d0: float
    d1: float
    p0: float
    p1: float
    log: bool = False

    def __call__(self, v: float) -> float:
        if self.log:
            v, d0, d1 = math.log10(v), math.log10(self.d0), math.log10(self.d1)
        else:
            d0, d1 = self.d0, self.d1
        return self.p0 + (v - d0) / (d1 - d0) * (self.p1 - self.p0)


Scores = dict[str, dict[str, dict[str, float]]]  # mode -> model -> metric -> mean


def load_scores(results_dir: Path, dataset: Dataset) -> Scores:
    """Mean metrics per mode and model. Runs without a results file are left out."""
    scores: Scores = {mode: {} for mode in MODES}
    for mode in MODES:
        for model in dataset.models:
            path = results_dir / dataset.eval_file.format(mode=mode, model=model)
            if not path.exists():
                print(f"No results for {dataset.key} {mode} {model}: {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            scores[mode][model] = {m: data[m]["mean"] for m in METRICS}
    if not scores["dense"]:
        raise FileNotFoundError(f"No dense results for {dataset.key} in {results_dir}")
    return scores


def load_times(path: Path) -> tuple[float, float]:
    """Total (embedding, upload) seconds of a `load` run."""
    with open(path) as f:
        data = json.load(f)
    return sum(data["embedding_time"]["per_batch"]), sum(
        data["upload_time"]["per_batch"]
    )


def _text(
    x: float, y: float, s: str, size: int, fill: str, anchor: str = "middle"
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}">{s}</text>'
    )


def _model_label(cx: float, label: tuple[str, str]) -> str:
    return _text(cx, PLOT_B + 36, label[0], 20, N50) + _text(
        cx, PLOT_B + 62, label[1], 20, N50
    )


def _page(title: str, svg_body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
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
    ticks: Sequence[float],
    y_title: str,
    x_title: str,
    fmt: str,
) -> list[str]:
    out = []
    for t in ticks:
        py = y(t)
        baseline = t == (1 if y.log else 0)
        out.append(
            f'<line x1="{PLOT_L}" y1="{py:.1f}" x2="{PLOT_R}" y2="{py:.1f}" '
            f'stroke="{N30 if baseline else N90}" stroke-width="{2 if baseline else 1}"/>'
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


def metric_chart(scores: Scores, metric: str, dataset: Dataset) -> str:
    """Grouped bars: one group per model, a dense and a hybrid bar in each."""
    y_max = dataset.y_max
    y = Scale(0, y_max, PLOT_B, PLOT_T)
    ticks = [round(i * 0.1, 1) for i in range(round(y_max / 0.1) + 1)]
    out = _axes(y, ticks, metric, "embedding model", ".1f")
    out.append(_legend([(m, MODE_COLORS[m]) for m in MODES]))

    centers, slot = _category_centers(len(dataset.models))
    bar_w, gap = slot * 0.27, slot * 0.04
    for cx, (model, label) in zip(centers, dataset.models.items()):
        for j, mode in enumerate(MODES):
            x0 = cx - gap / 2 - bar_w if j == 0 else cx + gap / 2
            if model not in scores[mode]:
                out.append(_text(x0 + bar_w / 2, PLOT_B - 12, "not run", 18, N60))
                continue
            v = scores[mode][model][metric]
            top = y(v)
            out.append(
                f'<rect x="{x0:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
                f'height="{PLOT_B - top:.1f}" fill="{MODE_COLORS[mode]}"/>'
            )
            out.append(_text(x0 + bar_w / 2, top - 12, f"{v:.3f}", 22, N30))
        out.append(_model_label(cx, label))
    return _page(f"{dataset.title} {metric}", "\n".join(out))


def delta_chart(scores: Scores, dataset: Dataset, metric: str = "ndcg@10") -> str:
    """One bar per model: hybrid minus dense, above or below zero."""
    models = {
        m: label
        for m, label in dataset.models.items()
        if m in scores["dense"] and m in scores["hybrid"]
    }
    deltas = {
        m: scores["hybrid"][m][metric] - scores["dense"][m][metric] for m in models
    }
    y = Scale(-0.05, 0.10, PLOT_B, PLOT_T)
    ticks = [-0.05, 0.0, 0.05, 0.10]
    out = _axes(y, ticks, f"hybrid minus dense, {metric}", "embedding model", "+.2f")

    centers, slot = _category_centers(len(models))
    bar_w = slot * 0.4
    for cx, (model, label) in zip(centers, models.items()):
        d = deltas[model]
        top, bottom = sorted((y(d), y(0)))
        out.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{bottom - top:.1f}" fill="{TEAL if d >= 0 else ERROR}"/>'
        )
        label_y = top - 12 if d >= 0 else bottom + 28
        out.append(_text(cx, label_y, f"{d:+.3f}", 22, N30))
        out.append(_model_label(cx, label))
    return _page(f"{dataset.title} hybrid gain", "\n".join(out))


def load_time_chart(times: dict[str, tuple[float, float]], dataset: Dataset) -> str:
    """Stacked columns, in minutes: embedding time on the bottom, upload time above."""
    to_min = {m: (e / 60, u / 60) for m, (e, u) in times.items()}
    y_max = 600
    y = Scale(0, y_max, PLOT_B, PLOT_T)
    out = _axes(
        y, range(0, y_max + 1, 100), "load time, minutes", "embedding model", ".0f"
    )
    out.append(_legend([("embedding", BLUE), ("upload to Qdrant", N80)]))

    centers, slot = _category_centers(len(dataset.models))
    bar_w = slot * 0.25
    for cx, (model, label) in zip(centers, dataset.models.items()):
        if model not in to_min:
            continue
        emb, upl = to_min[model]
        x0 = cx - bar_w / 2
        y_emb, y_total = y(emb), y(emb + upl)
        out.append(
            f'<rect x="{x0:.1f}" y="{y_emb:.1f}" width="{bar_w:.1f}" '
            f'height="{PLOT_B - y_emb:.1f}" fill="{BLUE}"/>'
        )
        out.append(
            f'<rect x="{x0:.1f}" y="{y_total:.1f}" width="{bar_w:.1f}" '
            f'height="{y_emb - y_total:.1f}" fill="{N80}"/>'
        )
        right = x0 + bar_w + 14
        out.append(
            _text(
                right,
                (PLOT_B + y_emb) / 2 + 7,
                f"{emb:.0f} min embedding",
                20,
                N30,
                "start",
            )
        )
        out.append(
            _text(
                right,
                (y_emb + y_total) / 2 + 7,
                f"{upl:.0f} min upload",
                20,
                N30,
                "start",
            )
        )
        out.append(_model_label(cx, label))
    return _page(f"{dataset.title} load time", "\n".join(out))


def throughput_chart(docs_per_s: dict[str, float], dataset: Dataset) -> str:
    """Embedding throughput per model on a log axis."""
    y = Scale(1, 100_000, PLOT_B, PLOT_T, log=True)
    ticks = [10**i for i in range(6)]
    out = _axes(
        y,
        ticks,
        "documents per second (log)",
        "embedding model",
        ",.0f",
    )

    centers, slot = _category_centers(len(dataset.models))
    bar_w = slot * 0.4
    for cx, (model, label) in zip(centers, dataset.models.items()):
        v = docs_per_s[model]
        top = y(v)
        out.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{PLOT_B - top:.1f}" fill="{BLUE}"/>'
        )
        out.append(
            _text(cx, top - 12, f"{v:,.1f}" if v < 100 else f"{v:,.0f}", 22, N30)
        )
        out.append(_model_label(cx, label))
    return _page(f"{dataset.title} embedding throughput", "\n".join(out))


Chart = tuple[str, str, str, str]  # file name, link title, description, html


def index_page(sections: list[tuple[str, list[tuple[str, str, str]]]]) -> str:
    """Landing page linking to every chart file (relative links), grouped by dataset."""
    body = "\n".join(
        f"<h2>{heading}</h2>\n<ul>\n"
        + "\n".join(
            f'  <li><a href="{name}">{title}</a><span>{description}</span></li>'
            for name, title, description in items
        )
        + "\n</ul>"
        for heading, items in sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Static embedding charts</title>
<style>
  html, body {{ margin: 0; }}
  body {{ background: {WASH}; font-family: {FONT}; color: {N30}; }}
  .frame {{ padding: 48px; }}
  .canvas {{ background: {WHITE}; padding: 48px; max-width: 760px; }}
  h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: normal; color: {N30}; }}
  h2 {{ margin: 40px 0 8px; font-size: 22px; font-weight: normal; color: {N50}; }}
  p {{ margin: 0 0 8px; font-size: 18px; color: {N60}; }}
  ul {{ list-style: none; margin: 0; padding: 0; }}
  li {{ display: flex; flex-direction: column; gap: 4px; padding: 16px 0; border-top: 1px solid {N90}; }}
  a {{ font-size: 22px; color: {BLUE}; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  span {{ font-size: 18px; color: {N50}; }}
</style>
</head>
<body>
<div class="frame"><div class="canvas">
<h1>Static embedding charts</h1>
<p>Embedding models, dense vs hybrid search</p>
{body}
</div></div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plot-results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="charts")
    args = parser.parse_args(argv)

    results = Path(args.results_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sections: list[tuple[str, list[Chart]]] = []

    # FiQA: quality, hybrid gain, and embedding throughput per model
    fiqa_scores = load_scores(results, FIQA)
    fiqa: list[Chart] = [
        (
            f"fiqa_{m.replace('@', 'at')}.html",
            m,
            f"Dense vs hybrid {m} for each embedding model",
            metric_chart(fiqa_scores, m, FIQA),
        )
        for m in METRICS
    ]
    fiqa.append(
        (
            "fiqa_hybrid_gain_ndcgat10.html",
            "hybrid gain, ndcg@10",
            "Hybrid minus dense nDCG@10 for each embedding model",
            delta_chart(fiqa_scores, FIQA),
        )
    )
    throughput = {
        m: FIQA_DOCS / load_times(results / f"dense_{m}_fiqa_load_timings.json")[0]
        for m in FIQA.models
    }
    fiqa.append(
        (
            "fiqa_embedding_throughput.html",
            "embedding throughput",
            "Documents embedded per second while loading FiQA",
            throughput_chart(throughput, FIQA),
        )
    )
    sections.append(("FiQA", fiqa))

    # MIRACL: quality per model, plus where the load time goes
    miracl_scores = load_scores(results, MIRACL)
    miracl: list[Chart] = [
        (
            f"miracl_{m.replace('@', 'at')}.html",
            m,
            f"Dense vs hybrid {m} for each embedding model",
            metric_chart(miracl_scores, m, MIRACL),
        )
        for m in METRICS
    ]
    miracl_times = {
        m: load_times(results / f"dense_{m}_load_timings.json") for m in MIRACL.models
    }
    miracl.append(
        (
            "miracl_load_time.html",
            "load time",
            "Embedding time and upload time for the full corpus",
            load_time_chart(miracl_times, MIRACL),
        )
    )
    sections.append(("MIRACL English", miracl))

    for _, charts in sections:
        for name, _, _, html in charts:
            if html.count("<svg") != html.count("</svg>") or html.count(
                "<text"
            ) != html.count("</text>"):
                raise RuntimeError(f"Unbalanced markup in {name}")
            (out / name).write_text(html)
            print(f"Wrote {out / name}")

    (out / "index.html").write_text(
        index_page([(h, [c[:3] for c in charts]) for h, charts in sections])
    )
    print(f"Wrote {out / 'index.html'}")
