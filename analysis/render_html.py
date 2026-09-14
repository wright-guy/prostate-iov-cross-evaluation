"""Render deliverables/dossier.json to a single self-contained HTML page.

    python analysis/render_html.py

Figures are embedded as data URIs so the page stands alone. Design language matches the
project's earlier plan artifact so the two read as one document set.
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import REPO_ROOT  # noqa: E402

CSS = """
:root{
  --ground:#f2f5f6; --surface:#fff; --surface-2:#e9eef0;
  --ink:#121a1d; --ink-2:#3a4a4f; --muted:#617277;
  --rule:#d3dcde; --rule-2:#bfcbce;
  --accent:#0c666d; --accent-soft:#e0eef0;
  --gold:#8f5e00; --gold-soft:#f9eed7;
  --f-display:"Archivo","Helvetica Neue",Arial,sans-serif;
  --f-body:"Source Serif 4",Georgia,"Times New Roman",serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  --measure:40rem; --wide:70rem;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0c1214; --surface:#121a1d; --surface-2:#182226;
  --ink:#e4ebec; --ink-2:#b9c7ca; --muted:#8b9ca1;
  --rule:#243033; --rule-2:#31403f;
  --accent:#54b9c0; --accent-soft:#12292c;
  --gold:#d9a445; --gold-soft:#2b2213;
}}
:root[data-theme="dark"]{
  --ground:#0c1214; --surface:#121a1d; --surface-2:#182226;
  --ink:#e4ebec; --ink-2:#b9c7ca; --muted:#8b9ca1;
  --rule:#243033; --rule-2:#31403f;
  --accent:#54b9c0; --accent-soft:#12292c;
  --gold:#d9a445; --gold-soft:#2b2213;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--f-body);
  font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:var(--wide);margin:0 auto;padding:0 1.5rem 6rem}
.mast{border-bottom:1px solid var(--rule);padding:3.5rem 0 2rem;
  display:flex;flex-direction:column;gap:1rem}
.eyebrow{font-family:var(--f-mono);font-size:.7rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent)}
h1{font-family:var(--f-display);font-weight:700;font-size:clamp(1.9rem,4.4vw,2.8rem);
  line-height:1.06;letter-spacing:-.02em;margin:0;text-wrap:balance}
.standfirst{font-size:1.1rem;color:var(--ink-2);max-width:42rem;margin:0}
.stamp{font-family:var(--f-mono);font-size:.74rem;color:var(--muted);
  display:flex;flex-wrap:wrap;gap:.3rem 1.4rem}
nav.toc{position:sticky;top:0;z-index:20;border-bottom:1px solid var(--rule);
  background:color-mix(in srgb,var(--ground) 92%,transparent);
  backdrop-filter:blur(8px);margin-bottom:2.5rem}
nav.toc ul{max-width:var(--wide);margin:0 auto;padding:.65rem 1.5rem;list-style:none;
  display:flex;gap:1.35rem;overflow-x:auto}
nav.toc a{font-family:var(--f-mono);font-size:.73rem;letter-spacing:.04em;
  text-transform:uppercase;color:var(--muted);text-decoration:none;white-space:nowrap;
  padding:.15rem 0;border-bottom:2px solid transparent}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--accent);border-bottom-color:var(--accent)}
section{padding:2.4rem 0 1rem;border-top:1px solid var(--rule)}
section:first-of-type{border-top:none}
h2{font-family:var(--f-display);font-weight:600;font-size:1.7rem;letter-spacing:-.014em;
  margin:0 0 .3rem;text-wrap:balance;color:var(--accent)}
h3{font-family:var(--f-display);font-weight:600;font-size:1.12rem;margin:2.4rem 0 .6rem}
.lead{color:var(--ink-2);font-style:italic;margin:.2rem 0 1.6rem;max-width:var(--measure)}
p{margin:0 0 1rem;max-width:var(--measure)}
.tag{font-family:var(--f-mono);font-size:.63rem;font-weight:600;letter-spacing:.11em;
  text-transform:uppercase;display:block;margin:1rem 0 .3rem}
.tag.result{color:var(--accent)} .tag.means{color:var(--gold)} .tag.caveat{color:var(--muted)}
.result-text{font-size:1.05rem}
.means-text{color:var(--ink-2)}
.caveat-text{color:var(--muted);font-size:.95rem;border-left:2px solid var(--rule-2);
  padding-left:.8rem}
.scroll{overflow-x:auto;margin:1.2rem 0;border:1px solid var(--rule);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;padding:.55rem .8rem;border-bottom:1px solid var(--rule);
  vertical-align:top}
thead th{font-family:var(--f-mono);font-size:.66rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);font-weight:600;background:var(--surface-2);
  white-space:nowrap;border-bottom:1px solid var(--rule-2)}
tbody tr:last-child td{border-bottom:none}
td.num{text-align:right;font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.cap{font-size:.85rem;color:var(--muted);margin:.3rem 0 1.2rem}
figure{margin:1.4rem 0}
figure img{width:100%;height:auto;border:1px solid var(--rule);background:#fff}
figcaption{font-size:.85rem;color:var(--muted);margin-top:.5rem}
.source{font-family:var(--f-mono);font-size:.68rem;color:var(--muted);
  border-top:1px solid var(--rule);padding-top:.5rem;margin-top:1.2rem}
code{font-family:var(--f-mono);font-size:.86em;background:var(--surface-2);
  padding:.08em .34em;border-radius:2px}
footer{border-top:1px solid var(--rule);margin-top:3rem;padding-top:1.4rem;
  font-family:var(--f-mono);font-size:.72rem;color:var(--muted)}
@media (max-width:640px){body{font-size:16px}}
"""


def esc(s: str | None) -> str:
    return html.escape(str(s)) if s is not None else ""


def fmt(v) -> str:  # noqa: ANN001
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.3g}"
    return esc(v)


def render_table(tbl: dict) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in tbl["columns"])
    rows = []
    for r in tbl["rows"]:
        cells = "".join(
            f'<td class="num">{fmt(v)}</td>' if isinstance(v, (int, float))
            else f"<td>{fmt(v)}</td>" for v in r)
        rows.append(f"<tr>{cells}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def render_figure(name: str, caption: str | None) -> str:
    path = REPO_ROOT / "figures" / name
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    cap = f"<figcaption>{esc(caption)}</figcaption>" if caption else ""
    return (f'<figure><img src="data:image/png;base64,{b64}" '
            f'alt="{esc(caption or name)}">{cap}</figure>')


def main() -> int:
    data = json.loads((REPO_ROOT / "deliverables" / "dossier.json").read_text("utf-8"))
    slug = {s["title"]: f"s{i}" for i, s in enumerate(data["sections"])}

    nav = "".join(
        f'<li><a href="#{slug[s["title"]]}">'
        f'{esc(s["title"].split("—")[-1].strip())}</a></li>'
        for s in data["sections"])

    body = []
    for section in data["sections"]:
        parts = [f'<section id="{slug[section["title"]]}">',
                 f'<h2>{esc(section["title"])}</h2>',
                 f'<p class="lead">{esc(section["lead"])}</p>']
        for b in section["blocks"]:
            parts.append(f'<h3>{esc(b["heading"])}</h3>')
            parts.append('<span class="tag result">Result</span>')
            parts.append(f'<p class="result-text">{esc(b["statement"])}</p>')
            parts.append('<span class="tag means">What it means</span>')
            parts.append(f'<p class="means-text">{esc(b["interpretation"])}</p>')
            if b.get("caveat"):
                parts.append('<span class="tag caveat">Caveat</span>')
                parts.append(f'<p class="caveat-text">{esc(b["caveat"])}</p>')
            if b.get("table"):
                parts.append(render_table(b["table"]))
                if b.get("table_caption"):
                    parts.append(f'<p class="cap">{esc(b["table_caption"])}'
                                 + (f' · <code>deliverables/tables/{esc(b["table_file"])}'
                                    "</code>" if b.get("table_file") else "") + "</p>")
            if b.get("figure"):
                parts.append(render_figure(b["figure"], b.get("figure_caption")))
            if b.get("source"):
                parts.append(f'<p class="source">Source: {esc(b["source"])}</p>')
        parts.append("</section>")
        body.append("\n".join(parts))

    page = f"""<title>Contour Cross-Evaluation Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}</style>

<header class="mast wrap">
  <span class="eyebrow">Results dossier · all seven hypotheses tested</span>
  <h1>{esc(data["title"])}</h1>
  <p class="standfirst">Every number carries the pipeline stage and results table that
  produced it. Interpretation is kept separate from measurement, so the argument in the
  paper stays yours.</p>
  <div class="stamp">
    <span>commit {esc(data["git_sha"])}</span>
    <span>config {esc(data["config_hash"])}</span>
    <span>generated {esc(data["generated_utc"])}</span>
  </div>
</header>

<nav class="toc"><ul>{nav}</ul></nav>

<main class="wrap">
{"".join(body)}
</main>

<footer class="wrap">
  Rendered from <code>deliverables/dossier.json</code>, itself built from
  <code>results/*.parquet</code>. Citable numbers are keyed in
  <code>results/quotable.csv</code>; the Word rendering is
  <code>deliverables/results_dossier.docx</code>.
</footer>
"""
    out = REPO_ROOT / "deliverables" / "results_dossier.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
