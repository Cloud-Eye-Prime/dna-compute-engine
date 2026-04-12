"""
compare_viewer.py — Generates an HTML matrix comparison grid from rendered variants.

Usage:
    python compare_viewer.py --report ~/renders/physics_matrix/render_report.json
    python compare_viewer.py --report render_report.json --frame 24
"""
import json, pathlib, argparse, base64, glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--frame",  type=int, default=1,
                   help="Which frame to show per variant (1-indexed)")
    p.add_argument("--output", default="matrix_compare.html")
    return p.parse_args()


def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def pick_frame(frames, n):
    if not frames: return None
    idx = min(n - 1, len(frames) - 1)
    return frames[idx]


def main():
    args = parse_args()
    report = json.loads(pathlib.Path(args.report).read_text())
    variants = report.get("variants", {})
    physics_type = report.get("physics_type", "unknown")

    cards = []
    for vid, info in sorted(variants.items()):
        frames = info.get("frames", [])
        frame_path = pick_frame(frames, args.frame)
        ok = info.get("ok", False)
        label = info.get("quality_label", vid)
        desc  = info.get("description", "")
        elapsed = info.get("elapsed", 0)
        params = info.get("param_values", {})

        if frame_path and pathlib.Path(frame_path).exists():
            b64 = img_to_b64(frame_path)
            img_tag = f'<img src="data:image/png;base64,{b64}" alt="{label}" loading="lazy">'
        else:
            img_tag = '<div class="no-frame">No frame rendered</div>'

        params_html = "".join(
            f'<span class="param">{k}: <b>{v}</b></span>' for k, v in params.items()
        ) if params else ""

        status_cls = "ok" if ok else "fail"
        cards.append(f"""
        <div class="card {status_cls}">
          <div class="card-img">{img_tag}</div>
          <div class="card-info">
            <div class="vid">{vid}</div>
            <div class="label">{label}</div>
            <div class="desc">{desc}</div>
            <div class="params">{params_html}</div>
            <div class="timing">{elapsed:.1f}s</div>
          </div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Physics Matrix — {physics_type}</title>
<style>
  :root {{
    --bg: #0e0e10; --surface: #1a1a1e; --border: #2a2a30;
    --text: #e0dfd8; --muted: #888; --accent: #4f98a3;
    --ok: #4a8c3f; --fail: #8c3f3f;
    --font: 'Segoe UI', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font);
         min-height: 100vh; padding: 2rem; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.4rem; color: var(--accent); }}
  .meta {{ color: var(--muted); font-size: 0.8rem; margin-bottom: 2rem; }}
  .grid {{ display: grid;
           grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
           gap: 1.2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 8px; overflow: hidden;
           transition: transform 0.15s; }}
  .card:hover {{ transform: translateY(-3px); }}
  .card.ok  {{ border-top: 3px solid var(--ok); }}
  .card.fail {{ border-top: 3px solid var(--fail); }}
  .card-img img {{ width: 100%; display: block; }}
  .no-frame {{ height: 160px; display: flex; align-items: center;
               justify-content: center; background: #111; color: var(--muted);
               font-size: 0.85rem; }}
  .card-info {{ padding: 0.75rem; }}
  .vid   {{ font-size: 0.7rem; color: var(--muted); margin-bottom: 0.2rem; }}
  .label {{ font-size: 0.95rem; font-weight: 600; margin-bottom: 0.3rem; }}
  .desc  {{ font-size: 0.78rem; color: var(--muted); margin-bottom: 0.5rem;
            line-height: 1.4; }}
  .params {{ display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.4rem; }}
  .param {{ background: #22252a; border-radius: 4px; padding: 2px 6px;
            font-size: 0.68rem; color: #aaa; }}
  .param b {{ color: var(--accent); }}
  .timing {{ font-size: 0.7rem; color: var(--muted); text-align: right; }}
</style>
</head>
<body>
<h1>Physics Matrix — {physics_type}</h1>
<div class="meta">
  {len(cards)} variants  ·  Frame {args.frame}  ·  {report.get("matrix_file","")}
</div>
<div class="grid">
{"".join(cards)}
</div>
</body></html>"""

    out = pathlib.Path(args.output)
    out.write_text(html)
    print(f"[compare] Viewer -> {out}  ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
