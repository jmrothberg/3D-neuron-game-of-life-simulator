#!/usr/bin/env python3
"""Build standalone neurosim_web.html (inline JS + README.md rendered as help HTML)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "README.md"


def load_readme_html() -> str:
    """Convert README.md to HTML (tables, fenced code, TOC ids on headings for H-key nav)."""
    text = README_PATH.read_text(encoding="utf-8")
    try:
        import markdown
    except ImportError:
        print(
            "ERROR: package 'markdown' is required to build formatted help from README.md.\n"
            "  python3 -m pip install markdown\n"
            "Falling back to escaped plain text (no tables/rendering).",
            file=sys.stderr,
        )
        import html as html_mod

        return f'<pre class="readme-fallback">{html_mod.escape(text)}</pre>'

    html = markdown.markdown(
        text,
        extensions=["toc", "tables", "fenced_code"],
    )
    # Open external links in a new tab when the app is served over http(s).
    html = re.sub(
        r'<a href="(https?://[^"#]+)"',
        r'<a target="_blank" rel="noopener noreferrer" href="\1"',
        html,
    )
    return html


def main():
    readme_html = load_readme_html()
    js_path = ROOT / "neurosim_web.js"
    js = js_path.read_text(encoding="utf-8")
    readme_js = json.dumps(readme_html, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JMR Neural Game of Life (Web)</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #222; color: #eee; }}
    #wrap {{ display: flex; flex-direction: row; height: 100vh; overflow: hidden; }}
    #mainCol {{ flex: 1; min-width: 0; min-height: 0; display: flex; flex-direction: column; }}
    /* Scroll inside this wrapper so the full 1008² grid is reachable; canvas may scale down on narrow screens */
    #canvasWrap {{ flex: 1 1 auto; min-height: 0; overflow: auto; background: #333; }}
    #cv2d, #cv3d {{ display: block; background: #fff; touch-action: none; width: 100%; max-width: 1008px; height: auto; aspect-ratio: 1 / 1; }}
    #cv3d {{ cursor: grab; }}
    #plotStrip {{ flex: 0 0 auto; display: flex; flex-direction: column; gap: 2px; min-height: 88px; height: min(18vh, 200px); max-height: 24vh; background: #000; padding: 3px 8px; min-width: 0; }}
    #plotControls {{ flex: 0 0 auto; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; color: #9bd; font: 11px system-ui, sans-serif; padding: 0 0 2px 0; }}
    #plotControls .plotLab input {{ width: 132px; vertical-align: middle; margin-left: 4px; }}
    #predBatchWrap {{ flex: 1 1 0; min-height: 32px; min-width: 0; width: 100%; max-width: 1008px; }}
    #predBatch {{ display: block; width: 100%; height: 100%; pointer-events: none; vertical-align: top; }}
    #predWrap {{ flex: 1 1 0; min-height: 32px; min-width: 0; width: 100%; max-width: 1008px; }}
    #pred {{ display: block; width: 100%; height: 100%; vertical-align: top; pointer-events: none; }}
    #statusDock {{ flex: 0 0 auto; background: #000; color: #cfc; font: 12px/1.3 monospace; padding: 6px 8px; border-top: 1px solid #333; }}
    #statusLines {{ line-height: 1.25; }}
    #colSplitter {{ flex: 0 0 6px; width: 6px; cursor: col-resize; background: #444; border: solid #222; border-width: 0 1px; }}
    #side {{ flex: 0 0 500px; width: 500px; min-width: 260px; max-width: 92vw; background: #e8e8e8; color: #111; display: flex; flex-direction: column; font: 13px/1.35 system-ui, sans-serif; padding: 0; min-height: 0; height: 100vh; box-sizing: border-box; }}
    /* Help = README.md preview (markdown HTML) + session log */
    #helpScroll {{ flex: 1 1 60%; min-height: 40px; margin: 0; padding: 10px 12px; overflow-y: auto; overflow-x: auto; white-space: normal; color: #111; }}
    #sideSplitter {{ flex: 0 0 5px; height: 5px; cursor: row-resize; background: #999; border: solid #bbb; border-width: 1px 0; }}
    #statsScroll {{ flex: 1 1 40%; min-height: 40px; margin: 0; padding: 10px; overflow-y: auto; white-space: pre-wrap; font: 12px/1.35 monospace; background: #dde8dd; color: #111; }}
    #modalMask {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.55); align-items: center; justify-content: center; z-index: 100; }}
    #modalMask.on {{ display: flex; }}
    #modalBox {{ background: #fff; color: #111; padding: 16px; min-width: 320px; max-width: 90vw; }}
    #hud3d {{ position: fixed; left: 8px; bottom: 56px; background: rgba(0,0,0,0.65); color: #fff; font: 12px monospace; padding: 8px; z-index: 50; max-width: 90vw; display: none; white-space: pre-wrap; }}
    #filePick {{ display: none; }}
    .row {{ margin-top: 8px; }}
    button {{ margin-right: 6px; }}
    /* README markdown preview */
    #helpScroll .quick-start {{ background: #e8f2fc; padding: 10px 12px; border-radius: 6px; margin-bottom: 14px; border: 1px solid #9bd; }}
    #helpScroll .quick-start .quick-h {{ margin: 0 0 6px 0; font-size: 1rem; color: #024; }}
    #helpScroll .quick-start .quick-pre {{ margin: 0; font: 12px/1.4 ui-monospace, monospace; white-space: pre-wrap; word-break: break-word; }}
    #helpScroll .readme-body {{ max-width: 100%; line-height: 1.45; }}
    #helpScroll .readme-body h1 {{ font-size: 1.28rem; margin: 0.9em 0 0.35em; padding-bottom: 4px; border-bottom: 2px solid #89a; color: #012; }}
    #helpScroll .readme-body h2 {{ font-size: 1.08rem; margin: 1em 0 0.3em; color: #023; scroll-margin-top: 6px; }}
    #helpScroll .readme-body h3 {{ font-size: 0.98rem; margin: 0.85em 0 0.25em; color: #034; }}
    #helpScroll .readme-body h4 {{ font-size: 0.92rem; margin: 0.75em 0 0.2em; }}
    #helpScroll .readme-body p {{ margin: 0.45em 0; }}
    #helpScroll .readme-body ul, #helpScroll .readme-body ol {{ margin: 0.4em 0; padding-left: 1.35em; }}
    #helpScroll .readme-body li {{ margin: 0.2em 0; }}
    #helpScroll .readme-body hr {{ border: none; border-top: 1px solid #bbb; margin: 1em 0; }}
    #helpScroll .readme-body blockquote {{ margin: 0.5em 0; padding: 4px 10px; border-left: 3px solid #89a; background: #f4f6f8; color: #222; }}
    #helpScroll .readme-body table {{ border-collapse: collapse; width: 100%; font-size: 11px; margin: 0.65em 0; max-width: 100%; }}
    #helpScroll .readme-body thead {{ background: #dde8f0; }}
    #helpScroll .readme-body th, #helpScroll .readme-body td {{ border: 1px solid #aab; padding: 4px 6px; vertical-align: top; text-align: left; }}
    #helpScroll .readme-body tr:nth-child(even) {{ background: rgba(0,0,0,0.03); }}
    #helpScroll .readme-body pre {{ overflow-x: auto; background: #f0f2f5; padding: 8px 10px; border-radius: 4px; font: 11px/1.28 ui-monospace, monospace; margin: 0.6em 0; border: 1px solid #ccd; white-space: pre; }}
    #helpScroll .readme-body code {{ background: #eaecef; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; font-family: ui-monospace, monospace; }}
    #helpScroll .readme-body pre code {{ background: none; padding: 0; font-size: inherit; }}
    #helpScroll .readme-body a {{ color: #06c; }}
    #helpScroll .readme-body strong {{ font-weight: 600; }}
    #helpScroll .readme-fallback {{ white-space: pre-wrap; font: 12px ui-monospace, monospace; }}
    #helpScroll .help-sep {{ border: none; border-top: 2px solid #999; margin: 16px 0 10px; }}
    #helpScroll .help-log-h {{ font-size: 0.95rem; margin: 0 0 6px 0; color: #222; }}
    #helpScroll .help-log {{ white-space: pre-wrap; font: 11px ui-monospace, monospace; margin: 0; padding: 8px; background: #fafafa; border: 1px solid #ccc; border-radius: 4px; max-height: min(28vh, 220px); overflow-y: auto; color: #111; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="mainCol">
      <div id="canvasWrap">
        <canvas id="cv2d" width="1008" height="1008" tabindex="0" title="Click for keyboard focus. If the square looks small, scroll inside this area — mouse is mapped to the full grid."></canvas>
        <canvas id="cv3d" width="1008" height="1008" style="display:none" tabindex="0"></canvas>
      </div>
      <div id="plotStrip">
        <div id="plotControls">
          <label class="plotLab">Epoch Y-scale<input type="range" id="plotYEpoch" min="0.1" max="20" step="0.1" value="10" /></label>
          <label class="plotLab">Minibatch Y-scale<input type="range" id="plotYMinibatch" min="0.1" max="20" step="0.1" value="10" /></label>
        </div>
        <div id="predBatchWrap" title="Mean loss per full epoch — one pass over all training samples (green)">
          <canvas id="predBatch" width="1008" height="72"></canvas>
        </div>
        <div id="predWrap" title="Mean loss per gradient minibatch (K key); one point per weight update (blue)">
          <canvas id="pred" width="1008" height="72"></canvas>
        </div>
      </div>
      <div id="statusDock">
        <div id="statusLines"></div>
      </div>
    </div>
    <div id="colSplitter" title="Drag to resize panels"></div>
    <div id="side">
      <div id="helpScroll"></div>
      <div id="sideSplitter" title="Drag to resize help / stats"></div>
      <pre id="statsScroll"></pre>
    </div>
  </div>
  <div id="hud3d"></div>
  <div id="modalMask">
    <div id="modalBox">
      <div id="modalPrompt"></div>
      <input id="modalInput" type="text" style="width:100%;margin-top:8px" />
      <div class="row">
        <button id="modalOk">OK</button>
        <button id="modalCancel">Cancel</button>
      </div>
    </div>
  </div>
  <input id="filePick" type="file" accept=".json,application/json" />
<script>
const README_HTML = {readme_js};
</script>
<script>
{js}
</script>
</body>
</html>
"""
    out = ROOT / "neurosim_web.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
