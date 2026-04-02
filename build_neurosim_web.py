#!/usr/bin/env python3
"""Build standalone neurosim_web.html (inline JS + help from get_help_defs.py)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_help():
    spec = importlib.util.spec_from_file_location("get_help_defs", ROOT / "get_help_defs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defs = mod.get_defs()
    keys = [
        "jmr_defs", "jmr_defs2", "conways_defs", "how_network_works", "forward_pass",
        "how_backprop_works", "how_backprop_works2", "controls",
    ]
    return {k: d.strip() for k, d in zip(keys, defs)}


def main():
    help_obj = load_help()
    js_path = ROOT / "neurosim_web.js"
    js = js_path.read_text(encoding="utf-8")
    help_js = json.dumps(help_obj, ensure_ascii=False)
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
    #side {{ flex: 0 0 500px; width: 500px; min-width: 260px; max-width: 92vw; background: #e8e8e8; color: #111; display: flex; flex-direction: column; font: 13px/1.35 monospace; padding: 0; min-height: 0; height: 100vh; box-sizing: border-box; }}
    #helpScroll {{ flex: 1 1 60%; min-height: 40px; margin: 0; padding: 10px; overflow-y: auto; white-space: pre-wrap; font: 13px/1.45 system-ui, sans-serif; }}
    #sideSplitter {{ flex: 0 0 5px; height: 5px; cursor: row-resize; background: #999; border: solid #bbb; border-width: 1px 0; }}
    #statsScroll {{ flex: 1 1 40%; min-height: 40px; margin: 0; padding: 10px; overflow-y: auto; white-space: pre-wrap; font: 12px/1.35 monospace; background: #dde8dd; color: #111; }}
    #modalMask {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.55); align-items: center; justify-content: center; z-index: 100; }}
    #modalMask.on {{ display: flex; }}
    #modalBox {{ background: #fff; color: #111; padding: 16px; min-width: 320px; max-width: 90vw; }}
    #hud3d {{ position: fixed; left: 8px; bottom: 56px; background: rgba(0,0,0,0.65); color: #fff; font: 12px monospace; padding: 8px; z-index: 50; max-width: 90vw; display: none; white-space: pre-wrap; }}
    #filePick {{ display: none; }}
    .row {{ margin-top: 8px; }}
    button {{ margin-right: 6px; }}
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
          <label class="plotLab">Epoch Y-scale<input type="range" id="plotYEpoch" min="1" max="200" step="1" value="50" /></label>
          <label class="plotLab">Minibatch Y-scale<input type="range" id="plotYMinibatch" min="1" max="200" step="1" value="50" /></label>
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
      <pre id="helpScroll"></pre>
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
const HELP_SCREEN = {help_js};
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
