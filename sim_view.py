"""
sim_view.py - animated canvas that replays a recorded episode.

The simulator records every vehicle's passage over every link as
(vehicle type, link, time entered, time reached the stop line, time departed).
The browser interpolates positions from those timestamps, stacks queued vehicles
behind the stop line, colours the signal heads from the phase log, and draws
pedestrians waiting at each junction. Two recordings with the same random seed can
be shown side by side on one clock (e.g. Fixed-time vs Q-learning). Each view carries live
counters (cars queued, queue delay and CO2 accumulated so far, pedestrians waiting) and
colours every approach by its current queue per lane.
"""
from __future__ import annotations

import json

TEMPLATE = r"""
<!doctype html>
<html><head><meta charset="utf-8">
<style>
  :root { --bg:#fcfcfb; --panel:#f2f1ec; --ink:#0b0b0b; --muted:#52514e; --line:#d9d7cf;
          --chrome:196px; }
  html, body { height:100%; }
  body { margin:0; padding:0 4px; box-sizing:border-box; overflow:hidden;
         font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg);
         color:var(--ink); }
  .bar { display:flex; gap:12px; align-items:center; flex-wrap:nowrap; padding:4px 2px 8px 2px; }
  button { border:1px solid var(--line); background:#fff; padding:7px 16px; border-radius:8px;
           font-size:16px; cursor:pointer; white-space:nowrap; }
  button.primary { background:#2a78d6; color:#fff; border-color:#2a78d6; }
  select { padding:5px 6px; border-radius:6px; border:1px solid var(--line); font-size:15px; }
  input[type=range] { flex:1; min-width:120px; accent-color:#2a78d6; }
  .clock { font-variant-numeric: tabular-nums; font-weight:700; font-size:20px; min-width:70px; }
  .wrap { display:flex; gap:14px; flex-wrap:nowrap; }
  .cell { flex:1 1 0; min-width:0; display:flex; flex-direction:column; align-items:center; }
  .title { font-weight:700; font-size:17px; margin:0 0 4px 0; align-self:flex-start; }
  .kpis { display:flex; gap:6px; width:min(100%, calc(100vh - var(--chrome))); margin-bottom:5px; }
  .kpi { flex:1 1 0; min-width:0; background:var(--panel); border-radius:8px; padding:4px 8px 3px 8px; }
  .kpi .v { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.15; }
  .kpi .l { font-size:11.5px; color:var(--muted); line-height:1.2; }
  .kpi .d { font-size:11.5px; font-weight:600; min-height:14px; line-height:1.2; white-space:nowrap; }
  .good { color:#16865f; } .bad { color:#c23b3b; }
  canvas { width:min(100%, calc(100vh - var(--chrome))); height:auto; aspect-ratio:1/1;
           background:#e9e7df; border-radius:8px; display:block; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12.5px; color:var(--muted);
            margin-top:6px; justify-content:center; }
  .sw { display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:4px;
        vertical-align:-1px; }
  .grad { display:inline-block; width:54px; height:10px; border-radius:2px; margin:0 4px;
          vertical-align:-1px;
          background:linear-gradient(90deg, hsl(125,65%,42%), hsl(60,80%,45%), hsl(0,80%,45%)); }
</style></head>
<body>
<div class="bar">
  <button id="play" class="primary">&#9654; Play</button>
  <button id="reset">&#8634; Reset</button>
  <label style="font-size:13px;color:#52514e">Speed
    <select id="speed">
      <option value="1">1x</option><option value="4">4x</option>
      <option value="10" selected>10x</option><option value="30">30x</option>
      <option value="60">60x</option>
    </select></label>
  <input id="scrub" type="range" min="0" max="900" step="1" value="0">
  <span class="clock" id="clock">00:00</span>
</div>
<div class="wrap" id="wrap"></div>
<div class="legend">
  <span>Road colour = queue per lane <span class="grad"></span>short &rarr; long</span>
  <span><span class="sw" style="background:#cfd3d9;outline:1px solid #8a887f"></span>Car</span>
  <span><span class="sw" style="background:#19c2b4"></span>Autonomous vehicle</span>
  <span><span class="sw" style="background:#c48a45"></span>Truck</span>
  <span><span class="sw" style="background:#f5ab25"></span>Bus</span>
  <span><span class="sw" style="background:#ff4d4f"></span>Emergency vehicle</span>
  <span><span class="sw" style="background:#7b4fd6"></span>Pedestrians waiting</span>
  <span><span class="sw" style="background:#22a447"></span>/<span class="sw" style="background:#f2c14e;margin-left:4px"></span>/<span class="sw" style="background:#d64545;margin-left:4px"></span>Signal</span>
</div>
<script>
const PAYLOADS = __PAYLOADS__;
const NAMES = __NAMES__;
const HV = [[0,-1],[1,0],[0,1],[-1,0]];
const COLORS = ["#cfd3d9", "#19c2b4", "#c48a45", "#f5ab25", "#ff4d4f"];
const SIZE = 640;                 // canvas pixels (square)
const LW = 4.2;                   // lateral pixels per lane (exaggerated)
// per vehicle type (car, AV, truck, bus, emergency): config.EMISSION  (g CO2 per km / per s idle / per stop)
const EM_KM = [249, 237, 900, 1100, 400], EM_IDLE = [0.74, 0.74, 2.8, 2.8, 1.2],
      EM_STOP = [41, 25, 250, 290, 60];
let t = 0, playing = false, speed = 10, last = null;
const tMax = Math.min(...PAYLOADS.map(p => p.t_max));
document.getElementById("scrub").max = Math.floor(tMax);

function prep(P) {
  const B = P.block, E = P.entry, W = 2 * E + 2 * B;
  const s = SIZE / W;
  const segs = Array.from({length: 36}, () => []);
  let id = 0;
  for (const g of P.segments) {
    segs[g[1]].push({vt: g[0], te: g[2], ta: g[3], td: g[4], id: id++});
  }
  const maxDur = [];
  for (let L = 0; L < 36; L++) {
    segs[L].sort((a, b) => a.te - b.te);
    let m = 0; for (const g of segs[L]) m = Math.max(m, g.td - g.te);
    maxDur.push(Math.min(m, 3600));
  }
  const streetLanes = r => (r === 1 ? 3 : 2);
  const geom = [];
  for (let L = 0; L < 36; L++) {
    const i = Math.floor(L / 4), a = L % 4, h = (a + 2) % 4;
    const r = Math.floor(i / 3), c = i % 3;
    const cx = (E + c * B) * s, cy = (E + r * B) * s;
    const d = HV[h];
    const horiz = (h === 1 || h === 3);
    const half = horiz ? 2 * LW : streetLanes(r) * LW;      // half width of crossing road
    const stop = [cx - d[0] * (half + 2), cy - d[1] * (half + 2)];
    let start;
    if (P.up[L] >= 0) {
      const ui = P.up[L], ur = Math.floor(ui / 3), uc = ui % 3;
      const ux = (E + uc * B) * s, uy = (E + ur * B) * s;
      const uhalf = horiz ? 2 * LW : streetLanes(ur) * LW;
      start = [ux + d[0] * (uhalf + 2), uy + d[1] * (uhalf + 2)];
    } else {
      start = [cx - d[0] * E * s, cy - d[1] * E * s];
    }
    const right = [-d[1], d[0]];
    geom.push({i, a, h, d, right, stop, start, lanes: P.lanes[L],
               len: Math.hypot(stop[0] - start[0], stop[1] - start[1])});
  }
  // flat copies of the segments for the cumulative counters (emergency vehicles excluded,
  // as in the dashboard KPIs)
  const n = P.segments.length;
  const flat = {n, vt: new Uint8Array(n), te: new Float32Array(n), ta: new Float32Array(n),
                td: new Float32Array(n), km: new Float32Array(n)};
  for (let k = 0; k < n; k++) {
    const g = P.segments[k];
    flat.vt[k] = g[0]; flat.te[k] = g[2]; flat.ta[k] = g[3]; flat.td[k] = g[4];
    flat.km[k] = geom[g[1]].len / s / 1000;
  }
  return {P, s, segs, maxDur, geom, B, E, flat};
}

function counters(S) {
  const F = S.flat;
  let delay = 0, co2 = 0;
  for (let k = 0; k < F.n; k++) {
    const vt = F.vt[k];
    if (vt === 4) continue;
    const te = F.te[k];
    if (t <= te) continue;
    const ta = F.ta[k], td = F.td[k];
    co2 += EM_KM[vt] * F.km[k] * Math.min(1, (t - te) / Math.max(0.1, ta - te));
    if (t > ta) {
      const w = Math.min(t, td) - ta;
      if (w > 0) { delay += w; co2 += EM_IDLE[vt] * w; }
      if (td - ta > 1) co2 += EM_STOP[vt];
    }
  }
  return {delay: delay / 60, co2: co2 / 1000};
}

function lowerBound(arr, x) {      // first index with te >= x
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m].te < x) lo = m + 1; else hi = m; }
  return lo;
}

function signalColor(P, step, i, a) {
  const ph = P.phase[step][i], sw = P.switched[step][i];
  const tau = t - step * P.dt;
  const mine = (a % 2 === 0) ? 0 : 1;          // approaches N,S -> phase 0 ; E,W -> phase 1
  if (sw && tau < 3) return mine === 1 - ph ? "#f2c14e" : "#d64545";
  if (sw && tau < P.lost) return "#d64545";
  return mine === ph ? "#22a447" : "#d64545";
}

function drawRoads(ctx, S) {
  const {B, E, s} = S;
  ctx.fillStyle = "#e9e7df"; ctx.fillRect(0, 0, SIZE, SIZE);
  // blocks (buildings) between roads
  ctx.fillStyle = "#dcd9cf";
  const xs = [0, 1, 2].map(c => (E + c * B) * s);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    const x0 = c === 0 ? 0 : xs[c - 1] + 2 * LW + 6, x1 = c === 3 ? SIZE : xs[c] - 2 * LW - 6;
    const y0 = r === 0 ? 0 : xs[r - 1] + (r - 1 === 1 ? 3 : 2) * LW + 6;
    const y1 = r === 3 ? SIZE : xs[r] - (r === 1 ? 3 : 2) * LW - 6;
    if (x1 > x0 && y1 > y0) { ctx.beginPath(); ctx.roundRect(x0, y0, x1 - x0, y1 - y0, 6); ctx.fill(); }
  }
  ctx.fillStyle = "#4a4d52";
  for (let r = 0; r < 3; r++) {                 // streets (E-W)
    const y = (E + r * B) * s, hw = (r === 1 ? 3 : 2) * LW;
    ctx.fillRect(0, y - hw, SIZE, 2 * hw);
  }
  for (let c = 0; c < 3; c++) {                 // avenues (N-S)
    const x = (E + c * B) * s, hw = 2 * LW;
    ctx.fillRect(x - hw, 0, 2 * hw, SIZE);
  }
  if (S.P.bus_lane) {                           // red bus lanes on Street 2
    ctx.fillStyle = "rgba(200,60,50,0.55)";
    const y = (E + B) * s;
    ctx.fillRect(0, y + 2 * LW, SIZE, LW); ctx.fillRect(0, y - 3 * LW, SIZE, LW);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.setLineDash([6, 6]); ctx.lineWidth = 1;
  for (let r = 0; r < 3; r++) { const y = (E + r * B) * s;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(SIZE, y); ctx.stroke(); }
  for (let c = 0; c < 3; c++) { const x = (E + c * B) * s;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, SIZE); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.font = "700 14px -apple-system, Segoe UI, sans-serif"; ctx.fillStyle = "#1f1e1b";
  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3), c = i % 3;
    ctx.fillText("J" + (i + 1), (E + c * B) * s + 2 * LW + 5, (E + r * B) * s - (r === 1 ? 3 : 2) * LW - 5);
  }
}

function tint(ctx, G, nQ) {              // colour one approach by queue per lane
  const v = Math.min(1, (nQ / G.lanes) / 8);
  ctx.fillStyle = nQ === 0 ? "hsla(130,60%,42%,0.20)"
                           : `hsla(${Math.round(125 * (1 - v))},80%,44%,${(0.38 + 0.4 * v).toFixed(2)})`;
  const w = G.lanes * LW;
  ctx.beginPath();
  ctx.moveTo(G.start[0], G.start[1]);
  ctx.lineTo(G.stop[0], G.stop[1]);
  ctx.lineTo(G.stop[0] + G.right[0] * w, G.stop[1] + G.right[1] * w);
  ctx.lineTo(G.start[0] + G.right[0] * w, G.start[1] + G.right[1] * w);
  ctx.closePath(); ctx.fill();
}

function draw(S, canvas) {
  const ctx = canvas.getContext("2d");
  const P = S.P;
  const step = Math.min(P.phase.length - 1, Math.max(0, Math.floor(t / P.dt)));
  drawRoads(ctx, S);
  let nQueued = 0;
  // pass 1: who is moving / queued on each approach
  const lists = [];
  for (let L = 0; L < 36; L++) {
    const G = S.geom[L], segs = S.segs[L];
    const lo = lowerBound(segs, t - S.maxDur[L] - 1);
    const queued = [], moving = [];
    for (let k = lo; k < segs.length; k++) {
      const g = segs[k];
      if (g.te > t) break;
      if (t >= g.td) continue;
      if (t >= g.ta) queued.push(g); else moving.push(g);
    }
    queued.sort((a, b) => a.ta - b.ta);
    const nCars = queued.filter(g => g.vt !== 4).length;
    nQueued += nCars;
    lists.push({queued, moving, nCars});
    tint(ctx, G, nCars);
  }
  // pass 2: vehicles on top of the coloured roads
  for (let L = 0; L < 36; L++) {
    const G = S.geom[L], {queued, moving} = lists[L];
    const qs = 7.5 * S.s * 1.0 + 0.6;
    const nl = G.lanes;
    let rank = 0;
    for (const g of queued) {
      let lane, row, along;
      if (g.vt === 4) { lane = -0.5; along = 3; }
      else { lane = rank % nl; row = Math.floor(rank / nl); along = row * qs + qs / 2 + 1; rank++; }
      drawVeh(ctx, G, lane, along, g, nl);
    }
    const tailRows = Math.ceil(rank / nl);
    for (const g of moving) {
      const p = (t - g.te) / Math.max(0.1, g.ta - g.te);
      let along = (1 - p) * G.len;
      const lane = g.vt === 4 ? -0.5 : (g.id % nl);
      const tail = tailRows * qs + qs / 2 + 1;
      if (g.vt !== 4 && along < tail) along = tail + (g.id % 3) * 0.8;
      drawVeh(ctx, G, lane, along, g, nl);
    }
  }
  // signals
  for (let L = 0; L < 36; L++) {
    const G = S.geom[L];
    const col = signalColor(P, step, G.i, G.a);
    ctx.strokeStyle = col; ctx.lineWidth = 3;
    const w = G.lanes * LW;
    const x0 = G.stop[0] + G.right[0] * 0.5, y0 = G.stop[1] + G.right[1] * 0.5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + G.right[0] * w, y0 + G.right[1] * w); ctx.stroke();
  }
  // pedestrians
  for (let i = 0; i < 9; i++) {
    const n = P.ped[step][i];
    if (!n) continue;
    const r = Math.floor(i / 3), c = i % 3;
    const cx = (S.E + c * S.B) * S.s, cy = (S.E + r * S.B) * S.s;
    const off = (r === 1 ? 3 : 2) * LW + 7;
    ctx.fillStyle = "#7b4fd6";
    const shown = Math.min(n, 16);
    for (let k = 0; k < shown; k++) {
      const corner = k % 4, j = Math.floor(k / 4);
      const sx = corner % 2 === 0 ? -1 : 1, sy = corner < 2 ? -1 : 1;
      ctx.beginPath();
      ctx.arc(cx + sx * (2 * LW + 6 + j * 3), cy + sy * (off), 1.8, 0, 6.283); ctx.fill();
    }
    if (n > 16) { ctx.font = "600 10px sans-serif"; ctx.fillText(String(n), cx + 2 * LW + 6, cy + off + 12); }
  }
  // incident marker
  const inc = P.incident;
  if (inc && t >= inc.t_start && t < inc.t_end) {
    const G = S.geom[inc.junction * 4 + inc.approach];
    const mx = G.stop[0] - G.d[0] * 40 + G.right[0] * LW * 2, my = G.stop[1] - G.d[1] * 40 + G.right[1] * LW * 2;
    ctx.fillStyle = "#e5383b"; ctx.beginPath(); ctx.moveTo(mx, my - 9); ctx.lineTo(mx + 8, my + 6);
    ctx.lineTo(mx - 8, my + 6); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "700 10px sans-serif"; ctx.fillText("!", mx - 2, my + 4);
  }
  if (P.event) {
    ctx.fillStyle = "rgba(123,79,214,0.85)"; ctx.font = "700 14px sans-serif";
    ctx.fillText("Arena", (S.E + 2 * S.B) * S.s + 26, (S.E + 1.5 * S.B) * S.s);
  }
  // weather overlay
  if (P.weather === "Sandstorm") { ctx.fillStyle = "rgba(214,170,100,0.28)"; ctx.fillRect(0, 0, SIZE, SIZE); }
  if (P.weather === "Rain") {
    ctx.strokeStyle = "rgba(60,110,200,0.22)"; ctx.lineWidth = 1;
    for (let k = 0; k < 140; k++) { const x = (k * 97 + t * 40) % SIZE, y = (k * 53 + t * 90) % SIZE;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 3, y + 8); ctx.stroke(); }
  }
  const c = counters(S);
  return {q: nQueued, delay: c.delay, co2: c.co2, ped: P.ped[step].reduce((a, b) => a + b, 0)};
}

function drawVeh(ctx, G, lane, along, g, nl) {
  const latOff = (lane + 0.5) * LW;
  const x = G.stop[0] - G.d[0] * along + G.right[0] * latOff;
  const y = G.stop[1] - G.d[1] * along + G.right[1] * latOff;
  const big = (g.vt === 2 || g.vt === 3);
  const lenPx = big ? 10 : 6.2, wPx = big ? 4.0 : 3.4;
  ctx.fillStyle = COLORS[g.vt];
  if (g.vt === 4 && Math.floor(t * 2) % 2 === 0) ctx.fillStyle = "#2f6fdb";
  const horiz = G.d[1] === 0;
  const w = horiz ? lenPx : wPx, h = horiz ? wPx : lenPx;
  ctx.fillRect(x - w / 2, y - h / 2, w, h);
}

const states = PAYLOADS.map(prep);
const wrap = document.getElementById("wrap");
const TILES = [["q", "Cars queued", true], ["delay", "Queue delay (veh-min)", true],
               ["co2", "CO\u2082 (kg)", true], ["ped", "Pedestrians waiting", false]];
const views = states.map((S, k) => {
  const cell = document.createElement("div"); cell.className = "cell";
  const title = document.createElement("div"); title.className = "title"; title.textContent = NAMES[k];
  const kp = document.createElement("div"); kp.className = "kpis";
  const tiles = {};
  for (const [key, label] of TILES) {
    const d = document.createElement("div"); d.className = "kpi";
    d.innerHTML = '<div class="v">0</div><div class="l">' + label + '</div><div class="d"></div>';
    kp.appendChild(d); tiles[key] = d;
  }
  const cv = document.createElement("canvas"); cv.width = SIZE; cv.height = SIZE;
  cell.appendChild(title); cell.appendChild(kp); cell.appendChild(cv); wrap.appendChild(cell);
  return {S, cv, tiles};
});
const FMT = {q: v => String(Math.round(v)), delay: v => v.toFixed(1), co2: v => v.toFixed(1),
             ped: v => String(Math.round(v))};
function render() {
  const M = views.map(v => draw(v.S, v.cv));
  views.forEach((v, k) => {
    for (const [key, , compare] of TILES) {
      const tile = v.tiles[key];
      tile.querySelector(".v").textContent = FMT[key](M[k][key]);
      const d = tile.querySelector(".d");
      if (k > 0 && compare && M[0][key] >= 1) {
        const pc = 100 * (M[k][key] - M[0][key]) / M[0][key];
        d.textContent = (pc <= 0 ? "\u2212" : "+") + Math.abs(pc).toFixed(0) + "% vs left";
        d.className = "d " + (pc < 0 ? "good" : (pc > 0 ? "bad" : ""));
      } else { d.textContent = ""; d.className = "d"; }
    }
  });
  const m = Math.floor(t / 60), sec = Math.floor(t % 60);
  document.getElementById("clock").textContent = String(m).padStart(2, "0") + ":" + String(sec).padStart(2, "0");
  document.getElementById("scrub").value = Math.floor(t);
}
function loop(ts) {
  if (!playing) { last = null; return; }
  if (last !== null) { t += (ts - last) / 1000 * speed; }
  last = ts;
  if (t >= tMax) { t = tMax; playing = false; document.getElementById("play").innerHTML = "&#9654; Play"; }
  render();
  if (playing) requestAnimationFrame(loop);
}
document.getElementById("play").onclick = () => {
  playing = !playing;
  document.getElementById("play").innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  if (playing) { if (t >= tMax) t = 0; requestAnimationFrame(loop); }
};
document.getElementById("reset").onclick = () => { t = 0; render(); };
document.getElementById("speed").onchange = e => { speed = Number(e.target.value); };
document.getElementById("scrub").oninput = e => { t = Number(e.target.value); render(); };
document.addEventListener("keydown", e => { if (e.code === "Space") { e.preventDefault(); document.getElementById("play").click(); } });
t = Math.min(__START__, tMax);
render();
</script></body></html>
"""


def render_html(payloads: list[dict], names: list[str], start: float = 0.0) -> str:
    slim = []
    for p in payloads:
        q = {k: p[k] for k in ("dt", "lost", "t_max", "block", "entry", "lanes", "up",
                               "segments", "phase", "switched", "ped", "incident",
                               "weather")}
        q["bus_lane"] = p.get("bus_lane", False)
        q["event"] = p.get("event", False)
        slim.append(q)
    html = TEMPLATE.replace("__PAYLOADS__", json.dumps(slim, separators=(",", ":")))
    html = html.replace("__NAMES__", json.dumps(names))
    html = html.replace("__START__", str(float(start)))
    return html
