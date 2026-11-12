"""
sim_view.py - animated canvas that replays a recorded episode.

The simulator records every vehicle's passage over every link as
(vehicle type, link, time entered, time reached the stop line, time departed).
The browser interpolates positions from those timestamps, stacks queued vehicles
behind the stop line, colours the signal heads from the phase log, and draws
pedestrians as walking figures (waiting at the corners, crossing on the zebras when served).
Any junction can be put in focus (bar selector or a click on the map): the view zooms in and
the counters and sparkline switch to that junction. Two recordings with the same random seed can
be shown side by side on one clock (e.g. Fixed-time vs Q-learning). Each view carries live
counters (cars queued, queue delay and CO2 accumulated so far, pedestrians waiting), a
queue-history sparkline with a playhead, an emergency-preemption badge, and colours every
approach by its current queue per lane (night-mode city backdrop drawn once per view).
"""
from __future__ import annotations

import json

TEMPLATE = r"""
<!doctype html>
<html><head><meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
  :root { --bg:#070b14; --panel:rgba(18,26,43,0.85); --ink:#e8eef8; --muted:#8693a8;
          --line:rgba(148,163,184,0.16); --cyan:#22d3ee; --chrome:282px; }
  html, body { height:100%; }
  body { margin:0; padding:0 4px; box-sizing:border-box; overflow:hidden;
         font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; background:transparent;
         color:var(--ink); }
  .bar { display:flex; gap:10px; align-items:center; flex-wrap:nowrap; padding:4px 2px 10px 2px; }
  button { border:1px solid var(--line); background:rgba(255,255,255,0.04); color:var(--ink);
           padding:7px 16px; border-radius:10px; font-size:15px; font-weight:600; cursor:pointer;
           white-space:nowrap; transition:all .15s; font-family:inherit; }
  button:hover { border-color:rgba(34,211,238,0.5); }
  button.primary { background:linear-gradient(135deg,#0891b2,#2563eb); border-color:transparent;
                   box-shadow:0 0 18px rgba(34,211,238,0.35); }
  select { padding:6px 8px; border-radius:8px; border:1px solid var(--line); font-size:14px;
           background:#0e1524; color:var(--ink); font-family:inherit; }
  input[type=range] { flex:1; min-width:120px; accent-color:var(--cyan); }
  .clock { font-family:'JetBrains Mono', monospace; font-weight:700; font-size:21px; min-width:74px;
           color:var(--cyan); text-shadow:0 0 12px rgba(34,211,238,0.6); }
  .lbl { font-size:12px; color:var(--muted); letter-spacing:.06em; text-transform:uppercase; }
  .wrap { display:flex; gap:16px; flex-wrap:nowrap; }
  .cell { flex:1 1 0; min-width:0; display:flex; flex-direction:column; align-items:center; }
  .head { display:flex; align-items:center; gap:8px; width:min(100%, calc(100vh - var(--chrome)));
          margin:0 0 6px 0; }
  .title { font-weight:700; font-size:16px; letter-spacing:.01em; }
  .dot { width:10px; height:10px; border-radius:50%; box-shadow:0 0 10px currentColor; }
  .badge { margin-left:auto; font-size:11px; font-weight:700; letter-spacing:.08em; padding:3px 8px;
           border-radius:999px; background:rgba(255,77,94,0.15); color:#ff6b7a;
           border:1px solid rgba(255,77,94,0.45); visibility:hidden; }
  .badge.on { visibility:visible; animation:blink 0.9s infinite; }
  @keyframes blink { 50% { opacity:.35; } }
  .kpis { display:flex; gap:6px; width:min(100%, calc(100vh - var(--chrome))); margin-bottom:6px; }
  .kpi { flex:1 1 0; min-width:0; background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:5px 9px 4px 9px; position:relative; overflow:hidden; }
  .kpi::before { content:""; position:absolute; left:0; top:0; height:2px; width:100%;
                 background:linear-gradient(90deg,var(--cyan),transparent); opacity:.7; }
  .kpi .v { font-family:'JetBrains Mono', monospace; font-size:20px; font-weight:700; line-height:1.2;
            white-space:nowrap; }
  .kpi .l { font-size:10.5px; color:var(--muted); line-height:1.25; white-space:nowrap;
            overflow:hidden; text-overflow:ellipsis; text-transform:uppercase; letter-spacing:.05em; }
  .kpi .d { font-size:11px; font-weight:700; min-height:13px; line-height:1.2; white-space:nowrap; }
  .good { color:#34d399; } .bad { color:#ff6b7a; }
  .sparkbox { width:min(100%, calc(100vh - var(--chrome))); margin-bottom:6px; position:relative; }
  .spark { width:100%; height:38px; display:block; background:var(--panel); border:1px solid var(--line);
           border-radius:8px; }
  .sparklbl { position:absolute; left:8px; top:3px; font-size:10px; color:var(--muted);
              letter-spacing:.06em; text-transform:uppercase; }
  canvas.city { width:min(100%, calc(100vh - var(--chrome))); height:auto; aspect-ratio:1/1;
                background:#060a12; border-radius:14px; display:block; border:1px solid var(--line);
                box-shadow:0 0 0 1px rgba(34,211,238,0.06), 0 20px 50px rgba(0,0,0,0.45);
                cursor:zoom-in; }
  canvas.city.zoomed { cursor:zoom-out; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted);
            margin-top:8px; justify-content:center; }
  .sw { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:4px;
        vertical-align:-1px; }
  .grad { display:inline-block; width:54px; height:8px; border-radius:4px; margin:0 4px;
          vertical-align:0px;
          background:linear-gradient(90deg, #21c98b, #f5b301, #ff4d5e); }
</style></head>
<body>
<div class="bar">
  <button id="play" class="primary">&#9654; Play</button>
  <button id="reset">&#8634;</button>
  <span class="lbl">Speed</span>
  <select id="speed">
    <option value="1">1&times;</option><option value="4">4&times;</option>
    <option value="10" selected>10&times;</option><option value="30">30&times;</option>
    <option value="60">60&times;</option>
  </select>
  <span class="lbl">View</span>
  <select id="focus" title="Zoom to one junction (or click a junction on the map)">
    <option value="-1">Whole network</option>
    <option value="0">J1</option><option value="1">J2</option><option value="2">J3</option>
    <option value="3">J4</option><option value="4">J5</option><option value="5">J6</option>
    <option value="6">J7</option><option value="7">J8</option><option value="8">J9</option>
  </select>
  <input id="scrub" type="range" min="0" max="900" step="1" value="0">
  <span class="clock" id="clock">00:00</span>
</div>
<div class="wrap" id="wrap"></div>
<div class="legend">
  <span>Road glow = queue per lane <span class="grad"></span></span>
  <span><span class="sw" style="background:linear-gradient(90deg,#eef2f7,#c0566a,#5b7fbf,#cdb68c)"></span>Car</span>
  <span><span class="sw" style="background:#22d3ee"></span>Autonomous vehicle</span>
  <span><span class="sw" style="background:#d39a5c"></span>Truck</span>
  <span><span class="sw" style="background:#f5b301"></span>Bus</span>
  <span><span class="sw" style="background:#f4f6fa;box-shadow:inset 0 -3px 0 #ff4d5e"></span>Ambulance</span>
  <span><span class="sw" style="background:#ff3b4f;width:6px"></span>Brake lights = queued</span>
  <span><svg width="9" height="13" viewBox="0 0 9 13" style="vertical-align:-2px;margin-right:4px"><circle cx="4.5" cy="2" r="1.7" fill="#f1c9a5"/><rect x="2.6" y="4" width="3.8" height="4.6" rx="1.2" fill="#a78bfa"/><path d="M3.4 8.4 L2.4 12.4 M5.6 8.4 L6.6 12.4 M2.6 5 L1.2 8 M6.4 5 L7.8 8" stroke="#cbd5e1" stroke-width="1.1" stroke-linecap="round"/></svg>Pedestrians (waiting / crossing)</span>
  <span>Click a junction to zoom</span>
  <span><span class="sw" style="background:#34d399"></span><span class="sw" style="background:#fbbf24"></span><span class="sw" style="background:#ff4d5e"></span>Signal</span>
</div>
<script>
const PAYLOADS = __PAYLOADS__;
const NAMES = __NAMES__;
const PCOL = {"Fixed-time":"#f5b301","Actuated":"#21c98b","Q-learning":"#3d8bff","SARSA":"#ff7a45"};
const HV = [[0,-1],[1,0],[0,1],[-1,0]];
const COLORS = ["#dbe4f3", "#22d3ee", "#d39a5c", "#f5b301", "#ff4d5e"];
const SIZE = 640;                 // canvas pixels (square)
const LW = 6.0;                   // lateral pixels per lane (exaggerated so vehicles read as vehicles)
const DPR = 2;                    // backing-store scale: crisp vehicle detail on any screen
// vehicle sprites (canvas px): car, AV, truck, bus, emergency
const VLEN = [9.0, 9.0, 15.0, 17.0, 11.0], VWID = [4.3, 4.3, 5.0, 5.2, 4.7];
const GAP = 1.6;                  // bumper-to-bumper gap in a queue
const CAR_COLS = ["#eef2f7", "#a8b3c4", "#6b7a90", "#c0566a", "#5b7fbf", "#cdb68c", "#6fae8a",
                  "#e8e1d0", "#8c5a9e", "#3f4a5c"];
// per vehicle type (car, AV, truck, bus, emergency): config.EMISSION  (g CO2 per km / per s idle / per stop)
const EM_KM = [249, 237, 900, 1100, 400], EM_IDLE = [0.74, 0.74, 2.8, 2.8, 1.2],
      EM_STOP = [41, 25, 250, 290, 60];
const PED_S = 1.55;               // pedestrian figure scale (exaggerated, like the vehicles)
const ZOOM = 2.8;                 // magnification when one junction is in focus
const NAMES_J = ["J1 · Street 1 × Avenue A", "J2 · Street 1 × Avenue B", "J3 · Street 1 × Avenue C",
                 "J4 · Street 2 × Avenue A", "J5 · Street 2 × Avenue B", "J6 · Street 2 × Avenue C",
                 "J7 · Street 3 × Avenue A", "J8 · Street 3 × Avenue B", "J9 · Street 3 × Avenue C"];
const SKIN = ["#f1c9a5", "#d9a477", "#b97d52", "#8d5a3a", "#e8b896", "#6b4430"];
const SHIRT = ["#a78bfa", "#f472b6", "#38bdf8", "#facc15", "#34d399", "#fb923c", "#e2e8f0",
               "#c084fc", "#60a5fa", "#f87171"];
const PANTS = ["#334155", "#1e3a5f", "#3f3f46", "#4b5563", "#1f2937"];
let focus = __FOCUS__;
let t = 0, playing = false, speed = 10, last = null;
const tMax = Math.min(...PAYLOADS.map(p => p.t_max));
document.getElementById("scrub").max = Math.floor(tMax);

function rand(seed) { let x = seed; return () => { x = (x * 16807) % 2147483647; return x / 2147483647; }; }

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
                td: new Float32Array(n), km: new Float32Array(n), j: new Uint8Array(n)};
  for (let k = 0; k < n; k++) {
    const g = P.segments[k];
    flat.vt[k] = g[0]; flat.te[k] = g[2]; flat.ta[k] = g[3]; flat.td[k] = g[4];
    flat.km[k] = geom[g[1]].len / s / 1000; flat.j[k] = Math.floor(g[1] / 4);
  }
  // queue history (cars queued every 10 s) for the sparkline
  const NS = Math.floor(P.t_max / 10) + 1, hist = new Float32Array(NS);
  const histJ = Array.from({length: 9}, () => new Float32Array(NS));
  for (let k = 0; k < n; k++) {
    if (flat.vt[k] === 4) continue;
    const a0 = Math.ceil(flat.ta[k] / 10), a1 = Math.floor((flat.td[k] - 1e-6) / 10);
    for (let j = Math.max(0, a0); j <= Math.min(NS - 1, a1); j++) { hist[j] += 1; histJ[flat.j[k]][j] += 1; }
  }
  const S = {P, s, segs, maxDur, geom, B, E, flat, hist, histJ, bgFocus: {}};
  S.bg = backdrop(S, null);
  return S;
}

function camera(S, f) {           // world -> screen for junction f (null = whole network)
  if (f === null || f < 0) return {z: 1, x0: 0, y0: 0};
  const r = Math.floor(f / 3), c = f % 3;
  const cx = (S.E + c * S.B) * S.s, cy = (S.E + r * S.B) * S.s, half = SIZE / (2 * ZOOM);
  const clamp = v => Math.min(SIZE - half, Math.max(half, v));
  return {z: ZOOM, x0: clamp(cx) - half, y0: clamp(cy) - half};
}

function backdrop(S, f) {        // static city layer, drawn once per camera
  const {B, E, s} = S;
  const cam = camera(S, f);
  const cv = document.createElement("canvas"); cv.width = SIZE * DPR; cv.height = SIZE * DPR;
  const ctx = cv.getContext("2d");
  ctx.setTransform(DPR * cam.z, 0, 0, DPR * cam.z, -cam.x0 * DPR * cam.z, -cam.y0 * DPR * cam.z);
  const g0 = ctx.createRadialGradient(SIZE / 2, SIZE / 2, 40, SIZE / 2, SIZE / 2, SIZE * 0.75);
  g0.addColorStop(0, "#0b1324"); g0.addColorStop(1, "#04070d");
  ctx.fillStyle = g0; ctx.fillRect(0, 0, SIZE, SIZE);
  const R = rand(7);
  const xs = [0, 1, 2].map(c => (E + c * B) * s);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    const x0 = c === 0 ? 0 : xs[c - 1] + 2 * LW + 6, x1 = c === 3 ? SIZE : xs[c] - 2 * LW - 6;
    const y0 = r === 0 ? 0 : xs[r - 1] + (r - 1 === 1 ? 3 : 2) * LW + 6;
    const y1 = r === 3 ? SIZE : xs[r] - (r === 1 ? 3 : 2) * LW - 6;
    if (x1 <= x0 || y1 <= y0) continue;
    ctx.fillStyle = "#0d1626"; ctx.beginPath(); ctx.roundRect(x0, y0, x1 - x0, y1 - y0, 8); ctx.fill();
    ctx.strokeStyle = "rgba(61,139,255,0.10)"; ctx.lineWidth = 1; ctx.stroke();
    // towers with lit windows
    const nb = 2 + Math.floor(R() * 3);
    for (let k = 0; k < nb; k++) {
      const bw = 18 + R() * 34, bh = 18 + R() * 34;
      const bx = x0 + 6 + R() * Math.max(1, x1 - x0 - bw - 12), by = y0 + 6 + R() * Math.max(1, y1 - y0 - bh - 12);
      if (bx + bw > x1 - 4 || by + bh > y1 - 4) continue;
      ctx.fillStyle = "#121e33"; ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 3); ctx.fill();
      for (let wx = bx + 3; wx < bx + bw - 3; wx += 5) for (let wy = by + 3; wy < by + bh - 3; wy += 5) {
        const q = R();
        if (q < 0.22) { ctx.fillStyle = q < 0.05 ? "rgba(34,211,238,0.55)" : "rgba(251,191,36,0.45)";
                        ctx.fillRect(wx, wy, 1.6, 1.6); }
      }
    }
  }
  ctx.fillStyle = "#151c2a";
  for (let r = 0; r < 3; r++) {                 // streets (E-W)
    const y = (E + r * B) * s, hw = (r === 1 ? 3 : 2) * LW;
    ctx.fillRect(0, y - hw, SIZE, 2 * hw);
  }
  for (let c = 0; c < 3; c++) {                 // avenues (N-S)
    const x = (E + c * B) * s, hw = 2 * LW;
    ctx.fillRect(x - hw, 0, 2 * hw, SIZE);
  }
  if (S.P.bus_lane) {                           // bus lanes on Street 2
    ctx.fillStyle = "rgba(245,179,1,0.20)";
    const y = (E + B) * s;
    ctx.fillRect(0, y + 2 * LW, SIZE, LW); ctx.fillRect(0, y - 3 * LW, SIZE, LW);
  }
  ctx.strokeStyle = "rgba(226,232,240,0.22)"; ctx.setLineDash([6, 7]); ctx.lineWidth = 1;
  for (let r = 0; r < 3; r++) { const y = (E + r * B) * s;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(SIZE, y); ctx.stroke(); }
  for (let c = 0; c < 3; c++) { const x = (E + c * B) * s;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, SIZE); ctx.stroke(); }
  ctx.setLineDash([]);
  // zebra crossings
  ctx.fillStyle = "rgba(226,232,240,0.16)";
  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3), c = i % 3, cx = (E + c * B) * s, cy = (E + r * B) * s;
    const hv = (r === 1 ? 3 : 2) * LW, hh = 2 * LW;
    for (let k = -hh + 1; k < hh; k += 3) { ctx.fillRect(cx + k, cy - hv - 6, 1.6, 4); ctx.fillRect(cx + k, cy + hv + 2, 1.6, 4); }
    for (let k = -hv + 1; k < hv; k += 3) { ctx.fillRect(cx - hh - 6, cy + k, 4, 1.6); ctx.fillRect(cx + hh + 2, cy + k, 4, 1.6); }
  }
  ctx.font = `700 ${(12 / Math.sqrt(cam.z)).toFixed(1)}px 'JetBrains Mono', monospace`;
  ctx.fillStyle = "rgba(148,163,184,0.85)";
  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3), c = i % 3;
    ctx.fillText("J" + (i + 1), (E + c * B) * s + 2 * LW + 6, (E + r * B) * s - (r === 1 ? 3 : 2) * LW - 7);
  }
  if (S.P.event) {
    const ax = (E + 2 * B) * s + 64, ay = (E + 1.5 * B) * s;
    const gg = ctx.createRadialGradient(ax, ay, 4, ax, ay, 46);
    gg.addColorStop(0, "rgba(167,139,250,0.55)"); gg.addColorStop(1, "rgba(167,139,250,0)");
    ctx.fillStyle = gg; ctx.beginPath(); ctx.arc(ax, ay, 46, 0, 6.283); ctx.fill();
    const fz = 12 / Math.sqrt(cam.z);
    ctx.fillStyle = "#c4b5fd"; ctx.font = `700 ${fz.toFixed(1)}px Inter, sans-serif`;
    ctx.fillText("ARENA", ax - 1.7 * fz, ay + fz / 3);
  }
  return cv;
}

function counters(S) {
  const F = S.flat;
  let delay = 0, co2 = 0, ev = 0;
  for (let k = 0; k < F.n; k++) {
    if (focus >= 0 && F.j[k] !== focus) continue;
    const vt = F.vt[k];
    const te = F.te[k];
    if (t <= te) continue;
    const ta = F.ta[k], td = F.td[k];
    if (vt === 4) { if (t < td) ev++; continue; }
    co2 += EM_KM[vt] * F.km[k] * Math.min(1, (t - te) / Math.max(0.1, ta - te));
    if (t > ta) {
      const w = Math.min(t, td) - ta;
      if (w > 0) { delay += w; co2 += EM_IDLE[vt] * w; }
      if (td - ta > 1) co2 += EM_STOP[vt];
    }
  }
  return {delay: delay / 60, co2: co2 / 1000, ev};
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
  if (sw && tau < 3) return mine === 1 - ph ? "#fbbf24" : "#ff4d5e";
  if (sw && tau < P.lost) return "#ff4d5e";
  return mine === ph ? "#34d399" : "#ff4d5e";
}

function tint(ctx, G, nQ) {              // glow one approach by queue per lane
  const v = Math.min(1, (nQ / G.lanes) / 8);
  if (nQ === 0) { ctx.fillStyle = "rgba(33,201,139,0.10)"; }
  else { const hue = Math.round(150 * (1 - v)) - 5;
         ctx.fillStyle = `hsla(${hue},95%,55%,${(0.22 + 0.45 * v).toFixed(2)})`; }
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
  const cam = camera(S, focus);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (focus >= 0 && !S.bgFocus[focus]) S.bgFocus[focus] = backdrop(S, focus);
  ctx.drawImage(focus >= 0 ? S.bgFocus[focus] : S.bg, 0, 0);
  ctx.setTransform(DPR * cam.z, 0, 0, DPR * cam.z, -cam.x0 * DPR * cam.z, -cam.y0 * DPR * cam.z);
  let nQueued = 0, nQJ = 0;
  const qByL = new Array(36).fill(0);
  const lists = [];
  ctx.globalCompositeOperation = "lighter";
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
    nQueued += nCars; qByL[L] = nCars;
    if (G.i === focus) nQJ += nCars;
    lists.push({queued, moving, nCars});
    tint(ctx, G, nCars);
  }
  ctx.globalCompositeOperation = "source-over";
  for (let L = 0; L < 36; L++) {
    const G = S.geom[L], {queued, moving} = lists[L];
    const nl = G.lanes;
    // queued vehicles: fill lanes in turn, bumper to bumper from the stop line
    const lanes = Array.from({length: nl}, () => []);
    let rank = 0;
    for (const g of queued) {
      if (g.vt === 4) { drawVeh(ctx, G, -0.5, 2 + VLEN[4] / 2, g, false); continue; }
      lanes[rank % nl].push(g); rank++;
    }
    const tails = new Array(nl).fill(1);
    for (let ln = 0; ln < nl; ln++) {
      let need = 0;
      for (const g of lanes[ln]) need += VLEN[g.vt] + GAP;
      const f = need > G.len - 4 ? Math.max(0.55, (G.len - 4) / need) : 1;   // jam: squeeze
      let off = 1;
      for (const g of lanes[ln]) {
        const half = VLEN[g.vt] / 2;
        const along = off + half * f;
        off += (VLEN[g.vt] + GAP) * f;
        if (along + half > G.len) continue;          // queue spills past the upstream junction
        drawVeh(ctx, G, ln, along, g, false);
      }
      tails[ln] = off;
    }
    // moving vehicles: follow the vehicle ahead in the same lane, never overlap the queue
    const mv = moving.map(g => {
      const p = (t - g.te) / Math.max(0.1, g.ta - g.te);
      return {g, along: (1 - p) * G.len, lane: g.vt === 4 ? -0.5 : (g.id % nl)};
    }).sort((u, v) => u.along - v.along);
    const front = tails.slice();
    for (const m of mv) {
      const half = VLEN[m.g.vt] / 2;
      if (m.g.vt !== 4) {
        m.along = Math.max(m.along, front[m.lane] + half);
        front[m.lane] = m.along + half + GAP;
        if (m.along - half > G.len) continue;
      }
      drawVeh(ctx, G, m.lane, m.along, m.g, true);
    }
  }
  // signals (glowing heads at each stop line)
  for (let L = 0; L < 36; L++) {
    const G = S.geom[L];
    const col = signalColor(P, step, G.i, G.a);
    const w = G.lanes * LW;
    const x0 = G.stop[0] + G.right[0] * 0.5, y0 = G.stop[1] + G.right[1] * 0.5;
    ctx.shadowColor = col; ctx.shadowBlur = 8;
    ctx.strokeStyle = col; ctx.lineWidth = 2.6;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + G.right[0] * w, y0 + G.right[1] * w); ctx.stroke();
  }
  ctx.shadowBlur = 0;
  // pedestrians: figures waiting at the four corners; those served this step walk across
  for (let i = 0; i < 9; i++) {
    const n = P.ped[step][i];
    const r = Math.floor(i / 3), c = i % 3;
    const cx = (S.E + c * S.B) * S.s, cy = (S.E + r * S.B) * S.s;
    const hv = (r === 1 ? 3 : 2) * LW, hh = 2 * LW;
    if (n) {
      const shown = Math.min(n, 24);
      for (let k = 0; k < shown; k++) {
        const corner = k % 4, j = Math.floor(k / 4), row = Math.floor(j / 3), col = j % 3;
        const sx = corner % 2 === 0 ? -1 : 1, sy = corner < 2 ? -1 : 1;
        drawPed(ctx, cx + sx * (hh + 8 + col * 5.2 + row * 2.6), cy + sy * (hv + 9.5 + row * 9.5),
                i * 31 + k, false);
      }
      if (n > 24) {
        ctx.fillStyle = "#c4b5fd"; ctx.font = "700 8px 'JetBrains Mono', monospace";
        ctx.fillText("+" + (n - 24), cx + hh + 8, cy + hv + 33);
      }
    }
    // crossing: pedestrians served in this step walk over the zebras that are green for them
    const prev = step > 0 ? P.ped[step - 1][i] : 0;
    const served = Math.min(8, Math.max(0, prev - n));
    if (served > 0) {
      const ph = P.phase[step][i], p = Math.min(1, (t - step * P.dt) / P.dt);
      for (let k = 0; k < served; k++) {
        const dir = k % 2 === 0 ? 1 : -1, side = k % 4 < 2 ? -1 : 1, lag = 0.12 * Math.floor(k / 4);
        const q = Math.min(1, Math.max(0, p * 1.15 - lag));
        if (ph === 1) {      // E-W green: walk across the avenue on the north / south zebras
          const x = cx + dir * (-hh - 5 + q * (2 * hh + 10));
          drawPed(ctx, x, cy + side * (hv + 4) + 2, i * 17 + k + 500, true);
        } else {             // N-S green: walk across the street on the west / east zebras
          const y = cy + dir * (-hv - 5 + q * (2 * hv + 10));
          drawPed(ctx, cx + side * (hh + 4), y, i * 17 + k + 700, true);
        }
      }
    }
  }
  // incident marker (pulsing)
  const inc = P.incident;
  if (inc && t >= inc.t_start && t < inc.t_end) {
    const G = S.geom[inc.junction * 4 + inc.approach];
    const mx = G.stop[0] - G.d[0] * 40 + G.right[0] * LW * 2, my = G.stop[1] - G.d[1] * 40 + G.right[1] * LW * 2;
    const pr = 10 + 6 * Math.abs(Math.sin(t * 1.5));
    ctx.strokeStyle = "rgba(255,77,94,0.6)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(mx, my, pr, 0, 6.283); ctx.stroke();
    ctx.fillStyle = "#ff4d5e"; ctx.beginPath(); ctx.moveTo(mx, my - 8); ctx.lineTo(mx + 7, my + 5);
    ctx.lineTo(mx - 7, my + 5); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "700 9px sans-serif"; ctx.fillText("!", mx - 1.5, my + 3.5);
  }
  // weather overlay
  if (P.weather === "Sandstorm") { ctx.fillStyle = "rgba(214,160,80,0.20)"; ctx.fillRect(0, 0, SIZE, SIZE); }
  if (P.weather === "Rain") {
    ctx.strokeStyle = "rgba(125,180,255,0.28)"; ctx.lineWidth = 1;
    for (let k = 0; k < 160; k++) { const x = (k * 97 + t * 40) % SIZE, y = (k * 53 + t * 90) % SIZE;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 3, y + 9); ctx.stroke(); }
  }
  if (focus >= 0) drawFocusInfo(ctx, S, step, qByL);
  const c = counters(S);
  return {q: focus >= 0 ? nQJ : nQueued, delay: c.delay, co2: c.co2,
          ped: focus >= 0 ? P.ped[step][focus] : P.ped[step].reduce((a, b) => a + b, 0), ev: c.ev};
}

function drawPed(ctx, x, y, seed, walking) {
  // a small upright figure (head, torso, arms, legs), sized like the vehicles (exaggerated)
  const skin = SKIN[seed % SKIN.length], shirt = SHIRT[(seed * 7 + 3) % SHIRT.length],
        pants = PANTS[(seed * 3 + 1) % PANTS.length];
  const ph = t * (walking ? 9 : 1.6) + seed * 1.7;
  const sw = walking ? Math.sin(ph) * 0.95 : 0;                 // stride
  const bob = walking ? Math.abs(Math.cos(ph)) * 0.25 : Math.sin(ph) * 0.12;
  ctx.save();
  ctx.translate(x, y - bob * PED_S);
  ctx.scale(PED_S, PED_S);
  ctx.fillStyle = "rgba(0,0,0,0.45)";                            // ground shadow
  ctx.beginPath(); ctx.ellipse(0, 2.9 + bob, 1.5, 0.45, 0, 0, 6.283); ctx.fill();
  ctx.lineCap = "round";
  ctx.strokeStyle = pants; ctx.lineWidth = 0.85;                 // legs
  ctx.beginPath(); ctx.moveTo(-0.35, 0.4); ctx.lineTo(-0.45 + sw * 0.7, 2.8);
  ctx.moveTo(0.35, 0.4); ctx.lineTo(0.45 - sw * 0.7, 2.8); ctx.stroke();
  ctx.strokeStyle = skin; ctx.lineWidth = 0.6;                   // arms
  ctx.beginPath(); ctx.moveTo(-0.85, -1.6); ctx.lineTo(-1.15 - sw * 0.5, 0.2);
  ctx.moveTo(0.85, -1.6); ctx.lineTo(1.15 + sw * 0.5, 0.2); ctx.stroke();
  ctx.fillStyle = shirt;                                          // torso
  ctx.beginPath(); ctx.roundRect(-0.95, -2.0, 1.9, 2.55, 0.7); ctx.fill();
  ctx.fillStyle = skin;                                           // head
  ctx.beginPath(); ctx.arc(0, -3.0, 0.82, 0, 6.283); ctx.fill();
  ctx.fillStyle = "rgba(20,20,28,0.75)";                          // hair
  ctx.beginPath(); ctx.arc(0, -3.25, 0.7, Math.PI * 1.05, Math.PI * 1.95); ctx.fill();
  ctx.restore();
}

function drawFocusInfo(ctx, S, step, qByL) {
  // labels in world units around the focused junction: queue per approach and phase
  const i = focus, r = Math.floor(i / 3), c = i % 3;
  const cx = (S.E + c * S.B) * S.s, cy = (S.E + r * S.B) * S.s;
  const HN = ["N", "E", "S", "W"];
  ctx.font = "700 5.2px 'JetBrains Mono', monospace";
  for (let a = 0; a < 4; a++) {
    const G = S.geom[i * 4 + a], q = qByL[i * 4 + a];
    const lx = G.stop[0] - G.d[0] * 34 + G.right[0] * (G.lanes * LW + 6);
    const ly = G.stop[1] - G.d[1] * 34 + G.right[1] * (G.lanes * LW + 6);
    const txt = HN[a] + " " + q;
    const w = ctx.measureText(txt).width + 4;
    ctx.fillStyle = "rgba(7,11,20,0.82)"; ctx.strokeStyle = "rgba(34,211,238,0.55)"; ctx.lineWidth = 0.4;
    ctx.beginPath(); ctx.roundRect(lx - w / 2, ly - 4, w, 6.4, 1.6); ctx.fill(); ctx.stroke();
    ctx.fillStyle = q > 12 ? "#ff6b7a" : (q > 5 ? "#fbbf24" : "#34d399");
    ctx.fillText(txt, lx - w / 2 + 2, ly + 0.9);
  }
  // screen-space caption
  ctx.save(); ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  const ph = S.P.phase[step][i];
  const cap = NAMES_J[i] + "  ·  " + (ph === 0 ? "N-S green" : "E-W green") + "  ·  queue per approach";
  ctx.font = "700 12px Inter, sans-serif";
  const w = ctx.measureText(cap).width + 16;
  ctx.fillStyle = "rgba(7,11,20,0.85)"; ctx.beginPath(); ctx.roundRect(8, 8, w, 22, 8); ctx.fill();
  ctx.strokeStyle = "rgba(34,211,238,0.45)"; ctx.lineWidth = 1; ctx.stroke();
  ctx.fillStyle = "#e8eef8"; ctx.fillText(cap, 16, 23);
  ctx.restore();
}

function drawVeh(ctx, G, lane, along, g, moving) {
  const latOff = (lane + 0.5) * LW;
  const x = G.stop[0] - G.d[0] * along + G.right[0] * latOff;
  const y = G.stop[1] - G.d[1] * along + G.right[1] * latOff;
  const vt = g.vt, L = VLEN[vt], W = VWID[vt], hl = L / 2, hw = W / 2;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(Math.atan2(G.d[1], G.d[0]));           // +x = direction of travel
  // soft shadow
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.beginPath(); ctx.roundRect(-hl + 0.6, -hw + 0.8, L, W, 1.6); ctx.fill();
  if (moving && vt !== 4) {                         // headlight beam on the road ahead
    ctx.fillStyle = "rgba(255,236,190,0.10)";
    ctx.beginPath(); ctx.moveTo(hl, -hw + 0.6); ctx.lineTo(hl + 9, -hw - 1.2);
    ctx.lineTo(hl + 9, hw + 1.2); ctx.lineTo(hl, hw - 0.6); ctx.closePath(); ctx.fill();
  }
  const glass = "rgba(9,15,28,0.88)";
  if (vt === 0 || vt === 1) {                       // car / autonomous car
    ctx.fillStyle = vt === 1 ? "#22d3ee" : CAR_COLS[g.id % CAR_COLS.length];
    ctx.beginPath(); ctx.roundRect(-hl, -hw, L, W, 1.8); ctx.fill();
    ctx.fillStyle = glass;
    ctx.beginPath(); ctx.roundRect(hl - 3.6, -hw + 0.6, 1.7, W - 1.2, 0.6); ctx.fill();   // windscreen
    ctx.beginPath(); ctx.roundRect(-hl + 1.0, -hw + 0.7, 1.2, W - 1.4, 0.5); ctx.fill();  // rear window
    ctx.fillStyle = "rgba(255,255,255,0.16)";                                             // roof
    ctx.fillRect(-hl + 2.4, -hw + 0.8, L - 6.2, W - 1.6);
    if (vt === 1) {                                 // lidar dome
      ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(-0.4, 0, 0.95, 0, 6.283); ctx.fill();
    }
  } else if (vt === 2) {                            // truck: cab + trailer
    ctx.fillStyle = "#c9cfd9";
    ctx.beginPath(); ctx.roundRect(-hl, -hw, L - 4.6, W, 1.0); ctx.fill();
    ctx.strokeStyle = "rgba(60,70,90,0.55)"; ctx.lineWidth = 0.5;
    for (let k = -hl + 2; k < hl - 5; k += 2) { ctx.beginPath(); ctx.moveTo(k, -hw + 0.4); ctx.lineTo(k, hw - 0.4); ctx.stroke(); }
    ctx.fillStyle = "#e0a25e";
    ctx.beginPath(); ctx.roundRect(hl - 4.0, -hw + 0.2, 4.0, W - 0.4, 1.2); ctx.fill();
    ctx.fillStyle = glass; ctx.fillRect(hl - 1.6, -hw + 0.8, 1.0, W - 1.6);
  } else if (vt === 3) {                            // bus
    ctx.fillStyle = "#f5b301";
    ctx.beginPath(); ctx.roundRect(-hl, -hw, L, W, 1.6); ctx.fill();
    ctx.fillStyle = glass;
    for (let k = -hl + 1.4; k < hl - 3.2; k += 2.3) {           // side windows
      ctx.fillRect(k, -hw + 0.35, 1.6, 0.9); ctx.fillRect(k, hw - 1.25, 1.6, 0.9);
    }
    ctx.fillRect(hl - 1.7, -hw + 0.7, 1.2, W - 1.4);            // windscreen
    ctx.fillStyle = "rgba(255,255,255,0.22)"; ctx.fillRect(-hl + 1.2, -0.5, L - 3.6, 1.0);   // roof stripe
  } else {                                          // ambulance
    ctx.fillStyle = "#f4f6fa";
    ctx.beginPath(); ctx.roundRect(-hl, -hw, L, W, 1.4); ctx.fill();
    ctx.fillStyle = "#ff4d5e"; ctx.fillRect(-hl, -0.55, L - 3.2, 1.1);
    ctx.fillStyle = glass; ctx.fillRect(hl - 2.2, -hw + 0.6, 1.3, W - 1.2);
    const on = Math.floor(t * 4) % 2 === 0;
    ctx.shadowBlur = 12;
    ctx.shadowColor = on ? "#ff4d5e" : "#3d8bff"; ctx.fillStyle = on ? "#ff4d5e" : "#3d8bff";
    ctx.fillRect(-0.6, -hw + 0.3, 1.6, hw - 0.3);
    ctx.shadowColor = on ? "#3d8bff" : "#ff4d5e"; ctx.fillStyle = on ? "#7fb0ff" : "#ff9aa5";
    ctx.fillRect(-0.6, 0, 1.6, hw - 0.3);
    ctx.shadowBlur = 0;
  }
  if (moving) {                                     // headlights
    ctx.fillStyle = "#fff6d6";
    ctx.fillRect(hl - 0.7, -hw + 0.4, 0.8, 1.0); ctx.fillRect(hl - 0.7, hw - 1.4, 0.8, 1.0);
  } else {                                          // brake lights while queued
    ctx.fillStyle = "#ff3b4f";
    ctx.fillRect(-hl - 0.1, -hw + 0.4, 0.8, 1.1); ctx.fillRect(-hl - 0.1, hw - 1.5, 0.8, 1.1);
  }
  ctx.restore();
}

function drawSpark(v) {
  const cv = v.spark, ctx = cv.getContext("2d");
  const W = cv.width = cv.clientWidth * 2, H = cv.height = cv.clientHeight * 2;
  const h = focus >= 0 ? v.S.histJ[focus] : v.S.hist;
  const mx = Math.max(1, ...(focus >= 0 ? states.map(S => Math.max(...S.histJ[focus])) : v.yMax));
  ctx.clearRect(0, 0, W, H);
  const X = j => (j / (h.length - 1)) * W, Y = q => H - 4 - (q / mx) * (H - 18);
  const col = PCOL[v.name] || "#22d3ee";
  const grd = ctx.createLinearGradient(0, 0, 0, H); grd.addColorStop(0, col + "55"); grd.addColorStop(1, col + "00");
  ctx.beginPath(); ctx.moveTo(0, H);
  for (let j = 0; j < h.length; j++) ctx.lineTo(X(j), Y(h[j]));
  ctx.lineTo(W, H); ctx.closePath(); ctx.fillStyle = grd; ctx.fill();
  ctx.beginPath();
  for (let j = 0; j < h.length; j++) j ? ctx.lineTo(X(j), Y(h[j])) : ctx.moveTo(X(j), Y(h[j]));
  ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.stroke();
  const px = (t / tMax) * W;
  ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
}

const states = PAYLOADS.map(prep);
const wrap = document.getElementById("wrap");
const TILES = [["q", "Queued now", true], ["delay", "Queue delay · veh-min", true],
               ["co2", "CO₂ · kg", true], ["ped", "Peds waiting", false]];
const yMaxAll = states.map(S => Math.max(...S.hist));
const views = states.map((S, k) => {
  const cell = document.createElement("div"); cell.className = "cell";
  const head = document.createElement("div"); head.className = "head";
  const col = PCOL[NAMES[k]] || "#22d3ee";
  head.innerHTML = `<span class="dot" style="background:${col};color:${col}"></span>` +
                   `<span class="title">${NAMES[k]}</span><span class="badge">EV PREEMPTION</span>`;
  const kp = document.createElement("div"); kp.className = "kpis";
  const tiles = {};
  for (const [key, label] of TILES) {
    const d = document.createElement("div"); d.className = "kpi";
    d.innerHTML = '<div class="v">0</div><div class="l">' + label + '</div><div class="d"></div>';
    kp.appendChild(d); tiles[key] = d;
  }
  const sb = document.createElement("div"); sb.className = "sparkbox";
  const sp = document.createElement("canvas"); sp.className = "spark";
  sb.appendChild(sp);
  const sl = document.createElement("div"); sl.className = "sparklbl"; sl.textContent = "Cars queued over time";
  sb.appendChild(sl);
  const cv = document.createElement("canvas"); cv.className = "city"; cv.width = SIZE * DPR; cv.height = SIZE * DPR;
  cell.appendChild(head); cell.appendChild(kp); cell.appendChild(sb); cell.appendChild(cv); wrap.appendChild(cell);
  cv.addEventListener("click", ev => {          // click a junction to zoom, click again to go back
    if (focus >= 0) { setFocus(-1); return; }
    const rc = cv.getBoundingClientRect();
    const wx = (ev.clientX - rc.left) / rc.width * SIZE, wy = (ev.clientY - rc.top) / rc.height * SIZE;
    let best = -1, bd = 1e9;
    for (let i = 0; i < 9; i++) {
      const r = Math.floor(i / 3), c = i % 3;
      const d = Math.hypot(wx - (S.E + c * S.B) * S.s, wy - (S.E + r * S.B) * S.s);
      if (d < bd) { bd = d; best = i; }
    }
    if (bd < 70) setFocus(best);
  });
  return {S, cv, tiles, spark: sp, sl, name: NAMES[k], badge: head.querySelector(".badge"), yMax: yMaxAll};
});
function setFocus(f) {
  focus = f;
  document.getElementById("focus").value = String(f);
  views.forEach(v => {
    v.cv.className = "city" + (f >= 0 ? " zoomed" : "");
    v.sl.textContent = f >= 0 ? "Cars queued at J" + (f + 1) + " over time" : "Cars queued over time";
    v.tiles.q.querySelector(".l").textContent = f >= 0 ? "Queued now · J" + (f + 1) : "Queued now";
    v.tiles.ped.querySelector(".l").textContent = f >= 0 ? "Peds waiting · J" + (f + 1) : "Peds waiting";
    v.tiles.delay.querySelector(".l").textContent = f >= 0 ? "J" + (f + 1) + " delay · veh-min" : "Queue delay · veh-min";
    v.tiles.co2.querySelector(".l").textContent = f >= 0 ? "J" + (f + 1) + " CO₂ · kg" : "CO₂ · kg";
  });
  render();
}
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
        d.textContent = (pc <= 0 ? "−" : "+") + Math.abs(pc).toFixed(0) + "% vs left";
        d.className = "d " + (pc < 0 ? "good" : (pc > 0 ? "bad" : ""));
      } else { d.textContent = ""; d.className = "d"; }
    }
    v.badge.className = "badge" + (M[k].ev > 0 ? " on" : "");
    drawSpark(v);
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
document.getElementById("focus").onchange = e => { setFocus(Number(e.target.value)); };
document.getElementById("scrub").oninput = e => { t = Number(e.target.value); render(); };
document.addEventListener("keydown", e => { if (e.code === "Space") { e.preventDefault(); document.getElementById("play").click(); } });
window.addEventListener("resize", render);
t = Math.min(__START__, tMax);
setFocus(focus);
if (__AUTOPLAY__) document.getElementById("play").click();
</script></body></html>
"""


def render_html(payloads: list[dict], names: list[str], start: float = 0.0,
                autoplay: bool = False, focus: int = -1) -> str:
    """focus: junction index 0-8 to open zoomed on that junction, -1 = whole network."""
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
    html = html.replace("__AUTOPLAY__", "true" if autoplay else "false")
    html = html.replace("__FOCUS__", str(int(focus) if focus is not None else -1))
    return html
