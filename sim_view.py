"""
sim_view.py - animated canvas that replays a recorded episode.

The simulator records every vehicle's passage over every link as
(vehicle type, link, time entered, time reached the stop line, time departed).
The browser interpolates positions from those timestamps, stacks queued vehicles
behind the stop line, colours the signal heads from the phase log, and draws
pedestrians waiting at each junction. Two recordings with the same random seed can
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
          --line:rgba(148,163,184,0.16); --cyan:#22d3ee; --chrome:262px; }
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
                box-shadow:0 0 0 1px rgba(34,211,238,0.06), 0 20px 50px rgba(0,0,0,0.45); }
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
  <input id="scrub" type="range" min="0" max="900" step="1" value="0">
  <span class="clock" id="clock">00:00</span>
</div>
<div class="wrap" id="wrap"></div>
<div class="legend">
  <span>Road glow = queue per lane <span class="grad"></span></span>
  <span><span class="sw" style="background:#dbe4f3"></span>Car</span>
  <span><span class="sw" style="background:#22d3ee"></span>Autonomous vehicle</span>
  <span><span class="sw" style="background:#d39a5c"></span>Truck</span>
  <span><span class="sw" style="background:#f5b301"></span>Bus</span>
  <span><span class="sw" style="background:#ff4d5e"></span>Emergency</span>
  <span><span class="sw" style="background:#a78bfa"></span>Pedestrians</span>
  <span><span class="sw" style="background:#34d399"></span><span class="sw" style="background:#fbbf24"></span><span class="sw" style="background:#ff4d5e"></span>Signal</span>
</div>
<script>
const PAYLOADS = __PAYLOADS__;
const NAMES = __NAMES__;
const PCOL = {"Fixed-time":"#f5b301","Actuated":"#21c98b","Q-learning":"#3d8bff","SARSA":"#ff7a45"};
const HV = [[0,-1],[1,0],[0,1],[-1,0]];
const COLORS = ["#dbe4f3", "#22d3ee", "#d39a5c", "#f5b301", "#ff4d5e"];
const SIZE = 640;                 // canvas pixels (square)
const LW = 4.2;                   // lateral pixels per lane (exaggerated)
// per vehicle type (car, AV, truck, bus, emergency): config.EMISSION  (g CO2 per km / per s idle / per stop)
const EM_KM = [249, 237, 900, 1100, 400], EM_IDLE = [0.74, 0.74, 2.8, 2.8, 1.2],
      EM_STOP = [41, 25, 250, 290, 60];
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
                td: new Float32Array(n), km: new Float32Array(n)};
  for (let k = 0; k < n; k++) {
    const g = P.segments[k];
    flat.vt[k] = g[0]; flat.te[k] = g[2]; flat.ta[k] = g[3]; flat.td[k] = g[4];
    flat.km[k] = geom[g[1]].len / s / 1000;
  }
  // queue history (cars queued every 10 s) for the sparkline
  const NS = Math.floor(P.t_max / 10) + 1, hist = new Float32Array(NS);
  for (let k = 0; k < n; k++) {
    if (flat.vt[k] === 4) continue;
    const a0 = Math.ceil(flat.ta[k] / 10), a1 = Math.floor((flat.td[k] - 1e-6) / 10);
    for (let j = Math.max(0, a0); j <= Math.min(NS - 1, a1); j++) hist[j] += 1;
  }
  const S = {P, s, segs, maxDur, geom, B, E, flat, hist};
  S.bg = backdrop(S);
  return S;
}

function backdrop(S) {           // static city layer, drawn once
  const {B, E, s} = S;
  const cv = document.createElement("canvas"); cv.width = SIZE; cv.height = SIZE;
  const ctx = cv.getContext("2d");
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
  ctx.font = "700 12px 'JetBrains Mono', monospace"; ctx.fillStyle = "rgba(148,163,184,0.85)";
  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3), c = i % 3;
    ctx.fillText("J" + (i + 1), (E + c * B) * s + 2 * LW + 6, (E + r * B) * s - (r === 1 ? 3 : 2) * LW - 7);
  }
  if (S.P.event) {
    const ax = (E + 2 * B) * s + 64, ay = (E + 1.5 * B) * s;
    const gg = ctx.createRadialGradient(ax, ay, 4, ax, ay, 46);
    gg.addColorStop(0, "rgba(167,139,250,0.55)"); gg.addColorStop(1, "rgba(167,139,250,0)");
    ctx.fillStyle = gg; ctx.beginPath(); ctx.arc(ax, ay, 46, 0, 6.283); ctx.fill();
    ctx.fillStyle = "#c4b5fd"; ctx.font = "700 12px Inter, sans-serif"; ctx.fillText("ARENA", ax - 20, ay + 4);
  }
  return cv;
}

function counters(S) {
  const F = S.flat;
  let delay = 0, co2 = 0, ev = 0;
  for (let k = 0; k < F.n; k++) {
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
  ctx.drawImage(S.bg, 0, 0);
  let nQueued = 0;
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
    nQueued += nCars;
    lists.push({queued, moving, nCars});
    tint(ctx, G, nCars);
  }
  ctx.globalCompositeOperation = "source-over";
  for (let L = 0; L < 36; L++) {
    const G = S.geom[L], {queued, moving} = lists[L];
    const qs = 7.5 * S.s * 1.0 + 0.6;
    const nl = G.lanes;
    let rank = 0;
    for (const g of queued) {
      let lane, row, along;
      if (g.vt === 4) { lane = -0.5; along = 3; }
      else { lane = rank % nl; row = Math.floor(rank / nl); along = row * qs + qs / 2 + 1; rank++; }
      drawVeh(ctx, G, lane, along, g, nl, false);
    }
    const tailRows = Math.ceil(rank / nl);
    for (const g of moving) {
      const p = (t - g.te) / Math.max(0.1, g.ta - g.te);
      let along = (1 - p) * G.len;
      const lane = g.vt === 4 ? -0.5 : (g.id % nl);
      const tail = tailRows * qs + qs / 2 + 1;
      if (g.vt !== 4 && along < tail) along = tail + (g.id % 3) * 0.8;
      drawVeh(ctx, G, lane, along, g, nl, true);
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
  // pedestrians
  for (let i = 0; i < 9; i++) {
    const n = P.ped[step][i];
    if (!n) continue;
    const r = Math.floor(i / 3), c = i % 3;
    const cx = (S.E + c * S.B) * S.s, cy = (S.E + r * S.B) * S.s;
    const off = (r === 1 ? 3 : 2) * LW + 8;
    ctx.fillStyle = "#a78bfa";
    const shown = Math.min(n, 16);
    for (let k = 0; k < shown; k++) {
      const corner = k % 4, j = Math.floor(k / 4);
      const sx = corner % 2 === 0 ? -1 : 1, sy = corner < 2 ? -1 : 1;
      ctx.beginPath();
      ctx.arc(cx + sx * (2 * LW + 7 + j * 3.2), cy + sy * off, 1.9, 0, 6.283); ctx.fill();
    }
    if (n > 16) { ctx.font = "700 10px 'JetBrains Mono', monospace"; ctx.fillText(String(n), cx + 2 * LW + 7, cy + off + 13); }
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
  const c = counters(S);
  return {q: nQueued, delay: c.delay, co2: c.co2, ped: P.ped[step].reduce((a, b) => a + b, 0), ev: c.ev};
}

function drawVeh(ctx, G, lane, along, g, nl, moving) {
  const latOff = (lane + 0.5) * LW;
  const x = G.stop[0] - G.d[0] * along + G.right[0] * latOff;
  const y = G.stop[1] - G.d[1] * along + G.right[1] * latOff;
  const big = (g.vt === 2 || g.vt === 3);
  const lenPx = big ? 10 : 6.2, wPx = big ? 3.8 : 3.2;
  const horiz = G.d[1] === 0;
  const w = horiz ? lenPx : wPx, h = horiz ? wPx : lenPx;
  if (g.vt === 4) {
    const on = Math.floor(t * 3) % 2 === 0;
    ctx.shadowColor = on ? "#ff4d5e" : "#3d8bff"; ctx.shadowBlur = 14;
    ctx.fillStyle = on ? "#ff4d5e" : "#3d8bff";
    ctx.fillRect(x - w / 2, y - h / 2, w, h); ctx.shadowBlur = 0;
    return;
  }
  ctx.fillStyle = COLORS[g.vt];
  ctx.fillRect(x - w / 2, y - h / 2, w, h);
  if (moving) {                    // headlights in the direction of travel
    ctx.fillStyle = "rgba(255,244,214,0.9)";
    const fx = x + G.d[0] * (lenPx / 2), fy = y + G.d[1] * (lenPx / 2);
    ctx.fillRect(fx - 0.9, fy - 0.9, 1.8, 1.8);
  }
}

function drawSpark(v) {
  const cv = v.spark, ctx = cv.getContext("2d");
  const W = cv.width = cv.clientWidth * 2, H = cv.height = cv.clientHeight * 2;
  const h = v.S.hist, mx = Math.max(1, ...v.yMax);
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
  const cv = document.createElement("canvas"); cv.className = "city"; cv.width = SIZE; cv.height = SIZE;
  cell.appendChild(head); cell.appendChild(kp); cell.appendChild(sb); cell.appendChild(cv); wrap.appendChild(cell);
  return {S, cv, tiles, spark: sp, name: NAMES[k], badge: head.querySelector(".badge"), yMax: yMaxAll};
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
document.getElementById("scrub").oninput = e => { t = Number(e.target.value); render(); };
document.addEventListener("keydown", e => { if (e.code === "Space") { e.preventDefault(); document.getElementById("play").click(); } });
window.addEventListener("resize", render);
t = Math.min(__START__, tMax);
render();
if (__AUTOPLAY__) document.getElementById("play").click();
</script></body></html>
"""


def render_html(payloads: list[dict], names: list[str], start: float = 0.0,
                autoplay: bool = False) -> str:
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
    return html
