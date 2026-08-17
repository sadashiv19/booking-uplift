// Shared helpers used by both screens.

const MODEL_COLORS = {
  "S-learner (naive)": "var(--c-s)",
  "T-learner (naive)": "var(--c-t)",
  "T-learner + IPW": "var(--c-ipw)",
  "Attention-DragonNet": "var(--c-dragonnet)",
};

const QUADRANT_LABEL = {
  persuadable: "Persuadable",
  sleeping_dog: "Sleeping dog",
  sure_thing: "Sure thing (leaning)",
  lost_cause: "Lost cause (leaning)",
};

const QUADRANT_COLOR = {
  persuadable: "var(--persuadable)",
  sleeping_dog: "var(--sleeping-dog)",
  sure_thing: "var(--sure-thing)",
  lost_cause: "var(--lost-cause)",
};

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request failed: ${res.status}`);
  }
  return res.json();
}

// Renders a horizontal bar comparison for the four models' predicted uplift.
function renderBars(container, predictions) {
  const values = Object.values(predictions);
  const maxAbs = Math.max(1, ...values.map((v) => Math.abs(v)));
  container.innerHTML = "";
  Object.entries(predictions).forEach(([name, val]) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const widthPct = (Math.abs(val) / maxAbs) * 100;
    const isNeg = val < 0;
    row.innerHTML = `
      <div class="label">${name}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${widthPct}%; background:${MODEL_COLORS[name]};
             ${isNeg ? "right:50%;" : "left:50%;"}"></div>
        <div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--border-strong);"></div>
      </div>
      <div class="val">${val >= 0 ? "+" : ""}${val.toFixed(1)}</div>
    `;
    container.appendChild(row);
  });
}

// Renders the quadrant gauge with a marker at the given predicted value.
function renderGauge(trackEl, markerEl, value) {
  const min = -30, max = 30;
  const zones = [
    { from: min, to: -3, color: "var(--sleeping-dog)" },
    { from: -3, to: 0, color: "var(--lost-cause)" },
    { from: 0, to: 3, color: "var(--sure-thing)" },
    { from: 3, to: max, color: "var(--persuadable)" },
  ];
  // Only rebuild the zones sub-container -- the marker is a sibling
  // element, not a child of it, so it's never destroyed by this rebuild
  // (previously the marker lived inside the same element this function
  // wiped with innerHTML="", which deleted it from the page after the
  // very first render).
  const zonesContainer = trackEl.querySelector(".gauge-zones");
  zonesContainer.innerHTML = "";
  zones.forEach((z) => {
    const zone = document.createElement("div");
    zone.className = "gauge-zone";
    zone.style.width = `${((z.to - z.from) / (max - min)) * 100}%`;
    zone.style.background = z.color;
    zone.style.opacity = "0.35";
    zonesContainer.appendChild(zone);
  });
  const clamped = Math.max(min, Math.min(max, value));
  const pct = ((clamped - min) / (max - min)) * 100;
  markerEl.style.left = `calc(${pct}% - 1px)`;
}

// Minimal dependency-free SVG line chart for the Qini curves.
function renderLineChart(svgEl, series, opts = {}) {
  const width = 640, height = 360, pad = { l: 56, r: 16, t: 16, b: 36 };
  const allX = series.flatMap((s) => s.frac);
  const allY = series.flatMap((s) => s.qini);
  const xMin = 0, xMax = 1;
  const yMin = Math.min(0, ...allY), yMax = Math.max(...allY);

  const sx = (x) => pad.l + (x - xMin) / (xMax - xMin) * (width - pad.l - pad.r);
  const sy = (y) => height - pad.b - (y - yMin) / (yMax - yMin) * (height - pad.t - pad.b);

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="100%" role="img" aria-label="Qini curves comparing four uplift models">`;

  // gridlines
  for (let i = 0; i <= 4; i++) {
    const y = yMin + (i / 4) * (yMax - yMin);
    svg += `<line x1="${pad.l}" y1="${sy(y)}" x2="${width - pad.r}" y2="${sy(y)}" stroke="var(--border)" stroke-width="1"/>`;
    svg += `<text x="${pad.l - 8}" y="${sy(y) + 4}" text-anchor="end" font-size="10" fill="var(--text-muted)" font-family="var(--font-mono)">${Math.round(y)}</text>`;
  }
  for (let i = 0; i <= 5; i++) {
    const x = i / 5;
    svg += `<text x="${sx(x)}" y="${height - pad.b + 18}" text-anchor="middle" font-size="10" fill="var(--text-muted)" font-family="var(--font-mono)">${Math.round(x * 100)}%</text>`;
  }

  series.forEach((s) => {
    const pts = s.frac.map((x, i) => `${sx(x)},${sy(s.qini[i])}`).join(" ");
    const dash = s.dashed ? `stroke-dasharray="5 4"` : "";
    svg += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="${s.dashed ? 1.5 : 2.2}" ${dash}/>`;
  });

  svg += `</svg>`;
  svgEl.innerHTML = svg;
}

function highlightActiveNav() {
  const path = window.location.pathname;
  document.querySelectorAll("nav.screens a").forEach((a) => {
    const href = a.getAttribute("href");
    const isActive = (href === "/" && (path === "/" || path === "/index.html")) || (href !== "/" && path.endsWith(href));
    a.classList.toggle("active", isActive);
  });
}

const THEME_KEY = "uplift-theme";

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = theme === "light" ? "🌙" : "☀️";
}

function initThemeToggle() {
  const stored = localStorage.getItem(THEME_KEY);
  applyTheme(stored || "dark");
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  highlightActiveNav();
  initThemeToggle();
});
