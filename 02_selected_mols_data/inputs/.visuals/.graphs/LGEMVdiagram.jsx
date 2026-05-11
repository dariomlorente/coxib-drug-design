import { useEffect, useRef } from "react";

const W = 620, H = 500;

const M = { cx: 310, cy: 250, rx: 180, ry: 122, rot: 0 };
const L = { cx: 260, cy: 250, rx: 95,  ry: 95,  rot: 0 };
const V = { cx: 360, cy: 250, rx: 95,  ry: 95,  rot: 0 };
const G = { cx: 310, cy: 185, rx: 115, ry: 100, rot: 0 };
const E = { cx: 310, cy: 315, rx: 115, ry: 100, rot: 0 };

function inEllipse(px, py, { cx, cy, rx, ry }) {
  return ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 <= 1;
}

function countRules(px, py) {
  return (
    (inEllipse(px, py, L) ? 1 : 0) +
    (inEllipse(px, py, G) ? 1 : 0) +
    (inEllipse(px, py, E) ? 1 : 0) +
    (inEllipse(px, py, M) ? 1 : 0) +
    (inEllipse(px, py, V) ? 1 : 0)
  );
}

const COLORS = {
  M: "#15803d",  // verde
  L: "#b91c1c",  // rojo
  V: "#92400e",  // marrón ámbar
  G: "#0e7490",  // teal
  E: "#1e40af",  // azul marino
};

const STROKE = { L: 2.2, G: 2.2, V: 2.2, M: 2.2, E: 2.2 };

// Points calculated on actual ellipse borders to avoid overlaps:
// G top:        θ=270° → (310, 84)
// E bottom-left: θ=120° → (200, 404)
// M bottom-right: θ=60° → (408, 371)
// L leftmost:   (155, 250)
// V rightmost:  (465, 250)
// Label positions on each ellipse border:
// G top: (310, 85)
// E bottom: (310, 415)
// M right side: approx (475, 250)
// L leftmost: (165, 250)
// V rightmost: (455, 250)
const LABELS = [
  { id: "G", text: "Ghose",    x: 310,         y: G.cy - G.ry     },
  { id: "E", text: "Egan",     x: 310,         y: E.cy + E.ry     },
  { id: "M", text: "Muegge",   x: 310,         y: E.cy + E.ry - 288},
  { id: "L", text: "Lipinski", x: L.cx - L.rx - 10, y: L.cy            },
  { id: "V", text: "Veber",    x: V.cx + V.rx + 10, y: V.cy            },
];

function Badge({ id, text, x, y }) {
  const color = COLORS[id];
  const charW = 9.2;
  const fw = text.length * charW + 20;
  const fh = 28;
  return (
    <g>
      <rect
        x={x - fw / 2} y={y - fh / 2}
        width={fw} height={fh}
        rx={4} ry={4}
        fill="white"
        stroke={color} strokeWidth={1.6}
      />
      <text
        x={x} y={y + 6}
        textAnchor="middle"
        fontSize={17} fontWeight="800"
        fill={color}
        fontFamily="Aptos, 'Aptos Display', Calibri, sans-serif"
      >{text}</text>
    </g>
  );
}

export default function EulerDrug() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(W, H);
    const d = imgData.data;
    for (let px = 0; px < W; px++) {
      for (let py = 0; py < H; py++) {
        const hasVeber = inEllipse(px, py, V);
        if (!hasVeber) continue;
        const others = (inEllipse(px, py, L) ? 1 : 0) +
                       (inEllipse(px, py, G) ? 1 : 0) +
                       (inEllipse(px, py, E) ? 1 : 0) +
                       (inEllipse(px, py, M) ? 1 : 0);
        if (others >= 3) {
          const idx = (py * W + px) * 4;
          d[idx]   = 251;
          d[idx+1] = 191;
          d[idx+2] = 36;
          d[idx+3] = 75;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, []);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#ffffff",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "Aptos, 'Aptos Display', Calibri, sans-serif",
      padding: "28px 16px",
    }}>
      <h1 style={{
        fontSize: 20, fontWeight: 700, color: "#1a1a1a",
        marginBottom: 4, textAlign: "center", letterSpacing: "0.3px",
      }}>
        Espacio Químico — Reglas de Biodisponibilidad Oral
      </h1>
      <div style={{ position: "relative", width: "100%", maxWidth: W }}>
        <canvas
          ref={canvasRef}
          width={W} height={H}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "auto" }}
        />
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block", position: "relative" }}>

          {/* Strokes only — no fills */}
          <ellipse cx={M.cx} cy={M.cy} rx={M.rx} ry={M.ry} fill="none" stroke={COLORS.M} strokeWidth={STROKE.M} />
          <ellipse cx={G.cx} cy={G.cy} rx={G.rx} ry={G.ry} fill="none" stroke={COLORS.G} strokeWidth={STROKE.G} />
          <ellipse cx={E.cx} cy={E.cy} rx={E.rx} ry={E.ry} fill="none" stroke={COLORS.E} strokeWidth={STROKE.E} />
          <circle  cx={L.cx} cy={L.cy} r={L.rx}            fill="none" stroke={COLORS.L} strokeWidth={STROKE.L} />
          <circle  cx={V.cx} cy={V.cy} r={V.rx}            fill="none" stroke={COLORS.V} strokeWidth={STROKE.V} />

          {/* Badges ON the lines */}
          {LABELS.map(l => <Badge key={l.id} {...l} />)}

        </svg>
      </div>
    </div>
  );
}
