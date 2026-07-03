import { useEffect, useRef } from "react";
import type { SweepPoint } from "../../lib/types";
import { resolveFateAtGate, type Fate } from "../../lib/inspectionMath";

interface PartFieldProps {
  sweep: SweepPoint[];
  point: SweepPoint;
  pDefect: number;
  onMissedShip: () => void;
  onBinArrival: (fate: "caught" | "alarm") => void;
}

const GATE_X_FRAC = 0.58;
const BIN_X_FRAC = 0.72;
const MAX_PARTICLES = 120;
const SPAWN_INTERVAL_MS = 140;
const SPEED_PX_PER_MS = 0.045;
const DIVERT_DURATION_MS = 450;

type Mode = "flowing" | "diverting" | "done";

interface Particle {
  x: number;
  y: number;
  label: "defect" | "healthy";
  fate: Fate | "pending";
  mode: Mode;
  divertStart: number;
  divertFromY: number;
  bornY: number;
}

function readColor(varName: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  return v || fallback;
}

/**
 * The flowing visual. Fully imperative: particle state lives in a ref and is drawn on a canvas
 * inside a requestAnimationFrame loop -- no React state changes per frame, so this never
 * competes with React's own render cycle. React only receives two discrete callbacks (a part
 * shipped without being caught; a part landed in the bin) to drive the non-animated,
 * always-visible OutcomeLedger and the small bin-settle pulse.
 */
export default function PartField({ sweep, point, pDefect, onMissedShip, onBinArrival }: PartFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const pointRef = useRef(point);
  const pDefectRef = useRef(pDefect);
  const rafRef = useRef<number | null>(null);
  const lastSpawnRef = useRef(0);
  const colorsRef = useRef({ caught: "#146a58", missed: "#b3261e", alarm: "#b45309", pass: "#5b6b81", defect: "#b3261e", healthy: "#9aa7b8" });

  useEffect(() => {
    pointRef.current = point;
  }, [point]);

  useEffect(() => {
    pDefectRef.current = pDefect;
  }, [pDefect]);

  // Re-seed the pool whenever the model changes (a new `sweep` array reference).
  useEffect(() => {
    particlesRef.current = [];
  }, [sweep]);

  useEffect(() => {
    colorsRef.current = {
      caught: readColor("--color-fate-caught", "#146a58"),
      missed: readColor("--color-fate-missed", "#b3261e"),
      alarm: readColor("--color-fate-alarm", "#b45309"),
      pass: readColor("--color-fate-pass", "#5b6b81"),
      defect: readColor("--color-fate-missed", "#b3261e"),
      healthy: readColor("--color-fate-pass", "#9aa7b8"),
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let running = true;
    let lastFrame = performance.now();

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);

    const onVisibility = () => {
      running = !document.hidden;
      if (running) lastFrame = performance.now();
    };
    document.addEventListener("visibilitychange", onVisibility);

    const frame = (now: number) => {
      rafRef.current = requestAnimationFrame(frame);
      if (!running) return;
      const dt = Math.min(64, now - lastFrame);
      lastFrame = now;

      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const gateX = w * GATE_X_FRAC;
      const binX = w * BIN_X_FRAC;
      const colors = colorsRef.current;

      // Spawn.
      if (now - lastSpawnRef.current > SPAWN_INTERVAL_MS && particlesRef.current.length < MAX_PARTICLES) {
        lastSpawnRef.current = now;
        const label = Math.random() < pDefectRef.current ? "defect" : "healthy";
        const bornY = h * (0.25 + Math.random() * 0.5);
        particlesRef.current.push({ x: 0, y: bornY, label, fate: "pending", mode: "flowing", divertStart: 0, divertFromY: bornY, bornY });
      }

      // Update + draw.
      ctx.clearRect(0, 0, w, h);

      // Lane guide.
      ctx.strokeStyle = "rgba(91,107,129,0.15)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h * 0.5);
      ctx.lineTo(w, h * 0.5);
      ctx.stroke();

      // Gate marker.
      ctx.strokeStyle = "rgba(30,58,95,0.35)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(gateX, h * 0.1);
      ctx.lineTo(gateX, h * 0.9);
      ctx.stroke();

      const next: Particle[] = [];
      for (const p of particlesRef.current) {
        if (p.mode === "flowing") {
          p.x += SPEED_PX_PER_MS * dt;
          if (p.fate === "pending" && p.x >= gateX) {
            p.fate = resolveFateAtGate(p.label, pointRef.current, Math.random);
            if (p.fate === "caught" || p.fate === "alarm") {
              p.mode = "diverting";
              p.divertStart = now;
              p.divertFromY = p.y;
            }
          }
          if (p.mode === "flowing" && p.x >= w) {
            if (p.fate === "missed") onMissedShip();
            continue; // remove, exited right edge
          }
        } else if (p.mode === "diverting") {
          const t = Math.min(1, (now - p.divertStart) / DIVERT_DURATION_MS);
          const eased = 1 - Math.pow(1 - t, 2);
          p.x = gateX + (binX - gateX) * eased;
          p.y = p.divertFromY + (h * 0.12 - p.divertFromY) * eased;
          if (t >= 1) {
            onBinArrival(p.fate as "caught" | "alarm");
            continue; // remove, arrived at bin
          }
        }
        next.push(p);

        // Draw.
        const color =
          p.fate === "caught"
            ? colors.caught
            : p.fate === "missed"
              ? colors.missed
              : p.fate === "alarm"
                ? colors.alarm
                : p.fate === "pass"
                  ? colors.pass
                  : p.label === "defect"
                    ? colors.defect
                    : colors.healthy;
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.globalAlpha = p.fate === "pending" ? 0.85 : 1;
        ctx.arc(p.x, p.y, p.mode === "diverting" ? 5 : 4, 0, Math.PI * 2);
        ctx.fill();
        if (p.mode === "diverting" && p.fate === "caught") {
          ctx.globalAlpha = 0.5;
          ctx.strokeStyle = colors.caught;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      particlesRef.current = next;
    };

    rafRef.current = requestAnimationFrame(frame);

    return () => {
      running = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      resizeObserver.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <canvas ref={canvasRef} className="part-field-canvas" aria-hidden="true" />;
}
