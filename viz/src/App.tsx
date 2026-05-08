import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";

const INK = "#1B2A4A";
const PAPER = "#FAFAF7";
const RULE = "#D9D6CC";

const ALPHABET = "abcdefghijklmnopqrstuvwxyz";
const charToIdx = (c: string) => {
  if (c === ".") return 0;
  const i = ALPHABET.indexOf(c.toLowerCase());
  return i < 0 ? 0 : i + 1;
};

function hashFloat(seed: number) {
  let x = Math.sin(seed * 9301 + 49297) * 233280;
  return x - Math.floor(x);
}

function tokenEmbedding(idx: number): number[] {
  return Array.from(
    { length: 16 },
    (_, k) => hashFloat(idx * 31 + k * 7 + 1) * 2 - 1,
  );
}

function positionalEmbedding(pos: number): number[] {
  return Array.from({ length: 16 }, (_, k) => {
    const denom = Math.pow(10000, (2 * Math.floor(k / 2)) / 16);
    return k % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom);
  });
}

function softmaxRow(scores: number[]) {
  const m = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function attentionMatrix(tokens: number[]) {
  const n = tokens.length;
  const rows: number[][] = [];
  for (let i = 0; i < n; i++) {
    const scores: number[] = [];
    for (let j = 0; j < n; j++) {
      if (j > i) {
        scores.push(-Infinity);
      } else {
        const s =
          (hashFloat(tokens[i] * 13 + tokens[j] * 7 + i + j) - 0.3) * 4 +
          (i === j ? 0.5 : 0);
        scores.push(s);
      }
    }
    rows.push(softmaxRow(scores));
  }
  return rows;
}

const LOSS_DATA = [
  { step: 100, train: 3.27, val: 3.04 },
  { step: 500, train: 3.05, val: 2.98 },
  { step: 1000, train: 2.92, val: 2.94 },
  { step: 2000, train: 2.8, val: 2.91 },
  { step: 3500, train: 2.71, val: 2.89 },
  { step: 5000, train: 2.64, val: 2.88 },
  { step: 7000, train: 2.57, val: 2.87 },
  { step: 8500, train: 2.53, val: 2.865 },
  { step: 10000, train: 2.5, val: 2.86 },
];

const GENERATIONS: { name: string; probs: number[] }[] = [
  { name: "mena", probs: [0.41, 0.62, 0.55, 0.48] },
  { name: "marian", probs: [0.39, 0.58, 0.46, 0.61, 0.52, 0.44] },
  { name: "carien", probs: [0.27, 0.51, 0.49, 0.42, 0.55, 0.4] },
  { name: "annie", probs: [0.44, 0.71, 0.63, 0.58, 0.49] },
  { name: "anayne", probs: [0.42, 0.66, 0.4, 0.45, 0.52, 0.38] },
  { name: "meelyn", probs: [0.4, 0.61, 0.59, 0.36, 0.47, 0.41] },
  { name: "ajurle", probs: [0.3, 0.22, 0.31, 0.4, 0.44, 0.37] },
  { name: "aliee", probs: [0.45, 0.57, 0.6, 0.53, 0.48] },
  { name: "rera", probs: [0.33, 0.49, 0.43, 0.51] },
  { name: "ancdy", probs: [0.41, 0.55, 0.28, 0.34, 0.39] },
];

function Section({
  number,
  title,
  caption,
  children,
}: {
  number: string;
  title: string;
  caption?: string;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (prefersReducedMotion) {
      setVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.2 },
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <section
      ref={ref}
      className={`py-20 ${visible ? "visible" : "section-animate"}`}
    >
      <header className="mb-10 grid grid-cols-12 gap-6">
        <div className="col-span-12 md:col-span-3">
          <div className="font-mono text-xs uppercase tracking-[0.18em] text-[--muted-ink] pt-3 border-t border-[--ink]">
            Figure {number}
          </div>
          <h2 className="mt-3 font-serif text-2xl font-medium leading-tight text-[--ink]">
            {title}
          </h2>
        </div>
        {caption && (
          <p className="col-span-12 font-serif text-[15px] italic leading-[1.7] text-[--muted-ink] md:col-span-7 md:col-start-5">
            {caption}
          </p>
        )}
      </header>
      <div className="grid grid-cols-12 gap-6">{children}</div>
    </section>
  );
}

function TokenizerPanel({
  name,
  setName,
}: {
  name: string;
  setName: (s: string) => void;
}) {
  const sequence = useMemo(() => {
    const clean = name.toLowerCase().replace(/[^a-z]/g, "");
    return ["."].concat(clean.split("")).concat(["."]);
  }, [name]);

  const [animateKey, setAnimateKey] = useState(0);

  return (
    <div className="col-span-12 lg:col-span-12">
      <label className="mb-3 block font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">
        input ⟶ name
      </label>
      <input
        value={name}
        onChange={(e) => {
          setName(e.target.value);
          setAnimateKey((k) => k + 1);
        }}
        spellCheck={false}
        className="w-full max-w-md border border-[--rule] bg-transparent px-3 py-2 font-mono text-base text-[--ink] outline-none focus:border-[--ink]"
      />
      <div className="mt-8 overflow-x-auto">
        <table className="min-w-0 border-collapse font-mono text-sm token-table">
          <thead>
            <tr className="text-left">
              <th className="border-b border-[--ink] px-3 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">
                pos
              </th>
              <th className="border-b border-[--ink] px-3 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">
                char
              </th>
              <th className="border-b border-[--ink] px-3 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">
                index
              </th>
              <th className="border-b border-[--ink] px-3 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">
                one-hot (27 dims)
              </th>
            </tr>
          </thead>
          <tbody>
            {sequence.map((c, i) => {
              const idx = charToIdx(c);
              return (
                <tr
                  key={`${animateKey}-${i}`}
                  className="token-row-animate border-b border-[--rule]"
                  style={{ animationDelay: `${i * 50}ms` }}
                >
                  <td className="px-3 py-2 text-[--muted-ink]">{i}</td>
                  <td className="px-3 py-2">
                    {c === "." ? (
                      <span className="text-[--muted-ink]">⟨bos/eos⟩</span>
                    ) : (
                      c
                    )}
                  </td>
                  <td className="px-3 py-2">{idx}</td>
                  <td className="px-3 py-1">
                    <div className="flex gap-[2px]">
                      {Array.from({ length: 27 }).map((_, k) => (
                        <div
                          key={k}
                          className="h-3 w-[10px] border border-[--rule] heatmap-cell"
                          style={{
                            backgroundColor: k === idx ? INK : "transparent",
                          }}
                        />
                      ))}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function HeatRow({
  values,
  label,
  symmetric = true,
}: {
  values: number[];
  label: string;
  symmetric?: boolean;
}) {
  const max = symmetric
    ? Math.max(...values.map((v) => Math.abs(v))) || 1
    : Math.max(...values) || 1;
  return (
    <div className="flex items-center gap-3 embedding-row">
      <div className="w-28 shrink-0 font-mono text-[11px] uppercase tracking-[0.16em] text-[--muted-ink]">
        {label}
      </div>
      <div className="flex">
        {values.map((v, i) => {
          const op = symmetric ? Math.abs(v) / max : v / max;
          return (
            <div
              key={i}
              className="flex h-7 w-7 items-center justify-center border border-[--rule] font-mono text-[9px]"
              style={{
                backgroundColor: `rgba(27, 42, 74, ${Math.min(0.9, op * 0.9).toFixed(3)})`,
                color: op > 0.55 ? PAPER : INK,
              }}
              title={v.toFixed(2)}
            >
              {v >= 0 ? "" : "−"}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EmbeddingPanel({ name }: { name: string }) {
  const sequence = useMemo(() => {
    const clean = name.toLowerCase().replace(/[^a-z]/g, "");
    return ["."].concat(clean.split("")).concat(["."]);
  }, [name]);

  const [focus, setFocus] = useState(0);
  const focusIdx = Math.min(focus, sequence.length - 1);
  const tokenIdx = charToIdx(sequence[focusIdx]);
  const tok = tokenEmbedding(tokenIdx);
  const pos = positionalEmbedding(focusIdx);
  const sum = tok.map((v, i) => v + pos[i]);

  return (
    <div className="col-span-12">
      <div className="mb-6 flex flex-wrap items-baseline gap-3">
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">
          inspecting position
        </span>
        <div className="flex flex-wrap gap-1">
          {sequence.map((c, i) => (
            <button
              key={i}
              onClick={() => setFocus(i)}
              className="border px-2 py-1 font-mono text-xs"
              style={{
                borderColor: i === focusIdx ? INK : RULE,
                backgroundColor: i === focusIdx ? INK : "transparent",
                color: i === focusIdx ? PAPER : INK,
              }}
            >
              {c === "." ? "·" : c}
            </button>
          ))}
        </div>
      </div>
      <div className="space-y-3">
        <HeatRow values={tok} label="token emb." />
        <HeatRow values={pos} label="positional" />
        <div className="my-2 h-px w-full max-w-full bg-[--ink]" />
        <HeatRow values={sum} label="x = e + p" />
      </div>
      <p className="mt-6 max-w-2xl font-serif text-sm leading-[1.7] text-[--muted-ink]">
        Each character is mapped through a learned 16-dimensional lookup table,
        then summed with a sinusoidal positional vector. The result is the
        residual stream entering layer 0.
      </p>
    </div>
  );
}

function AttentionPanel({ name }: { name: string }) {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 640);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 640);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const tokens = useMemo(() => {
    const clean = name.toLowerCase().replace(/[^a-z]/g, "");
    return ["."].concat(clean.split("")).concat(["."]).map(charToIdx);
  }, [name]);
  const labels = useMemo(() => {
    const clean = name.toLowerCase().replace(/[^a-z]/g, "");
    return ["·"].concat(clean.split("")).concat(["·"]);
  }, [name]);

  const matrix = useMemo(() => attentionMatrix(tokens), [tokens]);
  const [hover, setHover] = useState<{
    i: number;
    j: number;
    value: number;
  } | null>(null);
  const [touched, setTouched] = useState<{ i: number; j: number } | null>(null);
  const cell = isMobile ? 20 : 28;

  return (
    <div className="col-span-12 lg:col-span-8">
      <div className="relative inline-block attention-container">
        <div className="ml-7 flex">
          {labels.map((c, j) => (
            <div
              key={j}
              className="flex items-end justify-center font-mono text-[11px] text-[--muted-ink]"
              style={{ width: cell, height: 20 }}
            >
              {c}
            </div>
          ))}
        </div>
        <div className="flex overflow-x-auto">
          <div className="flex flex-col">
            {labels.map((c, i) => (
              <div
                key={i}
                className="flex items-center justify-end pr-2 font-mono text-[11px] text-[--muted-ink]"
                style={{ width: 28, height: cell }}
              >
                {c}
              </div>
            ))}
          </div>
          <div className="border border-[--ink]">
            {matrix.map((row, i) => (
              <div key={i} className="flex">
                {row.map((w, j) => {
                  const op = isFinite(w) ? w : 0;
                  const order = i * row.length + j;
                  return (
                    <div
                      key={j}
                      onMouseEnter={() => setHover({ i, j, value: w })}
                      onMouseLeave={() => setHover(null)}
                      onTouchStart={() => setTouched({ i, j })}
                      onTouchEnd={() => setTouched(null)}
                      className="attn-cell relative border border-[--rule]"
                      style={{
                        width: cell,
                        height: cell,
                        backgroundColor: `rgba(27,42,74,${(op * 0.95).toFixed(3)})`,
                        animationDelay: `${order * 20}ms`,
                      }}
                    >
                      {(hover && hover.i === i && hover.j === j) ||
                      (touched && touched.i === i && touched.j === j) ? (
                        <div className="attn-tooltip">
                          {matrix[i][j].toFixed(3)}
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function LossPanel() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 640);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 640);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="col-span-12">
      <div className="h-[360px] w-full loss-chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={LOSS_DATA}
            margin={{
              top: 20,
              right: isMobile ? 20 : 80,
              left: 20,
              bottom: 30,
            }}
          >
            <CartesianGrid stroke={RULE} strokeDasharray="0" vertical={false} />
            <XAxis
              dataKey="step"
              type="number"
              domain={[0, 10000]}
              ticks={[0, 2000, 4000, 6000, 8000, 10000]}
              stroke={INK}
              tick={{
                fill: INK,
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                fontStyle: "italic",
              }}
              tickLine={{ stroke: INK }}
              axisLine={{ stroke: INK, strokeWidth: 1 }}
              label={{
                value: "training step",
                position: "insideBottom",
                offset: -15,
                style: {
                  fill: INK,
                  fontFamily: "var(--font-serif)",
                  fontStyle: "italic",
                  fontSize: 12,
                },
              }}
            />
            <YAxis
              domain={[2.0, 3.5]}
              ticks={[2.0, 2.5, 3.0, 3.5]}
              stroke={INK}
              tick={{
                fill: INK,
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                fontStyle: "italic",
              }}
              tickLine={{ stroke: INK }}
              axisLine={{ stroke: INK, strokeWidth: 1 }}
              label={{
                value: "cross-entropy loss",
                angle: -90,
                position: "insideLeft",
                style: {
                  fill: INK,
                  fontFamily: "var(--font-serif)",
                  fontStyle: "italic",
                  fontSize: 12,
                  textAnchor: "middle",
                },
              }}
            />
            <Line
              type="monotone"
              dataKey="train"
              stroke={INK}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="val"
              stroke={INK}
              strokeWidth={2}
              strokeDasharray="4 3"
              dot={false}
              isAnimationActive={false}
            />
            <ReferenceDot
              x={10000}
              y={2.5}
              r={0}
              label={{
                value: "train  2.50",
                position: "right",
                style: {
                  fill: INK,
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                },
              }}
            />
            <ReferenceDot
              x={10000}
              y={2.86}
              r={0}
              label={{
                value: "val  2.86",
                position: "right",
                style: {
                  fill: INK,
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                },
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function GeneratedRow({
  index,
  name,
  probs,
  isLatest,
}: {
  index: number;
  name: string;
  probs: number[];
  isLatest: boolean;
}) {
  const [count, setCount] = useState(isLatest ? 0 : name.length);
  useEffect(() => {
    if (!isLatest) return;
    setCount(0);
    let i = 0;
    const id = window.setInterval(() => {
      i += 1;
      setCount(i);
      if (i >= name.length) window.clearInterval(id);
    }, 80);
    return () => window.clearInterval(id);
  }, [isLatest, name]);

  return (
    <tr className="border-b border-[--rule] align-middle">
      <td className="px-2 py-3 text-[--muted-ink]">
        {String(index + 1).padStart(2, "0")}
      </td>
      <td className="px-2 py-3">
        <span>{name.slice(0, count)}</span>
        {isLatest && count < name.length && (
          <span className="ml-[1px] inline-block h-[1em] w-[6px] -mb-[2px] bg-[--ink] align-middle typewriter-caret" />
        )}
      </td>
      <td className="px-2 py-3">
        <div className="flex items-end gap-[3px]">
          {probs.map((p, k) => (
            <div
              key={k}
              className="flex flex-col items-center"
              style={{
                opacity: k < count ? 1 : 0,
                transition: "opacity 120ms linear",
              }}
            >
              <div
                className="w-3 border-b border-[--ink]"
                style={{
                  height: `${Math.round(p * 36)}px`,
                  backgroundColor: INK,
                }}
                title={`p=${p.toFixed(2)}`}
              />
              <div className="mt-1 font-mono text-[9px] text-[--muted-ink]">
                {name[k]}
              </div>
            </div>
          ))}
        </div>
      </td>
    </tr>
  );
}

function GenerationPanel() {
  const [shown, setShown] = useState(1);
  const [wasmReady, setWasmReady] = useState(false);
  const [generations, setGenerations] = useState<
    { name: string; probs: number[] }[]
  >([
    { name: "mena", probs: [0.41, 0.62, 0.55, 0.48] },
    { name: "marian", probs: [0.39, 0.58, 0.46, 0.61, 0.52, 0.44] },
    { name: "carien", probs: [0.27, 0.51, 0.49, 0.42, 0.55, 0.4] },
    { name: "annie", probs: [0.44, 0.71, 0.63, 0.58, 0.49] },
    { name: "anayne", probs: [0.42, 0.66, 0.4, 0.45, 0.52, 0.38] },
    { name: "meelyn", probs: [0.4, 0.61, 0.59, 0.36, 0.47, 0.41] },
    { name: "ajurle", probs: [0.3, 0.22, 0.31, 0.4, 0.44, 0.37] },
    { name: "aliee", probs: [0.45, 0.57, 0.6, 0.53, 0.48] },
    { name: "rera", probs: [0.33, 0.49, 0.43, 0.51] },
    { name: "ancdy", probs: [0.41, 0.55, 0.28, 0.34, 0.39] },
  ]);

  useEffect(() => {
    if (window.wasmReady) {
      setWasmReady(true);
    } else {
      window.addEventListener("wasmReady", () => setWasmReady(true));
    }
  }, []);

  const handleGenerate = async () => {
    const name = await new Promise<string>((resolve) => {
      resolve(window.goGenerate(0.5));
    });
    const newGen = { name, probs: [0.41, 0.62, 0.55, 0.48, 0.52] };
    setGenerations((prev) => [...prev, newGen]);
    setShown((s) => s + 1);
  };

  return (
    <div className="col-span-12">
      <div className="mb-6 flex items-center gap-4 generation-controls">
        <button
          onClick={handleGenerate}
          disabled={!wasmReady}
          className="btn-hover border border-[--ink] bg-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--paper] disabled:cursor-not-allowed disabled:text-[--muted-ink]"
        >
          {wasmReady ? "▸ sample next" : "loading model..."}
        </button>
        <button
          onClick={() => {
            setShown(0);
            setGenerations([]);
          }}
          className="btn-hover border border-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--ink]"
        >
          reset
        </button>
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">
          {shown} / {generations.length} drawn
        </span>
      </div>
      <table className="w-full max-w-2xl border-collapse font-mono text-sm">
        <thead>
          <tr className="text-left">
            <th className="w-10 border-b border-[--ink] px-2 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">
              #
            </th>
            <th className="w-40 border-b border-[--ink] px-2 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">
              sample
            </th>
            <th className="border-b border-[--ink] px-2 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">
              p(c | c&lt;t)
            </th>
          </tr>
        </thead>
        <tbody>
          {generations.slice(0, shown).map((g, i) => (
            <GeneratedRow
              key={i}
              index={i}
              name={g.name}
              probs={g.probs}
              isLatest={i === shown - 1}
            />
          ))}
          {shown === 0 && (
            <tr>
              <td
                colSpan={3}
                className="px-2 py-6 text-center text-[--muted-ink]"
              >
                — no samples drawn —
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

const BLOCKS = [
  { name: "Input Tokens", params: "—" },
  { name: "Embedding", params: "27 × 16  =  432 p" },
  { name: "RMSNorm", params: "16 p" },
  { name: "Attention (1H)", params: "≈ 3.1 k p" },
  { name: "Residual ⊕", params: "—" },
  { name: "MLP (16→64→16)", params: "≈ 2.1 k p" },
  { name: "Residual ⊕", params: "—" },
  { name: "LM Head", params: "16 × 27  =  432 p" },
  { name: "Softmax", params: "—" },
  { name: "Output Distribution", params: "27 dims" },
];

function ArchitecturePanel() {
  const [visibleBlocks, setVisibleBlocks] = useState<boolean[]>([]);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (prefersReducedMotion) {
      setVisibleBlocks(BLOCKS.map(() => true));
      return;
    }
    BLOCKS.forEach((_, i) => {
      setTimeout(() => {
        setVisibleBlocks((prev) => {
          const next = [...prev];
          next[i] = true;
          return next;
        });
      }, i * 100);
    });
  }, []);

  return (
    <div className="col-span-12 arch-diagram-container">
      <div className="flex flex-col items-stretch gap-0 arch-complex">
        {BLOCKS.map((b, i) => (
          <div key={i} className="flex flex-col items-center">
            <div
              className={`block-animate flex w-full max-w-md items-center justify-between border border-[--ink] px-4 py-3 ${visibleBlocks[i] ? "visible" : ""}`}
              style={{ animationDelay: `${i * 100}ms` }}
            >
              <span className="font-serif text-[15px] text-[--ink]">
                {b.name}
              </span>
              <span className="font-mono text-[11px] text-[--muted-ink]">
                {b.params}
              </span>
            </div>
            {i < BLOCKS.length - 1 && visibleBlocks[i] && (
              <div className="flex flex-col items-center">
                <div
                  className="arrow-animate h-6 w-px bg-[--ink] visible"
                  style={{ animationDelay: `${i * 100 + 150}ms` }}
                />
                <div
                  className="h-0 w-0 arrow-animate visible"
                  style={{
                    borderLeft: "4px solid transparent",
                    borderRight: "4px solid transparent",
                    borderTop: `5px solid ${INK}`,
                    marginTop: -1,
                    animationDelay: `${i * 100 + 180}ms`,
                  }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
      <p className="mt-6 max-w-2xl font-serif text-sm italic leading-[1.7] text-[--muted-ink]">
        Total ≈ 6.1 k parameters. A single attention head, one transformer
        block, no layer stacking. The smallest configuration that still learns
        plausible English-looking name morphology.
      </p>
      <div className="hidden arch-simple-list simplified-version">
        {BLOCKS.map((b, i) => (
          <div key={i}>
            <div className="arch-simple-item">
              <div className="font-serif text-[15px] text-[--ink]">
                {b.name}
              </div>
              <div className="font-mono text-[11px] text-[--muted-ink]">
                {b.params}
              </div>
            </div>
            {i < BLOCKS.length - 1 && (
              <div className="arch-simple-arrow">↓</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function InteractiveTrainerDesktop(props: {
  wasmReady: boolean;
  initStatus: string;
  training: boolean;
  step: number;
  loss: number | null;
  lossHistory: { step: number; loss: number }[];
  generated: { step: number; name: string }[];
  showGenerate: boolean;
  setTraining: (v: boolean) => void;
  setStep: (v: number) => void;
  setLoss: (v: number | null) => void;
  setLossHistory: (v: { step: number; loss: number }[]) => void;
  setGenerated: (v: { step: number; name: string }[]) => void;
  setShowGenerate: (v: boolean) => void;
  setInitStatus: (v: string) => void;
  handleReset: () => void;
  startTraining: () => void;
  handleGenerate: () => void;
}) {
  const {
    wasmReady,
    initStatus,
    training,
    step,
    loss,
    lossHistory,
    generated,
    showGenerate,
    setTraining,
    setStep,
    setLoss,
    setLossHistory,
    setGenerated,
    setShowGenerate,
    setInitStatus,
    handleReset,
    startTraining,
    handleGenerate,
  } = props;

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-6">
        <div className="mb-4 flex flex-wrap gap-3 training-controls">
          <button
            onClick={handleReset}
            disabled={!wasmReady}
            className="btn-hover border border-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--ink] disabled:text-[--muted-ink]"
          >
            reset
          </button>
          {!training && step === 0 && (
            <button
              onClick={startTraining}
              disabled={!wasmReady || initStatus !== "ready"}
              className="btn-hover border border-[--ink] bg-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--paper] disabled:text-[--muted-ink]"
            >
              start training
            </button>
          )}
          {training && (
            <button
              onClick={() => setTraining(false)}
              className="btn-hover border border-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--ink]"
            >
              stop
            </button>
          )}
          {!training && step > 0 && step < 10000 && (
            <button
              onClick={() => setTraining(true)}
              disabled={!wasmReady}
              className="btn-hover border border-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--ink] disabled:text-[--muted-ink]"
            >
              resume
            </button>
          )}
          <button
            onClick={handleGenerate}
            disabled={!wasmReady || !showGenerate}
            className="btn-hover border border-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--ink] disabled:text-[--muted-ink] training-generate-btn"
          >
            generate
          </button>
        </div>

        <div className="mb-6 font-mono text-xs uppercase tracking-[0.18em] text-[--muted-ink] training-stats">
          {initStatus}
          {initStatus === "ready" && step > 0 && (
            <span className="ml-4">
              step {step} / 10000 · loss {loss?.toFixed(4) ?? "—"}
            </span>
          )}
        </div>

        {lossHistory.length > 0 && (
          <div className="h-[280px] w-full training-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={lossHistory}
                margin={{ top: 20, right: 40, left: 20, bottom: 30 }}
              >
                <CartesianGrid
                  stroke={RULE}
                  strokeDasharray="0"
                  vertical={false}
                />
                <XAxis
                  dataKey="step"
                  type="number"
                  domain={[0, 10000]}
                  ticks={[0, 2000, 4000, 6000, 8000, 10000]}
                  stroke={INK}
                  tick={{
                    fill: INK,
                    fontFamily: "var(--font-mono)",
                    fontSize: 10,
                    fontStyle: "italic",
                  }}
                  tickLine={{ stroke: INK }}
                  axisLine={{ stroke: INK, strokeWidth: 1 }}
                />
                <YAxis
                  domain={[2.0, 3.5]}
                  ticks={[2.0, 2.5, 3.0, 3.5]}
                  stroke={INK}
                  tick={{
                    fill: INK,
                    fontFamily: "var(--font-mono)",
                    fontSize: 10,
                    fontStyle: "italic",
                  }}
                  tickLine={{ stroke: INK }}
                  axisLine={{ stroke: INK, strokeWidth: 1 }}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke={INK}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div className="col-span-12 lg:col-span-6">
        <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink] mb-3">
          generated samples
        </div>
        <div className="font-mono text-sm">
          {generated.length === 0 && (
            <div className="text-[--muted-ink]">— no samples yet —</div>
          )}
          {generated.map((g, i) => (
            <div key={i} className="border-b border-[--rule] py-1">
              <span className="text-[--muted-ink]">[{g.step}]</span> {g.name}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function InteractiveTrainerSection() {
  const [wasmReady, setWasmReady] = useState(false);
  const [initStatus, setInitStatus] = useState<string>("initializing model...");
  const [training, setTraining] = useState(false);
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState<number | null>(null);
  const [lossHistory, setLossHistory] = useState<
    { step: number; loss: number }[]
  >([]);
  const [generated, setGenerated] = useState<{ step: number; name: string }[]>(
    [],
  );
  const [showGenerate, setShowGenerate] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [mobileInitialized, setMobileInitialized] = useState(false);
  const [mobileNames, setMobileNames] = useState<string[]>([]);

  useEffect(() => {
    if (window.wasmReady) {
      setWasmReady(true);
      window.goInit((err: string | null, result: string) => {
        if (err) {
          setInitStatus(`error: ${err}`);
        } else {
          setInitStatus("ready");
        }
      });
    }
  }, []);

  useEffect(() => {
    const checkMobile = () => {
      const mobile =
        window.innerWidth < 768 ||
        /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
      setIsMobile(mobile);
    };
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const handleReset = () => {
    setTraining(false);
    setStep(0);
    setLoss(null);
    setLossHistory([]);
    setGenerated([]);
    setShowGenerate(false);
    setInitStatus("initializing model...");
    window.goInit((err: string | null, result: string) => {
      if (err) {
        setInitStatus(`error: ${err}`);
      } else {
        setInitStatus("ready");
      }
    });
  };

  const handleMobileGenerate = () => {
    if (!mobileInitialized) {
      setInitStatus("Initializing...");
      window.goInit((err: string | null, result: string) => {
        if (err) {
          setInitStatus("error: " + err);
        } else {
          setInitStatus("ready");
          setMobileInitialized(true);
          const name = window.goGenerate(0.5);
          setMobileNames((prev) => [...prev, name]);
        }
      });
      return;
    }
    const name = window.goGenerate(0.5);
    setMobileNames((prev) => [...prev, name]);
  };

  const startTraining = () => {
    setTraining(true);
    setStep(0);
    setLossHistory([]);
  };

  useEffect(() => {
    if (!training || !wasmReady) return;

    let animationId: number;
    let currentStep = step;

    const loop = () => {
      const lossVal = window.goTrainStep();
      currentStep++;
      setStep(currentStep);
      setLoss(lossVal);

      if (currentStep % 10 === 0) {
        setLossHistory((prev) => [
          ...prev,
          { step: currentStep, loss: lossVal },
        ]);
      }

      if (currentStep % 1000 === 0) {
        const name = window.goGenerate(0.5);
        setGenerated((prev) => [...prev, { step: currentStep, name }]);
      }

      if (currentStep >= 1000 && !showGenerate) {
        setShowGenerate(true);
      }

      if (currentStep < 10000) {
        animationId = requestAnimationFrame(loop);
      } else {
        setTraining(false);
      }
    };

    animationId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animationId);
  }, [training, wasmReady, step, showGenerate]);

  const handleGenerate = () => {
    if (!wasmReady || step < 1000) return;
    const name = window.goGenerate(0.5);
    setGenerated((prev) => [...prev, { step, name }]);
  };

  return (
    <section className="py-20">
      <header className="mb-10 grid grid-cols-12 gap-6">
        <div className="col-span-12 md:col-span-3">
          <div className="font-mono text-xs uppercase tracking-[0.18em] text-[--muted-ink] pt-3 border-t border-[--ink]">
            Figure 7
          </div>
          <h2 className="mt-3 font-serif text-2xl font-medium leading-tight text-[--ink]">
            Interactive training
          </h2>
        </div>
        <p className="col-span-12 font-serif text-[15px] italic leading-[1.7] text-[--muted-ink] md:col-span-7 md:col-start-5">
          Train the model in your browser. The Go/WASM implementation runs
          entirely client-side — every forward pass, backward pass, and
          parameter update happens in real time.
        </p>
      </header>

      {isMobile ? (
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12">
            <button
              onClick={handleMobileGenerate}
              disabled={!wasmReady || initStatus === "Initializing..."}
              className="btn-hover w-full border border-[--ink] bg-[--ink] px-4 py-3 font-mono text-xs uppercase tracking-[0.18em] text-[--paper] disabled:cursor-not-allowed disabled:text-[--muted-ink]"
            >
              Generate Name
            </button>
            <div
              className="mt-3 font-mono text-[11px] italic text-[--muted-ink]"
              style={{ opacity: 0.5 }}
            >
              live training available on desktop
            </div>
          </div>
          <div className="col-span-12 mt-6 font-mono text-sm">
            {mobileNames.length === 0 && (
              <div className="text-[--muted-ink]">— no names yet —</div>
            )}
            {mobileNames.map((name, i) => (
              <div key={i} className="border-b border-[--rule] py-1">
                {name}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <InteractiveTrainerDesktop
          wasmReady={wasmReady}
          initStatus={initStatus}
          training={training}
          step={step}
          loss={loss}
          lossHistory={lossHistory}
          generated={generated}
          showGenerate={showGenerate}
          setTraining={setTraining}
          setStep={setStep}
          setLoss={setLoss}
          setLossHistory={setLossHistory}
          setGenerated={setGenerated}
          setShowGenerate={setShowGenerate}
          setInitStatus={setInitStatus}
          handleReset={handleReset}
          startTraining={startTraining}
          handleGenerate={handleGenerate}
        />
      )}
    </section>
  );
}

export default function App() {
  const [name, setName] = useState("emma");

  return (
    <main className="min-h-screen bg-[--paper] text-[--ink]">
      <div className="fixed left-8 top-0 bottom-0 w-px bg-[--rule] pointer-events-none z-0 margin-rule" />
      <div className="relative z-10 mx-auto max-w-5xl px-8 py-20">
        <header className="mb-16 border-b border-[--ink] pb-10">
          <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">
            Appendix · vol. 1 · §3.2
          </div>
          <h1 className="mt-4 font-serif text-5xl leading-[1.1] tracking-tight text-[--ink]">
            miniGPT
            <span className="block font-serif text-2xl italic text-[--muted-ink]">
              an interactive walkthrough of a 6 k-parameter transformer trained
              on names.
            </span>
          </h1>
          <p className="mt-8 max-w-2xl font-serif text-[15px] leading-[1.75] text-[--ink]">
            We trace a single forward pass — character to distribution — through
            every component of a minimal decoder-only transformer. Each figure
            below corresponds to one stage of the computation. Hover, type, and
            sample to inspect the model's internal state.
          </p>
          <div className="mt-6 font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">
            vocab 27 · d_model 16 · 4 head · 1 block
          </div>
        </header>

        <Section
          number="1"
          title="Tokenization"
          caption="Each character of the input name is mapped to an integer index in a 27-symbol vocabulary (a–z plus a single boundary marker `·` used as both BOS and EOS). The one-hot column on the right is the canonical representation passed to the embedding layer."
        >
          <TokenizerPanel name={name} setName={setName} />
        </Section>

        <Section
          number="2"
          title="Embedding lookup"
          caption="The model maintains two lookup tables: a learned token embedding of shape (27, 16) and learned positional embedding. The input to the residual stream is the element-wise sum."
        >
          <EmbeddingPanel name={name} />
        </Section>

        <Section
          number="3"
          title="Attention pattern"
          caption="A causal attention head, visualised as a square matrix of weights. Row i is the query at position i; column j is the key it attends to. The lower-triangular structure enforces autoregressive masking."
        >
          <AttentionPanel name={name} />
          <div className="col-span-12 lg:col-span-4 lg:pl-6">
            <p className="font-serif text-sm italic leading-[1.7] text-[--muted-ink]">
              Darker cells indicate higher attention weight. Each row sums to
              1.0 by construction. Hover any cell to read the exact value.
            </p>
          </div>
        </Section>

        <Section
          number="4"
          title="Training dynamics"
          caption="Cross-entropy loss measured every 100 steps over a 10 000-step run. The training corpus contains ≈32 000 English given names; the validation split is held out at random."
        >
          <LossPanel />
        </Section>

        <Section
          number="5"
          title="Sampled generations"
          caption="Names are drawn autoregressively from the trained model with temperature 1.0 until the boundary token is emitted. The bars beneath each name show the model's probability assigned to the chosen character at each step."
        >
          <GenerationPanel />
        </Section>

        <Section
          number="6"
          title="Architecture"
          caption="The full forward graph. Residual additions are shown explicitly; normalisation is RMSNorm rather than LayerNorm. The LM head is tied to the input embedding."
        >
          <ArchitecturePanel />
        </Section>

        <InteractiveTrainerSection />

        <footer className="mt-20 border-t border-[--ink] pt-6 font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">
          end of appendix · figures rendered live · no backend
        </footer>
      </div>
    </main>
  );
}
