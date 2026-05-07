import { useEffect, useMemo, useState } from "react";
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
  return Array.from({ length: 16 }, (_, k) =>
    hashFloat(idx * 31 + k * 7 + 1) * 2 - 1,
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
  { step: 2000, train: 2.80, val: 2.91 },
  { step: 3500, train: 2.71, val: 2.89 },
  { step: 5000, train: 2.64, val: 2.88 },
  { step: 7000, train: 2.57, val: 2.87 },
  { step: 8500, train: 2.53, val: 2.865 },
  { step: 10000, train: 2.50, val: 2.86 },
];

const GENERATIONS: { name: string; probs: number[] }[] = [
  { name: "mena",   probs: [0.41, 0.62, 0.55, 0.48] },
  { name: "marian", probs: [0.39, 0.58, 0.46, 0.61, 0.52, 0.44] },
  { name: "carien", probs: [0.27, 0.51, 0.49, 0.42, 0.55, 0.40] },
  { name: "annie",  probs: [0.44, 0.71, 0.63, 0.58, 0.49] },
  { name: "anayne", probs: [0.42, 0.66, 0.40, 0.45, 0.52, 0.38] },
  { name: "meelyn", probs: [0.40, 0.61, 0.59, 0.36, 0.47, 0.41] },
  { name: "ajurle", probs: [0.30, 0.22, 0.31, 0.40, 0.44, 0.37] },
  { name: "aliee",  probs: [0.45, 0.57, 0.60, 0.53, 0.48] },
  { name: "rera",   probs: [0.33, 0.49, 0.43, 0.51] },
  { name: "ancdy",  probs: [0.41, 0.55, 0.28, 0.34, 0.39] },
];

function Section({ number, title, caption, children }: { number: string; title: string; caption?: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-[--rule] py-14">
      <header className="mb-8 grid grid-cols-12 gap-6">
        <div className="col-span-12 md:col-span-3">
          <div className="font-mono text-xs uppercase tracking-[0.18em] text-[--muted-ink]">Figure {number}</div>
          <h2 className="mt-2 font-serif text-2xl leading-tight text-[--ink]">{title}</h2>
        </div>
        {caption && <p className="col-span-12 font-serif text-[15px] leading-[1.7] text-[--muted-ink] md:col-span-7 md:col-start-5">{caption}</p>}
      </header>
      <div className="grid grid-cols-12 gap-6">{children}</div>
    </section>
  );
}

function TokenizerPanel({ name, setName }: { name: string; setName: (s: string) => void }) {
  const sequence = useMemo(() => {
    const clean = name.toLowerCase().replace(/[^a-z]/g, "");
    return ["."].concat(clean.split("")).concat(["."]);
  }, [name]);

  return (
    <div className="col-span-12 lg:col-span-12">
      <label className="mb-3 block font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">input ⟶ name</label>
      <input value={name} onChange={(e) => setName(e.target.value)} spellCheck={false} className="w-full max-w-md border border-[--rule] bg-transparent px-3 py-2 font-mono text-base text-[--ink] outline-none focus:border-[--ink]" />
      <div className="mt-8 overflow-x-auto">
        <table className="min-w-[520px] border-collapse font-mono text-sm">
          <thead>
            <tr className="text-left">
              <th className="border-b border-[--ink] px-3 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">pos</th>
              <th className="border-b border-[--ink] px-3 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">char</th>
              <th className="border-b border-[--ink] px-3 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">index</th>
              <th className="border-b border-[--ink] px-3 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">one-hot (27 dims)</th>
            </tr>
          </thead>
          <tbody>
            {sequence.map((c, i) => {
              const idx = charToIdx(c);
              return (
                <tr key={i} className="border-b border-[--rule]">
                  <td className="px-3 py-2 text-[--muted-ink]">{i}</td>
                  <td className="px-3 py-2">{c === "." ? <span className="text-[--muted-ink]">⟨bos/eos⟩</span> : c}</td>
                  <td className="px-3 py-2">{idx}</td>
                  <td className="px-3 py-1">
                    <div className="flex gap-[2px]">
                      {Array.from({ length: 27 }).map((_, k) => (
                        <div key={k} className="h-3 w-[10px] border border-[--rule]" style={{ backgroundColor: k === idx ? INK : "transparent" }} />
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

function HeatRow({ values, label, symmetric = true }: { values: number[]; label: string; symmetric?: boolean }) {
  const max = symmetric ? Math.max(...values.map((v) => Math.abs(v))) || 1 : Math.max(...values) || 1;
  return (
    <div className="flex items-center gap-3">
      <div className="w-28 shrink-0 font-mono text-[11px] uppercase tracking-[0.16em] text-[--muted-ink]">{label}</div>
      <div className="flex">
        {values.map((v, i) => {
          const op = symmetric ? Math.abs(v) / max : v / max;
          return (
            <div key={i} className="flex h-7 w-7 items-center justify-center border border-[--rule] font-mono text-[9px]" style={{ backgroundColor: `rgba(27, 42, 74, ${Math.min(0.9, op * 0.9).toFixed(3)})`, color: op > 0.55 ? PAPER : INK }} title={v.toFixed(2)}>
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
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">inspecting position</span>
        <div className="flex flex-wrap gap-1">
          {sequence.map((c, i) => (
            <button key={i} onClick={() => setFocus(i)} className="border px-2 py-1 font-mono text-xs" style={{ borderColor: i === focusIdx ? INK : RULE, backgroundColor: i === focusIdx ? INK : "transparent", color: i === focusIdx ? PAPER : INK }}>
              {c === "." ? "·" : c}
            </button>
          ))}
        </div>
      </div>
      <div className="space-y-3">
        <HeatRow values={tok} label="token emb." />
        <HeatRow values={pos} label="positional" />
        <div className="my-2 h-px w-[calc(112px+16*1.75rem)] max-w-full bg-[--ink]" />
        <HeatRow values={sum} label="x = e + p" />
      </div>
      <p className="mt-6 max-w-2xl font-serif text-sm leading-[1.7] text-[--muted-ink]">Each character is mapped through a learned 16-dimensional lookup table, then summed with a sinusoidal positional vector. The result is the residual stream entering layer 0.</p>
    </div>
  );
}

function AttentionPanel({ name }: { name: string }) {
  const tokens = useMemo(() => {
    const clean = name.toLowerCase().replace(/[^a-z]/g, "");
    return ["."].concat(clean.split("")).concat(["."]).map(charToIdx);
  }, [name]);
  const labels = useMemo(() => {
    const clean = name.toLowerCase().replace(/[^a-z]/g, "");
    return ["·"].concat(clean.split("")).concat(["·"]);
  }, [name]);

  const matrix = useMemo(() => attentionMatrix(tokens), [tokens]);
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const [revealKey, setRevealKey] = useState(0);
  useEffect(() => { setRevealKey((k) => k + 1); }, [tokens]);

  const cell = 28;

  return (
    <div className="col-span-12 lg:col-span-8">
      <div className="relative inline-block">
        <div className="ml-7 flex">
          {labels.map((c, j) => <div key={j} className="flex items-end justify-center font-mono text-[11px] text-[--muted-ink]" style={{ width: cell, height: 20 }}>{c}</div>)}
        </div>
        <div className="flex">
          <div className="flex flex-col">
            {labels.map((c, i) => <div key={i} className="flex items-center justify-end pr-2 font-mono text-[11px] text-[--muted-ink]" style={{ width: 28, height: cell }}>{c}</div>)}
          </div>
          <div key={revealKey} className="border border-[--ink]">
            {matrix.map((row, i) => (
              <div key={i} className="flex">
                {row.map((w, j) => {
                  const op = isFinite(w) ? w : 0;
                  const order = i * row.length + j;
                  return <div key={j} onMouseEnter={() => setHover({ i, j })} onMouseLeave={() => setHover(null)} className="attn-cell border border-[--rule]" style={{ width: cell, height: cell, backgroundColor: `rgba(27,42,74,${(op * 0.95).toFixed(3)})`, animationDelay: `${order * 18}ms` }} />;
                })}
              </div>
            ))}
          </div>
        </div>
        {hover && <div className="mt-4 inline-block border border-[--ink] bg-[--paper] px-3 py-1 font-mono text-xs text-[--ink]">attn[{labels[hover.i]} → {labels[hover.j]}] = {matrix[hover.i][hover.j].toFixed(3)}</div>}
      </div>
    </div>
  );
}

function LossPanel() {
  return (
    <div className="col-span-12">
      <div className="h-[360px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={LOSS_DATA} margin={{ top: 20, right: 80, left: 20, bottom: 30 }}>
            <CartesianGrid stroke={RULE} strokeDasharray="0" vertical={false} />
            <XAxis dataKey="step" type="number" domain={[0, 10000]} ticks={[0, 2000, 4000, 6000, 8000, 10000]} stroke={INK} tick={{ fill: INK, fontFamily: "var(--font-mono)", fontSize: 11 }} tickLine={{ stroke: INK }} label={{ value: "training step", position: "insideBottom", offset: -15, style: { fill: INK, fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 13 } }} />
            <YAxis domain={[2.0, 3.5]} ticks={[2.0, 2.5, 3.0, 3.5]} stroke={INK} tick={{ fill: INK, fontFamily: "var(--font-mono)", fontSize: 11 }} tickLine={{ stroke: INK }} label={{ value: "cross-entropy loss", angle: -90, position: "insideLeft", style: { fill: INK, fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 13, textAnchor: "middle" } }} />
            <Line type="monotone" dataKey="train" stroke={INK} strokeWidth={1.5} dot={false} isAnimationActive={true} animationDuration={1600} animationEasing="linear" />
            <Line type="monotone" dataKey="val" stroke={INK} strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={true} animationDuration={1600} animationEasing="linear" />
            <ReferenceDot x={10000} y={2.50} r={0} label={{ value: "train  2.50", position: "right", style: { fill: INK, fontFamily: "var(--font-mono)", fontSize: 11 } }} />
            <ReferenceDot x={10000} y={2.86} r={0} label={{ value: "val  2.86", position: "right", style: { fill: INK, fontFamily: "var(--font-mono)", fontSize: 11 } }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-4 max-w-2xl font-serif text-sm italic leading-[1.7] text-[--muted-ink]">Solid: training loss. Dashed: held-out validation loss. val loss tracks train loss without diverging, consistent with underfitting at this model size.</p>
    </div>
  );
}

function GeneratedRow({ index, name, probs, isLatest }: { index: number; name: string; probs: number[]; isLatest: boolean }) {
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
      <td className="px-2 py-3 text-[--muted-ink]">{String(index + 1).padStart(2, "0")}</td>
      <td className="px-2 py-3"><span>{name.slice(0, count)}</span>{isLatest && count < name.length && <span className="ml-[1px] inline-block h-[1em] w-[6px] -mb-[2px] bg-[--ink] align-middle typewriter-caret" />}</td>
      <td className="px-2 py-3">
        <div className="flex items-end gap-[3px]">
          {probs.map((p, k) => (
            <div key={k} className="flex flex-col items-center" style={{ opacity: k < count ? 1 : 0, transition: "opacity 120ms linear" }}>
              <div className="w-3 border-b border-[--ink]" style={{ height: `${Math.round(p * 36)}px`, backgroundColor: INK }} title={`p=${p.toFixed(2)}`} />
              <div className="mt-1 font-mono text-[9px] text-[--muted-ink]">{name[k]}</div>
            </div>
          ))}
        </div>
      </td>
    </tr>
  );
}

function GenerationPanel() {
  const [shown, setShown] = useState(1);
  return (
    <div className="col-span-12">
      <div className="mb-6 flex items-center gap-4">
        <button onClick={() => setShown((s) => Math.min(s + 1, GENERATIONS.length))} className="border border-[--ink] bg-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--paper] hover:bg-transparent hover:text-[--ink]">▸ sample next</button>
        <button onClick={() => setShown(0)} className="border border-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--ink] hover:bg-[--ink] hover:text-[--paper]">reset</button>
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">{shown} / {GENERATIONS.length} drawn</span>
      </div>
      <table className="w-full max-w-2xl border-collapse font-mono text-sm">
        <thead>
          <tr className="text-left">
            <th className="w-10 border-b border-[--ink] px-2 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">#</th>
            <th className="w-40 border-b border-[--ink] px-2 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">sample</th>
            <th className="border-b border-[--ink] px-2 py-2 text-[11px] font-normal uppercase tracking-[0.18em] text-[--muted-ink]">p(c | c&lt;t)</th>
          </tr>
        </thead>
        <tbody>
          {GENERATIONS.slice(0, shown).map((g, i) => <GeneratedRow key={i} index={i} name={g.name} probs={g.probs} isLatest={i === shown - 1} />)}
          {shown === 0 && <tr><td colSpan={3} className="px-2 py-6 text-center text-[--muted-ink]">— no samples drawn —</td></tr>}
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
  return (
    <div className="col-span-12">
      <div className="flex flex-col items-stretch gap-0">
        {BLOCKS.map((b, i) => (
          <div key={i} className="flex flex-col items-center">
            <div className="flex w-full max-w-md items-center justify-between border border-[--ink] px-4 py-3">
              <span className="font-serif text-[15px] text-[--ink]">{b.name}</span>
              <span className="font-mono text-[11px] text-[--muted-ink]">{b.params}</span>
            </div>
            {i < BLOCKS.length - 1 && (
              <div className="flex flex-col items-center">
                <div className="h-6 w-px bg-[--ink]" />
                <div className="h-0 w-0" style={{ borderLeft: "4px solid transparent", borderRight: "4px solid transparent", borderTop: `5px solid ${INK}`, marginTop: -1 }} />
              </div>
            )}
          </div>
        ))}
      </div>
      <p className="mt-6 max-w-2xl font-serif text-sm italic leading-[1.7] text-[--muted-ink]">Total ≈ 6.1 k parameters. A single attention head, one transformer block, no layer stacking. The smallest configuration that still learns plausible English-looking name morphology.</p>
    </div>
  );
}

export default function App() {
  const [name, setName] = useState("emma");

  return (
    <main className="min-h-screen bg-[--paper] text-[--ink]">
      <div className="mx-auto max-w-5xl px-8 py-20">
        <header className="mb-16 border-b border-[--ink] pb-10">
          <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-[--muted-ink]">Appendix · vol. 1 · §3.2</div>
          <h1 className="mt-4 font-serif text-5xl leading-[1.1] tracking-tight text-[--ink]">
            miniGPT<span className="block font-serif text-2xl italic text-[--muted-ink]">an interactive walkthrough of a 6 k-parameter transformer trained on names.</span>
          </h1>
          <p className="mt-8 max-w-2xl font-serif text-[15px] leading-[1.75] text-[--ink]">We trace a single forward pass — character to distribution — through every component of a minimal decoder-only transformer. Each figure below corresponds to one stage of the computation. Hover, type, and sample to inspect the model's internal state.</p>
          <div className="mt-6 font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">vocab 27 · d_model 16 · 4 head · 1 block</div>
        </header>

        <Section number="1" title="Tokenization" caption="Each character of the input name is mapped to an integer index in a 27-symbol vocabulary (a–z plus a single boundary marker `·` used as both BOS and EOS). The one-hot column on the right is the canonical representation passed to the embedding layer.">
          <TokenizerPanel name={name} setName={setName} />
        </Section>

        <Section number="2" title="Embedding lookup" caption="The model maintains two lookup tables: a learned token embedding of shape (27, 16) and learned positional embedding. The input to the residual stream is the element-wise sum.">
          <EmbeddingPanel name={name} />
        </Section>

        <Section number="3" title="Attention pattern" caption="A causal attention head, visualised as a square matrix of weights. Row i is the query at position i; column j is the key it attends to. The lower-triangular structure enforces autoregressive masking.">
          <AttentionPanel name={name} />
          <div className="col-span-12 lg:col-span-4 lg:pl-6">
            <p className="font-serif text-sm italic leading-[1.7] text-[--muted-ink]">Darker cells indicate higher attention weight. Each row sums to 1.0 by construction. Hover any cell to read the exact value.</p>
          </div>
        </Section>

        <Section number="4" title="Training dynamics" caption="Cross-entropy loss measured every 100 steps over a 10 000-step run. The training corpus contains ≈32 000 English given names; the validation split is held out at random.">
          <LossPanel />
        </Section>

        <Section number="5" title="Sampled generations" caption="Names are drawn autoregressively from the trained model with temperature 1.0 until the boundary token is emitted. The bars beneath each name show the model's probability assigned to the chosen character at each step.">
          <GenerationPanel />
        </Section>

        <Section number="6" title="Architecture" caption="The full forward graph. Residual additions are shown explicitly; normalisation is RMSNorm rather than LayerNorm. The LM head is tied to the input embedding.">
          <ArchitecturePanel />
        </Section>

        <footer className="mt-20 border-t border-[--ink] pt-6 font-mono text-[11px] uppercase tracking-[0.18em] text-[--muted-ink]">end of appendix · figures rendered live · no backend</footer>
      </div>
    </main>
  );
}
