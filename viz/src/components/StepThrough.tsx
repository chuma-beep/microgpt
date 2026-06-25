import { useCallback, useEffect, useMemo, useState } from "react";
import ConceptTooltip from "@/components/ConceptTooltip";
import { useIsMobile } from "@/hooks/useIsMobile";

interface StepThroughStage {
  name: string;
  dims: number[];
  data: number[];
}

interface StepThroughData {
  nPositions: number;
  nEmb: number;
  headDim: number;
  nHead: number;
  nVocab: number;
  blockSize: number;
  tokens: number[];
  targets: number[];
  labels: string[];
  stages: StepThroughStage[];
}

const STAGE_DESCRIPTIONS: Record<number, React.ReactNode> = {
  0: (
    <>
      Character index looked up in learned <ConceptTooltip term="embedding">embedding</ConceptTooltip> table, added to
      sinusoidal <ConceptTooltip term="positionalEncoding">position vector</ConceptTooltip>.
    </>
  ),
  1: (
    <>
      <ConceptTooltip term="rmsnorm">RMSNorm</ConceptTooltip> rescales each vector so RMS ≈ 1. Learnable gamma{" "}
      <ConceptTooltip term="parameter">parameters</ConceptTooltip> scale each dimension.
    </>
  ),
  2: (
    <>
      Three learned projections (Wq, Wk, Wv) map each position into query, key, value space — the
      building blocks of <ConceptTooltip term="attention">attention</ConceptTooltip>.
    </>
  ),
  3: (
    <>
      Raw dot-product scores Q·Kᵀ / √d for each <ConceptTooltip term="attention">attention head</ConceptTooltip>. Not yet
      normalized.
    </>
  ),
  4: (
    <>
      <ConceptTooltip term="softmax">Softmax</ConceptTooltip> normalizes scores to weights summing to 1.0. Lower triangle is
      causal-masked (position can't look ahead).
    </>
  ),
  5: (
    <>
      Weighted sum of value vectors produces per-head output. Heads concatenated back to 16-dim.
    </>
  ),
  6: (
    <>
      Output projection Wo mixes head outputs. Ready for <ConceptTooltip term="residual">residual</ConceptTooltip> add.
    </>
  ),
  7: (
    <>
      <ConceptTooltip term="rmsnorm">RMSNorm</ConceptTooltip> before the <ConceptTooltip term="MLP">MLP</ConceptTooltip> block. Same operation, separate gamma{" "}
      <ConceptTooltip term="parameter">parameters</ConceptTooltip>.
    </>
  ),
  8: (
    <>
      <ConceptTooltip term="MLP">Fully-connected layer</ConceptTooltip> expands 16 → 64 dimensions. Each neuron sees all 16
      inputs.
    </>
  ),
  9: (
    <>
      <ConceptTooltip term="relu">ReLU</ConceptTooltip> zeroes out negative activations. Introduces non-linearity — without
      this, two linear layers would collapse into one.
    </>
  ),
  10: (
    <>
      Second fc layer compresses 64 → 16. <ConceptTooltip term="residual">Residual</ConceptTooltip> from before{" "}
      <ConceptTooltip term="MLP">MLP</ConceptTooltip> is then added.
    </>
  ),
  11: (
    <>
      LM head projects 16-dim hidden state to 27-dim <ConceptTooltip term="logits">logits</ConceptTooltip> (one score per{" "}
      <ConceptTooltip term="vocabulary">vocabulary</ConceptTooltip> token).
    </>
  ),
  12: (
    <>
      <ConceptTooltip term="softmax">Softmax</ConceptTooltip> converts <ConceptTooltip term="logits">logits</ConceptTooltip> to probabilities. The model's
      belief about the next token.
    </>
  ),
};

function VecRow({
  values,
  label,
  maxAbs,
}: {
  values: number[];
  label?: string;
  maxAbs?: number;
}) {
  const isMobile = useIsMobile(640);
  const cellSize = isMobile ? 18 : 24;
  const fontSize = isMobile ? 8 : 9;
  const max = (maxAbs ?? Math.max(...values.map((v) => Math.abs(v)))) || 1;

  return (
    <div className="flex items-start gap-2">
      {label && (
        <div className="w-24 shrink-0 pt-1 font-mono text-[10px] uppercase tracking-[0.14em] text-[--muted-ink]">
          {label}
        </div>
      )}
      <div className="flex flex-wrap gap-[2px]">
        {values.map((v, i) => {
          const op = Math.abs(v) / max;
          return (
            <div
              key={i}
              className="flex items-center justify-center border border-[--rule] font-mono"
              style={{
                width: cellSize,
                height: cellSize,
                backgroundColor: `rgba(27,42,74,${Math.min(0.9, op * 0.9).toFixed(3)})`,
                color: op > 0.55 ? "#FAFAF7" : "#1B2A4A",
                fontSize,
              }}
              title={v.toFixed(4)}
            >
              {v >= 0 ? "" : "−"}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function BarRow({
  values,
  labels,
}: {
  values: number[];
  labels: string[];
}) {
  const isMobile = useIsMobile(640);
  const maxVal = Math.max(...values) || 1;
  const barH = isMobile ? 48 : 64;

  return (
    <div className="flex items-end gap-[2px] flex-wrap">
      {values.map((v, i) => (
        <div key={i} className="flex flex-col items-center" title={`${labels[i]}: ${(v * 100).toFixed(1)}%`}>
          <div className="font-mono text-[9px] text-[--muted-ink] mb-[2px]">
            {(v * 100).toFixed(0)}
          </div>
          <div
            className="w-5 border-b border-[--ink]"
            style={{ height: Math.max(2, (v / maxVal) * barH), backgroundColor: "#1B2A4A" }}
          />
          <div className="mt-1 font-mono text-[8px] text-[--muted-ink] w-5 text-center truncate">
            {labels[i]}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function StepThrough() {
  const [name, setName] = useState("emma");
  const [data, setData] = useState<StepThroughData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [posIdx, setPosIdx] = useState(0);
  const [stageIdx, setStageIdx] = useState(0);
  const [wasmReady, setWasmReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);

  useEffect(() => {
    if (window.wasmReady) setWasmReady(true);
    else {
      const h = () => setWasmReady(true);
      window.addEventListener("wasmReady", h);
      return () => window.removeEventListener("wasmReady", h);
    }
  }, []);

  useEffect(() => {
    if (window.modelReady) setModelReady(true);
    else {
      const h = () => setModelReady(true);
      window.addEventListener("modelReady", h);
      return () => window.removeEventListener("modelReady", h);
    }
  }, []);

  const loadData = useCallback((n: string) => {
    if (!window.goStepThrough) return;
    try {
      const raw = window.goStepThrough(n);
      const parsed = JSON.parse(raw);
      if (parsed.error) {
        setError(parsed.error);
        setData(null);
      } else {
        setError(null);
        setData(parsed);
        setPosIdx(0);
        setStageIdx(0);
      }
    } catch (e) {
      setError(String(e));
      setData(null);
    }
  }, []);

  useEffect(() => {
    if (wasmReady && modelReady && name) loadData(name);
  }, [wasmReady, modelReady, name, loadData]);

  const handleNameChange = (n: string) => {
    setName(n.toLowerCase().replace(/[^a-z]/g, "").slice(0, 14));
  };

  const goNext = () => {
    if (!data) return;
    setStageIdx((s) => (s + 1) % data.stages.length);
  };
  const goPrev = () => {
    if (!data) return;
    setStageIdx((s) => (s - 1 + data.stages.length) % data.stages.length);
  };

  const stage = data?.stages[stageIdx];
  const totalStages = data?.stages.length ?? 0;
  const isMobile = useIsMobile(640);

  const perPositionFloats = useMemo(() => {
    if (!stage || !data) return 0;
    return stage.data.length / data.nPositions;
  }, [stage, data]);

  const posData = useMemo(() => {
    if (!stage || !data || perPositionFloats === 0) return [];
    const start = Math.floor(posIdx * perPositionFloats);
    const end = start + Math.floor(perPositionFloats);
    return stage.data.slice(start, end);
  }, [stage, data, posIdx, perPositionFloats]);

  const posSubVectors = useMemo(() => {
    if (!stage || posData.length === 0) return [];
    const vecs: number[][] = [];
    let offset = 0;
    for (const dim of stage.dims) {
      vecs.push(posData.slice(offset, offset + dim));
      offset += dim;
    }
    return vecs;
  }, [stage, posData]);

  const vocabLabels = useMemo(() => {
    if (!data) return [];
    const alphabet = "abcdefghijklmnopqrstuvwxyz";
    const labels: string[] = [];
    for (let i = 0; i < data.nVocab; i++) {
      if (i < 26) labels.push(alphabet[i]);
      else labels.push("BOS");
    }
    return labels;
  }, [data]);

  const targetLabel = useMemo(() => {
    if (!data) return "";
    const t = data.targets[posIdx];
    if (t === data.nVocab - 1) return "BOS";
    return vocabLabels[t] ?? "";
  }, [data, posIdx, vocabLabels]);

  if (!wasmReady || !modelReady) {
    return (
      <div className="col-span-12 text-[--muted-ink] font-serif text-sm italic">
        Loading model...
      </div>
    );
  }

  return (
    <div className="col-span-12">
      <div className="mb-6 flex flex-wrap items-end gap-4">
        <div>
          <label className="mb-1 block font-mono text-[10px] uppercase tracking-[0.16em] text-[--muted-ink]">
            input name
          </label>
          <input
            value={name}
            onChange={(e) => handleNameChange(e.target.value)}
            spellCheck={false}
            className="btn-ink w-full max-w-[180px] border border-[--rule] bg-transparent px-3 py-2 font-mono text-base text-[--ink] outline-none focus:border-[--ink]"
          />
        </div>

        {data && (
          <div className="flex flex-wrap items-center gap-1">
            <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-[--muted-ink] mr-1">
              position:
            </span>
            {data.labels.map((label, i) => (
              <button
                key={i}
                onClick={() => setPosIdx(i)}
                className="btn-ink border px-2 py-1 font-mono text-xs"
                style={{
                  borderColor: i === posIdx ? "#1B2A4A" : "#D9D6CC",
                  backgroundColor: i === posIdx ? "#1B2A4A" : "transparent",
                  color: i === posIdx ? "#FAFAF7" : "#1B2A4A",
                }}
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </div>

      {error && (
        <div className="mb-4 font-mono text-xs text-[--muted-ink] italic">
          {error}
        </div>
      )}

      {stage && data && (
        <div className="border border-[--rule] bg-[--paper]">
          <div className="flex items-center justify-between border-b border-[--rule] px-4 py-3">
            <div>
              <div className="font-serif text-base text-[--ink]">
                {stageIdx + 1}. {stage.name}
              </div>
              <div className="mt-1 font-serif text-[13px] italic leading-[1.5] text-[--muted-ink] max-w-xl">
                {STAGE_DESCRIPTIONS[stageIdx] ?? ""}
              </div>
            </div>
            <div className="font-mono text-[10px] text-[--muted-ink] shrink-0 ml-4">
              {stageIdx + 1} / {totalStages}
            </div>
          </div>

          <div className="px-4 py-4">
            {stage.dims[0] === data.nVocab ? (
              <div className="space-y-3">
                <BarRow values={posSubVectors[0] ?? []} labels={vocabLabels} />
                {data.targets[posIdx] !== undefined && (
                  <div className="font-mono text-[11px] text-[--muted-ink]">
                    actual next: <span className="text-[--ink] font-semibold">{targetLabel}</span>
                    {" → "}prob{" "}
                    {((posSubVectors[0]?.[data.targets[posIdx]] ?? 0) * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-2">
                {posSubVectors.map((vec, i) => (
                  <VecRow
                    key={i}
                    values={vec}
                    label={
                      stage.dims.length > 1
                        ? ["Q", "K", "V"][i] ?? `#${i + 1}`
                        : undefined
                    }
                  />
                ))}
              </div>
            )}
          </div>

          <div className="flex items-center justify-between border-t border-[--rule] px-4 py-3">
            <button
              onClick={goPrev}
              className="btn-ink border border-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--ink]"
            >
              ← Prev
            </button>

            <div className="flex gap-1">
              {data.stages.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setStageIdx(i)}
                  className="h-2 transition-all"
                  style={{
                    width: i === stageIdx ? 16 : 8,
                    backgroundColor: i <= stageIdx ? "#1B2A4A" : "#D9D6CC",
                  }}
                  title={data.stages[i].name}
                />
              ))}
            </div>

            <button
              onClick={goNext}
              className="btn-ink border border-[--ink] px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-[--ink]"
            >
              Next →
            </button>
          </div>
        </div>
      )}

      <p className="mt-4 max-w-2xl font-serif text-sm italic leading-[1.7] text-[--muted-ink]">
        Walk through each computation stage of the forward pass. Select a
        position to inspect that token's perspective. The model's belief about
        the next character is shown as a bar chart at the final stage.
      </p>
    </div>
  );
}
