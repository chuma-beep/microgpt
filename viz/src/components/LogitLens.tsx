import { useCallback, useEffect, useMemo, useState } from "react";

interface TokenProb {
  idx: number;
  char: string;
  prob: number;
}

interface LogitLensData {
  nPositions: number;
  tokens: number[];
  targets: number[];
  labels: string[];
  topK: TokenProb[][];
}

function ProbBar({ value, maxVal }: { value: number; maxVal: number }) {
  const h = Math.max(2, (value / maxVal) * 48);
  return (
    <div
      className="w-4 border-b border-[--ink] shrink-0"
      style={{ height: h, backgroundColor: "#1B2A4A" }}
      title={`${(value * 100).toFixed(1)}%`}
    />
  );
}

export default function LogitLens() {
  const [prefix, setPrefix] = useState("emm");
  const [data, setData] = useState<LogitLensData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
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

  const load = useCallback((p: string) => {
    if (!window.goLogitLens) return;
    try {
      const raw = window.goLogitLens(p);
      const parsed = JSON.parse(raw);
      if (parsed.error) {
        setError(parsed.error);
        setData(null);
      } else {
        setError(null);
        setData(parsed);
      }
    } catch (e) {
      setError(String(e));
      setData(null);
    }
  }, []);

  useEffect(() => {
    if (wasmReady && modelReady && prefix) load(prefix);
  }, [wasmReady, modelReady, prefix, load]);

  const handlePrefixChange = (v: string) => {
    const clean = v.toLowerCase().replace(/[^a-z]/g, "").slice(0, 14);
    setPrefix(clean);
    if (!clean) {
      setData(null);
      setError(null);
    }
  };

  const allChars = useMemo(() => {
    const alphabet = "abcdefghijklmnopqrstuvwxyz";
    const chars: string[] = [];
    for (let i = 0; i < 26; i++) chars.push(alphabet[i]);
    chars.push("BOS");
    chars.push("EOS");
    return chars;
  }, []);

  if (!wasmReady || !modelReady) {
    return (
      <div className="col-span-12 text-[--muted-ink] font-serif text-sm italic">
        Loading model...
      </div>
    );
  }

  return (
    <div className="col-span-12">
      <div className="mb-6">
        <label className="mb-1 block font-mono text-[10px] uppercase tracking-[0.16em] text-[--muted-ink]">
          type a prefix
        </label>
        <input
          value={prefix}
          onChange={(e) => handlePrefixChange(e.target.value)}
          spellCheck={false}
          placeholder="emm"
          className="btn-ink w-full max-w-[200px] border border-[--rule] bg-transparent px-3 py-2 font-mono text-base text-[--ink] outline-none focus:border-[--ink]"
        />
      </div>

      {error && (
        <div className="mb-4 font-mono text-xs text-[--muted-ink] italic">
          {error}
        </div>
      )}

      {data && (
        <div className="overflow-x-auto">
          <table className="w-full max-w-2xl border-collapse font-mono text-sm">
            <thead>
              <tr className="text-left">
                <th className="border-b border-[--ink] px-2 py-2 text-[10px] font-normal uppercase tracking-[0.14em] text-[--muted-ink] w-10">
                  pos
                </th>
                <th className="border-b border-[--ink] px-2 py-2 text-[10px] font-normal uppercase tracking-[0.14em] text-[--muted-ink] w-12">
                  char
                </th>
                <th className="border-b border-[--ink] px-2 py-2 text-[10px] font-normal uppercase tracking-[0.14em] text-[--muted-ink]">
                  top-5 predicted next
                </th>
                <th className="border-b border-[--ink] px-2 py-2 text-[10px] font-normal uppercase tracking-[0.14em] text-[--muted-ink] w-16">
                  actual
                </th>
              </tr>
            </thead>
            <tbody>
              {data.topK.map((top5, pos) => {
                const targetIdx = data.targets[pos];
                const targetChar = allChars[targetIdx] ?? "?";
                const topMatch = top5.find((t) => t.idx === targetIdx);
                const correct = topMatch !== undefined;

                return (
                  <tr key={pos}>
                    <td className="border-b border-[--rule] px-2 py-2 text-[--muted-ink]">
                      {pos}
                    </td>
                    <td className="border-b border-[--rule] px-2 py-2">
                      {data.labels[pos]}
                    </td>
                    <td className="border-b border-[--rule] px-2 py-2">
                      <button
                        onClick={() =>
                          setExpandedRow(expandedRow === pos ? null : pos)
                        }
                        className="flex flex-wrap items-center gap-x-3 gap-y-[2px] text-left hover:opacity-70 transition-opacity"
                      >
                        {top5.map((t, k) => (
                          <span
                            key={k}
                            className="inline-flex items-baseline gap-[3px]"
                          >
                            <span
                              className={t.idx === targetIdx ? "font-semibold" : ""}
                            >
                              {t.char}
                            </span>
                            <span className="text-[10px] text-[--muted-ink]">
                              {(t.prob * 100).toFixed(0)}%
                            </span>
                          </span>
                        ))}
                      </button>
                      {expandedRow === pos && (
                        <div className="mt-3 flex flex-wrap items-end gap-[2px] pt-2 border-t border-[--rule]">
                          {top5.map((t) => {
                            const isCorrect = t.idx === targetIdx;
                            return (
                              <div
                                key={t.idx}
                                className="flex flex-col items-center"
                              >
                                <div className="font-mono text-[8px] text-[--muted-ink]">
                                  {(t.prob * 100).toFixed(0)}
                                </div>
                                <ProbBar value={t.prob} maxVal={top5[0]?.prob ?? 1} />
                                <div
                                  className={`mt-[2px] font-mono text-[9px] ${isCorrect ? "font-semibold text-[--ink]" : "text-[--muted-ink]"}`}
                                >
                                  {t.char}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </td>
                    <td className="border-b border-[--rule] px-2 py-2">
                      <span
                        className={
                          correct ? "text-[--ink]" : "text-[--muted-ink]"
                        }
                      >
                        {targetChar}{" "}
                        {correct ? (
                          <span className="text-[10px]">✓</span>
                        ) : (
                          <span className="text-[10px] text-[--muted-ink]">
                            ✗
                          </span>
                        )}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <p className="mt-4 max-w-2xl font-serif text-sm italic leading-[1.7] text-[--muted-ink]">
        What the model believes, before any sampling or temperature. Each row
        shows the top-5 predicted next characters for that position. ✓ means the
        actual next character was in the top 5. Click a row to see the
        probability bars.
      </p>
    </div>
  );
}
