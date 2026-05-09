declare global {
  interface Window {
    wasmReady: boolean;
    goInit: (callback: (err: string | null, result: string) => void) => void;
    goTrainStep: () => number;
    goGenerate: (temperature: number) => string;
    goGenerateWithProbs: (temperature: number) => { name: string; probs: number[] };
  }
}

export {};
