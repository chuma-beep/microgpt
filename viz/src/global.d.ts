declare global {
  interface Window {
    wasmReady: boolean;
    goGenerate: (temperature: number) => string;
  }
}

export {};
