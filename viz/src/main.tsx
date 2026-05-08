import "./styles.css";
import "katex/dist/katex.min.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

const rootElement = document.getElementById("app");
if (!rootElement) throw new Error("No #app element found");

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
