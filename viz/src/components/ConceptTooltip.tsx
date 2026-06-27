import { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import CONCEPTS, { type ConceptDef, type InlineLink } from "@/data/concepts";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const SHARED_INLINE_LINKS: InlineLink[] = [
  {
    match: "neural network",
    url: "https://www.3blue1brown.com/lessons/neural-networks",
  },
  {
    match: "attention",
    url: "https://www.3blue1brown.com/lessons/attention",
  },
  {
    match: "training",
    url: "https://www.3blue1brown.com/lessons/neural-networks",
  },
  {
    match: "gradients",
    url: "https://distill.pub/2016/resnet/",
  },
  {
    match: "activation function",
    url: "https://en.wikipedia.org/wiki/Activation_function",
  },
  {
    match: "normalisation",
    url: "https://en.wikipedia.org/wiki/Layer_normalization",
  },
  {
    match: "normalization",
    url: "https://en.wikipedia.org/wiki/Layer_normalization",
  },
];

function renderWithLinks(
  text: string,
  links: InlineLink[],
): React.ReactNode {
  if (!links.length) return text;

  const sorted = [...links].sort((a, b) => b.match.length - a.match.length);
  const pattern = sorted
    .map((l) => l.match.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  const regex = new RegExp(`(${pattern})`, "gi");

  const parts = text.split(regex);
  return parts.map((part, i) => {
    const link = sorted.find(
      (l) => l.match.toLowerCase() === part.toLowerCase(),
    );
    if (link) {
      return (
        <a
          key={i}
          className="concept-inline-link"
          href={link.url}
          target="_blank"
          rel="noopener noreferrer"
        >
          {part} ↗︎
        </a>
      );
    }
    return part;
  });
}

const KNOWN_CONCEPTS: Record<string, ConceptDef> = CONCEPTS;

export default function ConceptTooltip({
  term,
  children,
}: {
  term: string;
  children: React.ReactNode;
}) {
  const concept = KNOWN_CONCEPTS[term];
  if (!concept) return <>{children}</>;

  const [isOpen, setIsOpen] = useState(false);
  const [isTouch, setIsTouch] = useState(
    () => window.matchMedia("(hover: none)").matches,
  );

  useEffect(() => {
    const mq = window.matchMedia("(hover: none)");
    const handler = (e: MediaQueryListEvent) => setIsTouch(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  useEffect(() => {
    if (!isTouch) setIsOpen(false);
  }, [isTouch]);

  useEffect(() => {
    if (isOpen && isTouch) {
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = "";
      };
    }
  }, [isOpen, isTouch]);

  const allLinks = [
    ...SHARED_INLINE_LINKS,
    ...(concept.inlineLinks ?? []),
  ];

  if (isTouch) {
    return (
      <>
        <span
          className="concept-term"
          data-term={term}
          data-active={isOpen || undefined}
          onClick={() => setIsOpen((p) => !p)}
        >
          {children}
        </span>
        {isOpen &&
          createPortal(
            <>
              <div
                className="concept-sheet-dim"
                onClick={() => setIsOpen(false)}
              />
              <div className="concept-sheet">
                <div className="concept-sheet-handle" />
                <div className="concept-sheet-body">
                  <div className="concept-sheet-header">
                    <span className="concept-label">{term}</span>
                    <button
                      className="concept-sheet-close"
                      onClick={() => setIsOpen(false)}
                    >
                      ✕
                    </button>
                  </div>
                  <span className="concept-body">
                    {renderWithLinks(concept.short, allLinks)}
                  </span>

                  {concept.analogy && (
                    <span className="concept-analogy">{concept.analogy}</span>
                  )}

                  <div className="concept-footer">
                    <a
                      className="concept-footer-link"
                      href={concept.source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      learn more ↗︎
                    </a>
                    <span
                      className={`concept-badge ${concept.source.type}`}
                    >
                      {concept.source.badge}
                    </span>
                  </div>

                  {concept.detail && (
                    <details className="concept-detail">
                      <summary>more</summary>
                      <p>{renderWithLinks(concept.detail, allLinks)}</p>
                    </details>
                  )}
                </div>
              </div>
            </>,
            document.body,
          )}
      </>
    );
  }

  return (
    <TooltipProvider delay={200}>
      <Tooltip>
        <TooltipTrigger
          render={
            <span className="concept-term" data-term={term}>
              {children}
            </span>
          }
        />
        <TooltipContent
          side="top"
          align="center"
          sideOffset={6}
          className="max-w-sm border border-[--rule] shadow-sm rounded-none"
          style={{ backgroundColor: "#FAFAF7", color: "#1B2A4A" }}
        >
          <div className="concept-tooltip-content">
            <span className="concept-label">{term}</span>

            <span className="concept-body">
              {renderWithLinks(concept.short, allLinks)}
            </span>

            {concept.analogy && (
              <span className="concept-analogy">{concept.analogy}</span>
            )}

            <div className="concept-footer">
              <a
                className="concept-footer-link"
                href={concept.source.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                learn more ↗︎
              </a>
              <span className={`concept-badge ${concept.source.type}`}>
                {concept.source.badge}
              </span>
            </div>

            {concept.detail && (
              <details className="concept-detail">
                <summary>more</summary>
                <p>{renderWithLinks(concept.detail, allLinks)}</p>
              </details>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
