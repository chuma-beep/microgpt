import CONCEPTS, { type ConceptDef } from "@/data/concepts";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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
          <div className="font-serif text-sm leading-relaxed">
            <span className="block font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--muted-ink)] mb-1">
              {term}
            </span>
            <p className="italic text-[13px] leading-[1.6]">
              {concept.short}
            </p>
            {concept.detail && (
              <details className="mt-2 text-[12px] text-[var(--muted-ink)] leading-[1.5]">
                <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-[0.12em]">
                  more
                </summary>
                <p className="mt-1">{concept.detail}</p>
              </details>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
