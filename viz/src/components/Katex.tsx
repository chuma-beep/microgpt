import katex from "katex";
import { useEffect, useRef } from "react";

export function Katex({
  math,
  inline = false,
}: {
  math: string;
  inline?: boolean;
}) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (ref.current) {
      katex.render(math, ref.current, {
        throwOnError: false,
        displayMode: !inline,
      });
    }
  }, [math, inline]);

  return <span ref={ref} />;
}
