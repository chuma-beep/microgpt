import { useEffect, useState, type RefObject } from "react";

export function useInView(
  ref: RefObject<HTMLElement | null>,
  options?: { threshold?: number },
) {
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
      { threshold: options?.threshold ?? 0.15 },
    );

    const el = ref.current;
    if (el) {
      observer.observe(el);
    }

    return () => observer.disconnect();
  }, []);

  return visible;
}
