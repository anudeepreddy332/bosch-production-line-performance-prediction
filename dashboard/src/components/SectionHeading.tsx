import type { ReactNode } from "react";

interface SectionHeadingProps {
  eyebrow?: string;
  children: ReactNode;
  intro?: ReactNode;
  id?: string;
}

/** Eyebrow + H2 + optional one-line intro. Used to open every major section on a page. */
export default function SectionHeading({ eyebrow, children, intro, id }: SectionHeadingProps) {
  return (
    <>
      {eyebrow && <span className="eyebrow">{eyebrow}</span>}
      <h2 id={id}>{children}</h2>
      {intro && <p className="section-intro">{intro}</p>}
    </>
  );
}
