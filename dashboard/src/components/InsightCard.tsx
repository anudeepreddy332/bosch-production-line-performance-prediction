import type { ReactNode } from "react";

interface InsightCardProps {
  title: string;
  children: ReactNode;
  icon?: ReactNode;
}

/** Icon + title + a short (<=2 sentence) body. Used for scannable, single-idea cards. */
export default function InsightCard({ title, children, icon }: InsightCardProps) {
  return (
    <div className="card insight-card">
      {icon && (
        <span className="icon-dot" aria-hidden="true">
          {icon}
        </span>
      )}
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
