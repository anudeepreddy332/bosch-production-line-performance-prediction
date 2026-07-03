import type { ReactNode } from "react";
import CountUpNumber from "./CountUpNumber";

interface StatCardProps {
  /** A pre-formatted string (e.g. a range like "0.06–0.18") always renders as-is. A number
   * only animates if countUp is set -- otherwise it's shown static via toLocaleString. */
  value: number | string;
  label: ReactNode;
  tone?: "default" | "good" | "warn" | "danger";
  countUp?: boolean;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  /** Flashes the card background once, for values that just changed live (Decision Explorer). */
  live?: boolean;
  title?: string;
  tight?: boolean;
}

const TONE_CLASS: Record<NonNullable<StatCardProps["tone"]>, string> = {
  default: "",
  good: "stat-good",
  warn: "stat-warn",
  danger: "stat-danger",
};

export default function StatCard({
  value,
  label,
  tone = "default",
  countUp = false,
  decimals = 0,
  prefix = "",
  suffix = "",
  live = false,
  title,
}: StatCardProps) {
  const toneClass = TONE_CLASS[tone];
  const className = ["card", "stat", toneClass, live ? "stat-live" : ""].filter(Boolean).join(" ");

  return (
    <div className={className}>
      {typeof value === "number" && countUp ? (
        <CountUpNumber value={value} decimals={decimals} prefix={prefix} suffix={suffix} />
      ) : (
        <span className="stat-value" title={title}>
          {typeof value === "number" ? `${prefix}${value.toLocaleString()}${suffix}` : value}
        </span>
      )}
      <span className="stat-label">{label}</span>
    </div>
  );
}
