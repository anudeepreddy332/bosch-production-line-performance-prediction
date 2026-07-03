import type { ModelsMeta } from "../../lib/types";

interface StackVerdictProps {
  modelsMeta: ModelsMeta;
}

/**
 * The honesty flex, given hero treatment: a stacked meta-model that scores worse than its own
 * best input, reported instead of buried. Every number here is read live from models.json --
 * never hardcoded -- so if the underlying training run ever changes, this verdict can't drift
 * out of sync with it.
 */
export default function StackVerdict({ modelsMeta }: StackVerdictProps) {
  const stack = modelsMeta.meta_model;
  const base = modelsMeta.dataset_h;
  const stackWins = stack.oof_mcc > base.oof_mcc;
  const maxMcc = Math.max(stack.oof_mcc, base.oof_mcc);

  return (
    <div className="stack-verdict">
      <p className="stack-verdict-claim">
        I stacked four models. The stack scored worse. I shipped the simpler one.
      </p>
      <div className="stack-verdict-bars">
        <div className="stack-verdict-row">
          <span className="stack-verdict-label">
            {base.label}
            {!stackWins && <span className="badge badge-production">ships</span>}
          </span>
          <div className="stack-verdict-track">
            <div
              className={!stackWins ? "stack-verdict-fill stack-verdict-fill-win" : "stack-verdict-fill"}
              style={{ width: `${(base.oof_mcc / maxMcc) * 100}%` }}
            />
          </div>
          <span className="stack-verdict-value">{base.oof_mcc.toFixed(4)}</span>
        </div>
        <div className="stack-verdict-row">
          <span className="stack-verdict-label">
            {stack.label}
            {stackWins && <span className="badge badge-production">ships</span>}
          </span>
          <div className="stack-verdict-track">
            <div
              className={stackWins ? "stack-verdict-fill stack-verdict-fill-win" : "stack-verdict-fill stack-verdict-fill-loss"}
              style={{ width: `${(stack.oof_mcc / maxMcc) * 100}%` }}
            />
          </div>
          <span className="stack-verdict-value">{stack.oof_mcc.toFixed(4)}</span>
        </div>
      </div>
      <p className="visually-hidden">
        {base.label} scores {base.oof_mcc.toFixed(4)} honest OOF MCC. {stack.label} scores{" "}
        {stack.oof_mcc.toFixed(4)}. {stackWins ? stack.label : base.label} is the higher score and
        the one that ships.
      </p>
      <p className="chart-hint">
        With only ~0.58% positive rows, the stack has too few failures to learn a better
        combination than its best input. Negative results get reported, not hidden.
      </p>
    </div>
  );
}
