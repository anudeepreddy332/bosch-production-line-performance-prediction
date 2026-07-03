interface DialProps {
  value: number;
  onChange: (value: number) => void;
  tunedValue: number;
  onResetToTuned: () => void;
}

/**
 * The threshold control. A real <input type="range"> rotated to read vertically at the gate --
 * native keyboard support survives the rotation untouched (arrow keys step by `step`, Home/End
 * jump to min/max, screen readers announce it as a slider with the real value/min/max).
 */
export default function Dial({ value, onChange, tunedValue, onResetToTuned }: DialProps) {
  const atTuned = Math.abs(value - tunedValue) < 0.0005;

  return (
    <div className="dial-column">
      <div className="dial-track">
        <input
          type="range"
          className="dial-input"
          min={0}
          max={1}
          step={0.005}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-label="Inspection threshold"
          aria-valuetext={`Threshold ${value.toFixed(3)}`}
        />
      </div>
      <span className="dial-value">{value.toFixed(3)}</span>
      {!atTuned && (
        <button type="button" className="btn btn-secondary dial-reset" onClick={onResetToTuned}>
          Reset to tuned ({tunedValue.toFixed(2)})
        </button>
      )}
    </div>
  );
}
