/**
 * Right column: ISL video playback for the speech -> sign direction.
 *
 * Placeholder. Nothing here is wired to real clips yet — that lands in P10.
 * It deliberately renders an explicit "not built" state rather than a fake
 * player, so the panel can't be mistaken for working functionality in a demo.
 */
interface Props {
  /** The word a clip would play for, once clips exist. */
  cue: string | null;
}

export function SignVideoPanel({ cue }: Props) {
  return (
    <section className="panel panel--video" aria-label="ISL video playback">
      <header className="panel__head">
        <h2 className="panel__title">ISL playback</h2>
        <span className="badge badge--pending">P10</span>
      </header>

      <div className="panel__body panel__body--flush">
        <div className="placeholder">
          <div className="placeholder__frame" aria-hidden>
            <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="2.5" y="5" width="13" height="14" rx="2.5" />
              <path d="M15.5 11.5l6-3.5v8l-6-3.5z" strokeLinejoin="round" />
            </svg>
          </div>
          <p className="placeholder__title">Speech → sign not built yet</p>
          <p className="placeholder__body">
            This panel will play an ISL clip for each spoken word. Wired up in P10.
          </p>
          {cue && (
            <p className="placeholder__cue">
              Would play: <strong>{cue}</strong>
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
