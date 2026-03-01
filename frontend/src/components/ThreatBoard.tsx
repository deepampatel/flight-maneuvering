/**
 * ThreatBoard - Air Defense Threat Tracking Table
 *
 * Displays a compact data table of all tracked threats with engagement
 * decisions, status, and summary counts.
 */

import { useMemo } from 'react';
import type { SimStateEvent, ImpactPrediction, EntityState } from '../types';

interface ThreatBoardProps {
  state: SimStateEvent | null;
}

interface ThreatRow {
  entity: EntityState;
  prediction: ImpactPrediction | null;
  ippLabel: 'ENGAGE' | 'TRACK' | 'IGNORE' | '---';
  status: 'ACTIVE' | 'INTERCEPTED' | 'IMPACT';
  isDecoy: boolean;
}

export function ThreatBoard({ state }: ThreatBoardProps) {
  const threats = useMemo<ThreatRow[]>(() => {
    if (!state?.entities) return [];

    const targets = state.entities.filter(
      (e: EntityState) => e.type === 'target'
    );

    const interceptedTargetIds = new Set<string>();
    if (state.intercepted_pairs) {
      for (const [, targetId] of state.intercepted_pairs) {
        interceptedTargetIds.add(targetId);
      }
    }

    return targets.map((entity) => {
      const prediction: ImpactPrediction | null =
        state.impact_predictions?.[entity.id] ?? null;

      let ippLabel: ThreatRow['ippLabel'] = '---';
      if (prediction) {
        if (prediction.engage) {
          ippLabel = 'ENGAGE';
        } else if (prediction.threatens_area) {
          ippLabel = 'TRACK';
        } else {
          ippLabel = 'IGNORE';
        }
      }

      let status: ThreatRow['status'] = 'ACTIVE';
      if (interceptedTargetIds.has(entity.id)) {
        status = 'INTERCEPTED';
      }
      // If position is at or below ground and not intercepted, mark IMPACT
      if (status === 'ACTIVE' && entity.position.z <= 0) {
        status = 'IMPACT';
      }

      const isDecoy = entity.threat_type === 'decoy';

      return { entity, prediction, ippLabel, status, isDecoy };
    });
  }, [state]);

  const counts = useMemo(() => {
    const total = threats.length;
    const engage = threats.filter((t) => t.ippLabel === 'ENGAGE').length;
    const intercepted = threats.filter((t) => t.status === 'INTERCEPTED').length;
    return { total, engage, intercepted };
  }, [threats]);

  if (!state) {
    return (
      <div className="threat-board">
        <div style={styles.header}>THREAT BOARD</div>
        <div style={styles.noData}>NO DATA</div>
      </div>
    );
  }

  return (
    <div className="threat-board" style={styles.container}>
      <div style={styles.header}>THREAT BOARD</div>

      <div style={styles.tableWrapper}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>ID</th>
              <th style={styles.th}>TYPE</th>
              <th style={styles.thRight}>SPD</th>
              <th style={styles.thRight}>ALT</th>
              <th style={styles.th}>IPP</th>
              <th style={styles.th}>STATUS</th>
            </tr>
          </thead>
          <tbody>
            {threats.length === 0 ? (
              <tr>
                <td colSpan={6} style={styles.emptyRow}>
                  NO TARGETS TRACKED
                </td>
              </tr>
            ) : (
              threats.map((threat) => (
                <tr
                  key={threat.entity.id}
                  style={{
                    ...styles.row,
                    ...(threat.ippLabel === 'ENGAGE' ? styles.rowEngage : {}),
                    ...(threat.isDecoy ? styles.rowDecoy : {}),
                  }}
                >
                  <td style={styles.td}>
                    <span style={styles.idCell}>{threat.entity.id}</span>
                  </td>
                  <td style={styles.td}>
                    <span style={styles.typeCell}>
                      {(threat.entity.threat_type || 'UNK').toUpperCase()}
                    </span>
                  </td>
                  <td style={styles.tdRight}>
                    {threat.entity.speed.toFixed(0)}
                  </td>
                  <td style={styles.tdRight}>
                    {threat.entity.position.z.toFixed(0)}
                  </td>
                  <td style={styles.td}>
                    <span style={ippStyles[threat.ippLabel]}>
                      {threat.ippLabel}
                    </span>
                  </td>
                  <td style={styles.td}>
                    <span style={statusStyles[threat.status]}>
                      {threat.status === 'INTERCEPTED' ? '✕ INTERCEPTED' : threat.status}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div style={styles.summary}>
        <span>
          THREATS: <span style={styles.summaryValue}>{counts.total}</span>
        </span>
        <span style={styles.summaryDivider}>|</span>
        <span>
          ENGAGE: <span style={styles.summaryValueRed}>{counts.engage}</span>
        </span>
        <span style={styles.summaryDivider}>|</span>
        <span>
          INTERCEPTED: <span style={styles.summaryValueGreen}>{counts.intercepted}</span>
        </span>
      </div>
    </div>
  );
}

/* ---------- Inline styles ---------- */

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    fontFamily: "'Inter', system-ui, sans-serif",
    fontSize: '0.7rem',
    letterSpacing: '0.01em',
    color: '#fafafa',
    background: 'rgba(24, 24, 27, 0.85)',
    border: '1px solid rgba(63, 63, 70, 0.5)',
    borderRadius: 6,
  },
  header: {
    padding: '0.35rem 0.6rem',
    fontSize: '0.65rem',
    fontWeight: 600,
    letterSpacing: '0.06em',
    color: '#a1a1aa',
    borderBottom: '1px solid rgba(63, 63, 70, 0.5)',
    textAlign: 'center' as const,
  },
  noData: {
    padding: '1.5rem',
    textAlign: 'center' as const,
    color: '#71717a',
    fontSize: '0.65rem',
  },
  tableWrapper: {
    overflowX: 'auto' as const,
    overflowY: 'auto' as const,
    maxHeight: 260,
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
    tableLayout: 'fixed' as const,
  },
  th: {
    padding: '0.3rem 0.5rem',
    textAlign: 'left' as const,
    fontWeight: 600,
    fontSize: '0.6rem',
    color: '#71717a',
    borderBottom: '1px solid rgba(63, 63, 70, 0.5)',
    whiteSpace: 'nowrap' as const,
    userSelect: 'none' as const,
  },
  thRight: {
    padding: '0.3rem 0.5rem',
    textAlign: 'right' as const,
    fontWeight: 600,
    fontSize: '0.6rem',
    color: '#71717a',
    borderBottom: '1px solid rgba(63, 63, 70, 0.5)',
    whiteSpace: 'nowrap' as const,
    userSelect: 'none' as const,
  },
  row: {
    borderBottom: '1px solid rgba(63, 63, 70, 0.3)',
    transition: 'background 0.15s ease',
  },
  rowEngage: {
    background: 'rgba(239, 68, 68, 0.08)',
  },
  rowDecoy: {
    borderLeft: '2px dotted #71717a',
    borderRight: '2px dotted #71717a',
  },
  td: {
    padding: '0.25rem 0.5rem',
    whiteSpace: 'nowrap' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis' as const,
  },
  tdRight: {
    padding: '0.25rem 0.5rem',
    textAlign: 'right' as const,
    whiteSpace: 'nowrap' as const,
    fontVariantNumeric: 'tabular-nums',
  },
  idCell: {
    color: '#3b82f6',
    fontWeight: 600,
  },
  typeCell: {
    color: '#a1a1aa',
  },
  emptyRow: {
    padding: '1rem',
    textAlign: 'center' as const,
    color: '#71717a',
    fontStyle: 'italic' as const,
  },
  summary: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.6rem',
    padding: '0.35rem 0.6rem',
    fontSize: '0.6rem',
    fontWeight: 600,
    letterSpacing: '0.02em',
    color: '#71717a',
    borderTop: '1px solid rgba(63, 63, 70, 0.5)',
    background: 'rgba(0, 0, 0, 0.15)',
  },
  summaryDivider: {
    color: 'rgba(63, 63, 70, 0.5)',
  },
  summaryValue: {
    color: '#fafafa',
  },
  summaryValueRed: {
    color: '#ef4444',
  },
  summaryValueGreen: {
    color: '#22c55e',
  },
};

const ippStyles: Record<string, React.CSSProperties> = {
  ENGAGE: {
    color: '#ef4444',
    fontWeight: 700,
  },
  TRACK: {
    color: '#eab308',
    fontWeight: 600,
  },
  IGNORE: {
    color: '#71717a',
    fontWeight: 400,
  },
  '---': {
    color: '#52525b',
  },
};

const statusStyles: Record<string, React.CSSProperties> = {
  ACTIVE: {
    color: '#22c55e',
    fontWeight: 600,
  },
  INTERCEPTED: {
    color: '#ef4444',
    textDecoration: 'line-through',
    fontWeight: 600,
  },
  IMPACT: {
    color: '#eab308',
    fontWeight: 700,
  },
};
