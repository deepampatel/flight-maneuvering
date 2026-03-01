/**
 * Tooltip — Lightweight CSS-only tooltip
 *
 * Dark panel style with amber border, small triangle pointer.
 * No external library required. Wrap any element.
 */

import { useState, useRef, useEffect, type ReactNode } from 'react';

type Position = 'top' | 'bottom' | 'left' | 'right';

interface TooltipProps {
  text: string;
  position?: Position;
  children: ReactNode;
  delay?: number;
}

export function Tooltip({ text, position = 'top', children, delay = 300 }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = () => {
    timerRef.current = setTimeout(() => setVisible(true), delay);
  };

  const hide = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setVisible(false);
  };

  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  return (
    <span
      className="tooltip-wrapper"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      {visible && (
        <span className={`tooltip-bubble tooltip-${position}`} role="tooltip">
          {text}
        </span>
      )}
    </span>
  );
}
