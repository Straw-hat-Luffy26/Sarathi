import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, HelpCircle } from 'lucide-react';
import styles from './ConfirmDialog.module.css';

/** How consequential the action is, which decides the accent and the icon. */
export type ConfirmTone = 'default' | 'danger';

export interface ConfirmRequest {
  title: string;
  /** The prose. Line breaks are preserved as written. */
  message: string;
  /**
   * Something to be read literally rather than as prose — a shell command
   * awaiting approval, for instance. Set in monospace and boxed, because a
   * command wrapped into a paragraph cannot be checked before it is run.
   */
  detail?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  tone?: ConfirmTone;
}

interface Props extends ConfirmRequest {
  onResolve: (confirmed: boolean) => void;
}

/**
 * The application's own confirmation, in place of `window.confirm`.
 *
 * Beyond appearance, three things the browser dialog cannot do and this must:
 * it says what the action is rather than which port asked, it can set a command
 * apart from the sentence around it, and it can make a destructive action look
 * destructive.
 *
 * Focus starts on Cancel. The dialog appears because something irreversible is
 * about to happen, so a stray Enter should not be the thing that confirms it.
 */
export const ConfirmDialog: React.FC<Props> = ({
  title,
  message,
  detail,
  confirmLabel = 'OK',
  cancelLabel = 'Cancel',
  tone = 'default',
  onResolve,
}) => {
  const cancelRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    cancelRef.current?.focus();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onResolve(false);
        return;
      }

      // Focus is kept inside the dialog. A modal question whose buttons can be
      // tabbed away from is answerable by accident from somewhere else.
      if (e.key !== 'Tab' || !panelRef.current) return;
      const focusable = panelRef.current.querySelectorAll<HTMLElement>('button');
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [onResolve]);

  const danger = tone === 'danger';
  const Icon = danger ? AlertTriangle : HelpCircle;

  return createPortal(
    <div
      className={styles.scrim}
      // Dismissing on a backdrop click would make "cancel" the easiest thing to
      // do by accident. It is also the safe answer, so it stays available on
      // Escape and on the button, but not on a stray click.
      role="presentation"
    >
      <div
        ref={panelRef}
        className={styles.panel}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-body"
      >
        <div className={styles.head}>
          <span className={`${styles.icon} ${danger ? styles.iconDanger : ''}`} aria-hidden="true">
            <Icon size={17} />
          </span>
          <h2 className={styles.title} id="confirm-title">
            {title}
          </h2>
        </div>

        <div className={styles.body} id="confirm-body">
          {message}
          {detail && <code className={styles.detail}>{detail}</code>}
        </div>

        <div className={styles.actions}>
          <button
            ref={cancelRef}
            type="button"
            className={`${styles.btn} ${styles.cancel}`}
            onClick={() => onResolve(false)}
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            className={`${styles.btn} ${danger ? styles.confirmDanger : styles.confirm}`}
            onClick={() => onResolve(true)}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
};
