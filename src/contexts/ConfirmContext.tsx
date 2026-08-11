import React, { createContext, useCallback, useContext, useRef, useState, ReactNode } from 'react';
import { ConfirmDialog, type ConfirmRequest } from '../components/ui/ConfirmDialog';

/**
 * Asks the user a yes/no question and waits for the answer.
 *
 * Deliberately promise-shaped, because that is the shape `window.confirm`
 * already had at every call site: `const ok = await confirm(...)` replaces
 * `const ok = window.confirm(...)` without turning straight-line code into a
 * state machine. The difference is the `await`.
 */
export type ConfirmFn = (request: ConfirmRequest) => Promise<boolean>;

const ConfirmContext = createContext<ConfirmFn | undefined>(undefined);

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [request, setRequest] = useState<ConfirmRequest | null>(null);

  // Held in a ref rather than in state: resolving is not a render, and keeping
  // it in state would tie settling the promise to a commit.
  const resolver = useRef<((confirmed: boolean) => void) | null>(null);

  const confirm = useCallback<ConfirmFn>((next) => {
    return new Promise<boolean>((resolve) => {
      // A second question arriving while one is open would otherwise leave the
      // first promise pending forever, and its caller waiting for a click that
      // can no longer happen. The displaced question is answered "no", which is
      // the safe answer for every use here.
      resolver.current?.(false);
      resolver.current = resolve;
      setRequest(next);
    });
  }, []);

  const handleResolve = useCallback((confirmed: boolean) => {
    const resolve = resolver.current;
    resolver.current = null;
    setRequest(null);
    resolve?.(confirmed);
  }, []);

  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      {request && <ConfirmDialog {...request} onResolve={handleResolve} />}
    </ConfirmContext.Provider>
  );
}

/**
 * The application's confirmation dialog.
 *
 * Use this rather than `window.confirm`, which renders a browser chrome dialog
 * captioned with the dev-server origin, ignores the theme, and cannot
 * distinguish a destructive action from an ordinary one.
 */
export function useConfirm(): ConfirmFn {
  const context = useContext(ConfirmContext);
  if (!context) throw new Error('useConfirm must be used within ConfirmProvider');
  return context;
}
