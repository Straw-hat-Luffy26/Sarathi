import { useSyncExternalStore } from 'react';
import {
  mcpDeliveryReport,
  notebookLmHealthCheck,
  notebookLmInstall,
  notebookLmLogin,
  notebookLmLogout,
  notebookLmProviders,
  notebookLmRedetect,
  notebookLmSetRegistered,
  notebookLmState,
  onNotebookLmProgress,
  onNotebookLmStatus,
  type NotebookLmStatus,
  type ProviderFit,
  type ProviderMcpReport,
} from './notebooklm.service';

/**
 * One NotebookLM connection for the whole front end.
 *
 * ## Why this is not a hook, a context or component state
 *
 * React unmounts the Launch page every time the user visits another tab. When
 * the card owned its own `useState`, every return threw the connection away,
 * re-ran detection from nothing, and landed on a screen offering **Connect /
 * Login** to somebody who had signed in minutes earlier. The session was never
 * actually lost — the UI had simply forgotten it, and forgetting is
 * indistinguishable from being signed out if the only button on offer is
 * "sign in".
 *
 * So the state lives at module scope, outside React's lifecycle entirely, and
 * mirrors the backend manager that owns the real thing. Components subscribe;
 * they do not initialise. Mounting a card reads a value. It cannot start a
 * detection, a health check or a sign-in, because none of those are reachable
 * from the render path.
 *
 * ## What is allowed to change the authentication state
 *
 * Only two things: Google, via a live check, and the user, via Sign out.
 * Not a mount, not a tab switch, not a model load, not a re-render.
 */

/** Everything a subscriber can see. Replaced wholesale on every change. */
export interface NotebookLmView {
  status: NotebookLmStatus;
  /** Which providers are eligible, and which are actually receiving it. */
  providers: ProviderFit[];
  /** Per-provider MCP delivery, for the details panel. */
  delivery: ProviderMcpReport[];
  /** Label of the action running, or null. One at a time, by construction. */
  busy: string | null;
  /** Latest progress line from the backend for the running phase. */
  progress: string;
  /** Last action failure, cleared when the next action starts. */
  error: string | null;
}

const UNKNOWN: NotebookLmStatus = {
  state: 'checking',
  mcpAvailable: false,
  inRegistry: false,
  hasLocalSession: false,
  signedOut: false,
  compatibleProviders: [],
  fromCache: false,
};

let view: NotebookLmView = {
  status: UNKNOWN,
  providers: [],
  delivery: [],
  busy: null,
  progress: '',
  error: null,
};

const listeners = new Set<() => void>();
let started = false;

function set(patch: Partial<NotebookLmView>) {
  view = { ...view, ...patch };
  listeners.forEach((l) => l());
}

/**
 * Connects the store to the backend, exactly once for the life of the window.
 *
 * The event subscriptions are registered before the first read, so a state
 * change published while that read is in flight is not lost — which for a
 * detection that finishes in the same tick is the difference between a card
 * that updates and one that sits on "Checking" until something else happens.
 */
async function start() {
  if (started) return;
  started = true;

  await onNotebookLmStatus((status) => {
    set({ status });
    void loadProviders();
  });
  await onNotebookLmProgress((progress) => set({ progress }));

  try {
    set({ status: await notebookLmState() });
  } catch (e) {
    set({ error: String(e) });
  }
  void loadProviders();
}

async function loadProviders() {
  try {
    const [providers, delivery] = await Promise.all([
      notebookLmProviders(),
      mcpDeliveryReport(),
    ]);
    set({ providers, delivery });
  } catch {
    /* supporting detail; its absence must not break the card */
  }
}

/**
 * Runs one user action.
 *
 * Serialised: a second action while one is running is dropped rather than
 * queued. Two sign-ins would mean two browsers; two health checks would mean
 * the user waiting twice for one answer.
 */
async function act(label: string, fn: () => Promise<NotebookLmStatus>) {
  if (view.busy) return;
  set({ busy: label, error: null, progress: '' });
  try {
    set({ status: await fn() });
    await loadProviders();
  } catch (e) {
    set({ error: String(e) });
  } finally {
    set({ busy: null, progress: '' });
  }
}

export const notebookLm = {
  /** Idempotent. Safe to call from every mount, and meant to be. */
  start,
  connect: () => act('Signing in', notebookLmLogin),
  healthCheck: () => act('Checking the connection', notebookLmHealthCheck),
  install: () => act('Installing', notebookLmInstall),
  signOut: () => act('Signing out', notebookLmLogout),
  redetect: () => act('Looking for the installation', notebookLmRedetect),
  setRegistered: (enabled: boolean) =>
    act(enabled ? 'Offering it to providers' : 'Withdrawing it', () =>
      notebookLmSetRegistered(enabled)
    ),
};

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Kicks off the one-time connection on the first subscriber, and is a no-op
  // for every subscriber after. Nothing here authenticates.
  void start();
  return () => {
    listeners.delete(listener);
  };
}

const getSnapshot = () => view;

/** Subscribes a component to the application's NotebookLM state. */
export function useNotebookLm(): NotebookLmView {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/** For tests: the current view without a React tree. */
export function currentView(): NotebookLmView {
  return view;
}
