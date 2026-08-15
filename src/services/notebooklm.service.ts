import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

/**
 * How far Sarathi can actually vouch for NotebookLM.
 *
 * There is deliberately no state meaning "probably fine". `unverified` is what
 * a session file on disk earns; only a live call to Google promotes it to
 * `connected`, and only a live refusal demotes it to `authenticationExpired`.
 *
 * The transient states name the work in progress, so the card never shows a
 * spinner over an unexplained pause.
 */
export type NotebookLmState =
  | 'checking'
  | 'notInstalled'
  | 'installing'
  | 'installFailed'
  | 'notAuthenticated'
  | 'authenticating'
  | 'verifying'
  | 'unverified'
  | 'connected'
  | 'authenticationExpired'
  | 'connectionFailed';

export interface NotebookLmStatus {
  state: NotebookLmState;
  version?: string;
  cliPath?: string;
  mcpServerPath?: string;
  /** Whether an MCP server exists to offer providers at all. */
  mcpAvailable: boolean;
  /** Whether it is currently in Sarathi's shared MCP registry. */
  inRegistry: boolean;
  /** A session file exists. Presence only — never its contents. */
  hasLocalSession: boolean;
  /** Includes a verification from a previous run of Sarathi. */
  lastVerifiedAt?: string;
  /** The user chose Sign out, as opposed to never having signed in. */
  signedOut: boolean;
  /** Providers that speak MCP, derived from the registry rather than listed. */
  compatibleProviders: string[];
  /** These numbers came from remembered paths, not a fresh probe. */
  fromCache: boolean;
  /** Scrubbed of anything credential-shaped before it ever reaches here. */
  detail?: string;
}

/** Which providers are eligible for the capability, and which have it. */
export interface ProviderFit {
  id: string;
  name: string;
  compatible: boolean;
  receiving: boolean;
}

/** What one provider was actually handed, as opposed to what is configured. */
export interface ProviderMcpReport {
  toolId: string;
  toolName: string;
  supported: boolean;
  /** The provider's own config key, e.g. `mcp.servers`. */
  key?: string;
  delivered: string[];
  dropped: string[];
  reason?: string;
}

/**
 * The application's current NotebookLM state, and a nudge to start detection
 * if it has not run yet.
 *
 * Cheap by construction: it reads state the backend already holds and returns
 * immediately. It cannot start a sign-in — there is no path from here to a
 * browser — which is what makes it safe to call on every mount.
 */
export async function notebookLmState(): Promise<NotebookLmStatus> {
  return invoke('notebooklm_state');
}

/** Re-runs the full probe. The Refresh button, not the mount. */
export async function notebookLmRedetect(): Promise<NotebookLmStatus> {
  return invoke('notebooklm_redetect');
}

/** Verifies the session against Google. The only thing that yields `connected`. */
export async function notebookLmHealthCheck(): Promise<NotebookLmStatus> {
  return invoke('notebooklm_health_check');
}

/** Installs `notebooklm-py[mcp]`. Only ever from an explicit user action. */
export async function notebookLmInstall(): Promise<NotebookLmStatus> {
  return invoke('notebooklm_install');
}

/**
 * Opens Google's sign-in in a console the user completes themselves.
 *
 * Sarathi never sees the password or any second factor; it runs the CLI's login
 * command and then verifies the result.
 */
export async function notebookLmLogin(): Promise<NotebookLmStatus> {
  return invoke('notebooklm_login');
}

export async function notebookLmLogout(): Promise<NotebookLmStatus> {
  return invoke('notebooklm_logout');
}

/** Adds or removes NotebookLM from the shared MCP registry. */
export async function notebookLmSetRegistered(enabled: boolean): Promise<NotebookLmStatus> {
  return invoke('notebooklm_set_registered', { enabled });
}

/** Providers eligible for the capability, derived from the provider registry. */
export async function notebookLmProviders(): Promise<ProviderFit[]> {
  return invoke('notebooklm_providers');
}

/** Which MCP servers each provider would actually receive right now. */
export async function mcpDeliveryReport(): Promise<ProviderMcpReport[]> {
  return invoke('mcp_delivery_report');
}

/** Every state change, pushed. One event, any number of subscribers. */
export async function onNotebookLmStatus(
  handler: (status: NotebookLmStatus) => void
): Promise<UnlistenFn> {
  return listen<NotebookLmStatus>('notebooklm:status', (e) => handler(e.payload));
}

/** Progress lines for whatever phase is running. */
export async function onNotebookLmProgress(
  handler: (line: string) => void
): Promise<UnlistenFn> {
  return listen<string>('notebooklm:progress', (e) => handler(e.payload));
}

/** The status row's headline. One short phrase, never a sentence. */
export function describeState(status: NotebookLmStatus): string {
  switch (status.state) {
    case 'checking':
      return 'Checking…';
    case 'notInstalled':
      return 'Not installed';
    case 'installing':
      return 'Installing…';
    case 'installFailed':
      return 'Installation failed';
    case 'notAuthenticated':
      return status.signedOut ? 'Signed out' : 'Not connected';
    case 'authenticating':
      return 'Waiting for Google sign-in…';
    case 'verifying':
      return 'Checking connection…';
    case 'unverified':
      return 'Not verified yet';
    case 'connected':
      return 'Connected';
    case 'authenticationExpired':
      return 'Authentication expired';
    case 'connectionFailed':
      return 'Connection check failed';
  }
}

/** The line under the headline: what it means, and what happens next. */
export function explainState(status: NotebookLmStatus): string {
  switch (status.state) {
    case 'checking':
      return 'Looking for the NotebookLM command on this machine.';
    case 'notInstalled':
      return 'NotebookLM research tools are not on this machine yet.';
    case 'installing':
      return 'Installing notebooklm-py with its MCP extra.';
    case 'installFailed':
      return 'Nothing was changed. You can retry, or install the package yourself.';
    case 'notAuthenticated':
      return status.signedOut
        ? 'You signed out. Sign in again whenever you want NotebookLM back.'
        : 'Sign in once with Google. The session is kept until you sign out.';
    case 'authenticating':
      return 'Complete the Google sign-in in the window that opened, then come back.';
    case 'verifying':
      return 'Asking Google whether the saved session is still good. No sign-in needed.';
    case 'unverified':
      return 'A saved sign-in is on this machine and has not been checked yet.';
    case 'connected':
      return 'Ready. Providers can use it without you doing anything else.';
    case 'authenticationExpired':
      return 'Google no longer accepts the saved session. Reconnect to sign in again.';
    case 'connectionFailed':
      return 'The check itself failed — this says nothing about your sign-in. Try again.';
  }
}
