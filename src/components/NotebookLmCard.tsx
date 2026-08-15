import React from 'react';
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Download,
  LogIn,
  LogOut,
  Plug,
  RefreshCw,
} from 'lucide-react';
import { Button, Spinner } from './ui';
import { useConfirm } from '../contexts/ConfirmContext';
import { describeState, explainState } from '../services/notebooklm.service';
import { notebookLm, useNotebookLm } from '../services/notebooklm.store';
import styles from '../pages/Launch.module.css';

/**
 * NotebookLM as a capability, not a provider.
 *
 * Three things are deliberate about this card:
 *
 * 1. **Its shape never changes.** Title, status, availability and actions are
 *    present from the very first frame, including while detection is still
 *    running. The previous card rendered a stub saying "Checking what is
 *    installed…" and then replaced itself with something structurally
 *    different, which read as the page rebuilding itself under the user.
 * 2. **It reads application state; it never initialises anything.** Mounting
 *    subscribes to [`notebookLm`] and nothing else. There is no code path from
 *    a render to a sign-in, which is the strongest form the rule "opening
 *    Launch must never start a Google login" can take.
 * 3. **Connected means verified.** A saved session shows as "not verified yet"
 *    until a live call to Google succeeds, because a stale cookie jar is
 *    indistinguishable from a good one until something asks.
 *
 * NotebookLM has no privileged path into the launcher. Once registered it is
 * one more entry in `mcp.json`, and the delivery table below renders it exactly
 * like `git` or `searxng`.
 */
export const NotebookLmCard: React.FC = () => {
  const { status, providers, delivery, busy, progress, error } = useNotebookLm();
  const confirm = useConfirm();

  const eligible = providers.filter((p) => p.compatible);
  const receiving = providers.filter((p) => p.receiving);
  const installed = status.state !== 'notInstalled' && status.state !== 'checking';
  const hasSession = status.hasLocalSession && !status.signedOut;

  const dot =
    status.state === 'connected'
      ? styles.dotLive
      : status.state === 'authenticationExpired' ||
          status.state === 'connectionFailed' ||
          status.state === 'installFailed'
        ? styles.dotAlert
        : status.state === 'checking' ||
            status.state === 'verifying' ||
            status.state === 'authenticating' ||
            status.state === 'installing'
          ? styles.dotBusy
          : '';

  /** Who can use this, phrased for what is actually true right now. */
  const availability = () => {
    if (eligible.length === 0) return 'No provider on this machine speaks MCP yet.';
    if (receiving.length > 0) return receiving.map((p) => p.name).join(' · ');
    return `${eligible.length} compatible provider${eligible.length === 1 ? '' : 's'}: ${eligible
      .map((p) => p.name)
      .join(' · ')}`;
  };

  const signOut = async () => {
    const yes = await confirm({
      title: 'Sign out of NotebookLM?',
      message:
        'The saved Google session is removed from this machine. Providers keep the tool but it will not answer until you sign in again.',
      confirmLabel: 'Sign out',
    });
    if (yes) void notebookLm.signOut();
  };

  return (
    <section className={styles.capability} aria-busy={busy !== null || status.state === 'checking'}>
      <div className={styles.capabilityHead}>
        <div>
          <h3 className={styles.cardTitle}>
            <BookOpen size={15} /> NotebookLM
          </h3>
          <p className={styles.kicker}>Research &amp; knowledge</p>
          <p className={styles.cardDesc}>
            Grounded research over the sources you add to a notebook.
          </p>
        </div>
        <span className={styles.stateChip}>
          <span className={`${styles.dot} ${dot}`} aria-hidden="true" />
          {describeState(status)}
        </span>
      </div>

      {/* A fixed set of facts in a fixed order. Only the values change. */}
      <div className={styles.facts}>
        <div className={styles.fact}>
          <p className={styles.factLabel}>Status</p>
          <p className={styles.factValue}>{describeState(status)}</p>
          <p className={styles.factNote}>{explainState(status)}</p>
        </div>

        <div className={styles.fact}>
          <p className={styles.factLabel}>
            {receiving.length > 0 ? 'Available to' : 'Compatible with'}
          </p>
          <p className={styles.providerList}>{availability()}</p>
          <p className={styles.factNote}>
            Any provider that speaks MCP receives this automatically — including ones added
            later. Nothing here is a fixed list.
          </p>
        </div>

        <div className={styles.fact}>
          <p className={styles.factLabel}>Installation</p>
          <p className={styles.factValue}>
            {status.state === 'checking'
              ? 'Looking…'
              : installed
                ? `notebooklm-py${status.version ? ` ${status.version}` : ''}`
                : 'Not installed'}
          </p>
          <p className={styles.factNote}>
            {status.state === 'checking'
              ? 'Reading what is already on this machine.'
              : status.mcpAvailable
                ? status.inRegistry
                  ? 'MCP server present and offered to providers.'
                  : 'MCP server present, not offered to providers yet.'
                : installed
                  ? 'No MCP server — reinstall with the [mcp] extra.'
                  : 'Nothing has been installed on your behalf.'}
          </p>
        </div>
      </div>

      {/* The running phase, always named. The card's height changes by one line
        * at most, so nothing below it jumps. */}
      <p className={styles.phase} role="status">
        {busy || status.state === 'checking' ? (
          <>
            <Spinner />
            <span>{progress || busy || 'Checking what is installed…'}</span>
          </>
        ) : status.lastVerifiedAt ? (
          <span className={styles.factNote}>
            Session last verified with Google at{' '}
            {new Date(status.lastVerifiedAt).toLocaleString()}
          </span>
        ) : (
          <span className={styles.factNote}>
            Sarathi never sees your Google password, and never stores your session itself.
          </span>
        )}
      </p>

      {status.detail && (
        <div className={styles.warning} role="status">
          <AlertTriangle size={16} />
          <span>{status.detail}</span>
        </div>
      )}
      {error && (
        <div className={styles.warning} role="alert">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* The action area exists in every state. Buttons are disabled rather
        * than removed while work is running, so nothing reflows mid-click. */}
      <div className={styles.capabilityActions}>
        {!installed && (
          <Button
            size="sm"
            disabled={!!busy || status.state === 'checking'}
            onClick={() => void notebookLm.install()}
          >
            <Download size={14} /> Install
          </Button>
        )}

        {installed && (
          <>
            {/* Sign-in is the primary action only when there is genuinely no
              * usable session. A connected or merely unverified card offers it
              * as a quiet secondary, because presenting "Connect" to somebody
              * who is already connected is what made this feel like it kept
              * logging them out. */}
            <Button
              size="sm"
              variant={
                status.state === 'notAuthenticated' || status.state === 'authenticationExpired'
                  ? 'primary'
                  : 'ghost'
              }
              disabled={!!busy}
              onClick={() => void notebookLm.connect()}
            >
              <LogIn size={14} />
              {status.state === 'authenticationExpired'
                ? 'Reconnect'
                : hasSession
                  ? 'Sign in again'
                  : 'Connect / Login'}
            </Button>

            <Button
              size="sm"
              variant="ghost"
              disabled={!!busy}
              onClick={() => void notebookLm.healthCheck()}
            >
              <RefreshCw size={14} /> Health check
            </Button>

            {status.mcpAvailable && (
              <Button
                size="sm"
                variant="ghost"
                disabled={!!busy}
                onClick={() => void notebookLm.setRegistered(!status.inRegistry)}
              >
                <Plug size={14} />
                {status.inRegistry ? 'Withdraw from providers' : 'Offer to providers'}
              </Button>
            )}

            {hasSession && (
              <Button size="sm" variant="ghost" disabled={!!busy} onClick={() => void signOut()}>
                <LogOut size={14} /> Sign out
              </Button>
            )}
          </>
        )}
      </div>

      {/* What each provider was actually handed. Reads the generated configs,
        * not the registry, so "configured" can never be displayed as
        * "delivered". Folded away because it is evidence, not the headline. */}
      {delivery.length > 0 && (
        <details className={styles.details}>
          <summary>What each provider receives</summary>
          <div className={styles.detailsBody}>
            <table className={styles.clients}>
              <caption className={styles.factNote}>
                MCP servers written into each provider&rsquo;s config at launch. Sarathi writes
                the config; the provider makes the connection.
              </caption>
              <tbody>
                {delivery.map((row) => (
                  <tr key={row.toolId}>
                    <th scope="row">{row.toolName}</th>
                    <td>
                      {row.supported ? (
                        <code className={styles.tagMuted}>{row.key}</code>
                      ) : (
                        <span className={styles.tagMuted}>no MCP client</span>
                      )}
                    </td>
                    <td>
                      {row.supported && row.delivered.length > 0 && (
                        <span className={styles.tag}>
                          <CheckCircle2 size={12} /> {row.delivered.join(', ')}
                        </span>
                      )}
                      {row.supported && row.delivered.length === 0 && (
                        <span className={styles.tagMuted}>none configured</span>
                      )}
                      {!row.supported && <span className={styles.tagMuted}>{row.reason}</span>}
                      {row.dropped.length > 0 && row.supported && (
                        <span className={styles.tagMuted}>
                          {' '}
                          · cannot take: {row.dropped.join(', ')}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {status.mcpServerPath && (
              <p className={styles.factNote}>
                Server command: <code>{status.mcpServerPath}</code>
              </p>
            )}
            <div>
              <Button
                size="sm"
                variant="ghost"
                disabled={!!busy}
                onClick={() => void notebookLm.redetect()}
              >
                <RefreshCw size={14} /> Look again for the installation
              </Button>
            </div>
          </div>
        </details>
      )}
    </section>
  );
};
