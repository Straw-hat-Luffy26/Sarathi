import React, { useCallback, useEffect, useState } from 'react';
import {
  HardDrive,
  Trash2,
  RefreshCw,
  Power,
  Play,
  AlertTriangle,
  Layers,
  Download,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { Button, Spinner, DownloadBar } from '../components/ui';
import { useToast } from '../hooks/useToast';
import { useConfirm } from '../contexts/ConfirmContext';
import { useDownloads } from '../hooks/useDownloads';
import {
  getInstalledModels,
  deleteInstalledModel,
  getStorageSummary,
} from '../services/download.service';
import { getInferenceStatus, loadInstalledModel, unloadActiveModel } from '../services/ai.service';
import {
  listInstalledAdapters,
  removeAdapter,
  setAdapterCapability,
  ASSIGNMENT_SOURCE,
  CAPABILITIES,
  CAPABILITY_LABELS,
  type Capability,
  type InstalledAdapter,
} from '../services/adapters.service';
import { formatSize } from '../services/catalog.service';
import {
  GROUP_DESCRIPTIONS,
  GROUP_LABELS,
  MODEL_GROUPS,
  isLoadableGroup,
  type Classification,
  type ModelGroup,
} from '../types/ai';
import styles from './Storage.module.css';

/** Human-readable parameter count: `20.9B`, `7.6B`, `350M`. */
function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  return `${Math.round(n / 1e6)}M`;
}

/**
 * Storage — manage what is on disk.
 *
 * Deliberately does *not* recommend or download models; that belongs in
 * Discover. Having both meant two screens that each did half the job and
 * shared a confusing name.
 *
 * Models are shelved by what their files actually are, which the backend reads
 * from each GGUF header. Nothing here matches on a model's name, so a model
 * downloaded tomorrow is filed correctly without this file changing.
 */
export const Storage: React.FC = () => {
  const { addToast } = useToast();
  const confirm = useConfirm();
  const [models, setModels] = useState<any[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  /** Adapters per model id, loaded when a model is expanded. */
  const [adapters, setAdapters] = useState<Record<string, InstalledAdapter[]>>({});
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  /**
   * Installed models, split onto shelves and ordered.
   *
   * The group comes from the backend, which reads it out of each file's GGUF
   * header — so nothing here knows any model's name, and a kind that is not
   * present simply produces no heading rather than an empty one.
   *
   * A model installed before classification existed has no group; it is filed
   * as `dense`, the ordinary case, rather than being dropped from a list whose
   * job is to account for what is on disk.
   */
  const grouped = React.useMemo(() => {
    const buckets = new Map<ModelGroup, any[]>();
    for (const m of models) {
      const group: ModelGroup = m.classification?.group ?? 'dense';
      const list = buckets.get(group);
      if (list) list.push(m);
      else buckets.set(group, [m]);
    }
    return MODEL_GROUPS.filter((g) => (buckets.get(g)?.length ?? 0) > 0).map(
      (g) => [g, buckets.get(g) as any[]] as const
    );
  }, [models]);

  /**
   * Identifies the newest refresh, so an older one cannot overwrite it.
   *
   * A ref rather than state: it is read and written inside `refresh` and must
   * not itself cause a render, which would start another refresh.
   */
  const latestRefresh = React.useRef(0);
  /** Set when the screen unmounts, so a reply that arrives after cannot set state. */
  const alive = React.useRef(true);

  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      // Anything still in flight is now stale by definition: bumping the token
      // makes it lose the check below rather than land on an unmounted screen.
      latestRefresh.current += 1;
    };
  }, []);

  const refresh = useCallback(async () => {
    const ticket = (latestRefresh.current += 1);
    // Not a cancellation of the backend work — the backend shares one scan
    // between callers, so a second open costs nothing to leave running. This
    // discards its *result*, which is what makes rapid open/close safe.
    const current = () => alive.current && latestRefresh.current === ticket;

    setError(null);
    try {
      const [installed, sum, status] = await Promise.all([
        getInstalledModels(),
        getStorageSummary().catch(() => null),
        getInferenceStatus().catch(() => null),
      ]);
      if (!current()) return;
      setModels(installed ?? []);
      setSummary(sum);
      setLoadedModelId(status?.model?.modelId ?? null);
    } catch (err) {
      if (!current()) return;
      setError(String(err));
    } finally {
      if (current()) setLoading(false);
    }
  }, []);

  // A finished download becomes an installed model, so the list below it has to
  // catch up on its own — otherwise the bar says "Finished" while the model is
  // nowhere to be seen until a manual refresh.
  const onDownloadCompleted = useCallback(
    (d: { modelName: string }) => {
      addToast('success', `${d.modelName} is ready to use`);
      void refresh();
    },
    [addToast, refresh]
  );

  const { downloads, pause, resume, cancel, dismiss } = useDownloads(onDownloadCompleted);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const toggleAdapters = async (m: any) => {
    const next = new Set(expanded);
    if (next.has(m.modelId)) {
      next.delete(m.modelId);
      setExpanded(next);
      return;
    }
    next.add(m.modelId);
    setExpanded(next);

    if (!adapters[m.modelId]) {
      try {
        const found = await listInstalledAdapters(m.providerId, m.modelId);
        setAdapters((prev) => ({ ...prev, [m.modelId]: found.adapters }));
      } catch {
        setAdapters((prev) => ({ ...prev, [m.modelId]: [] }));
      }
    }
  };

  const handleDeleteModel = async (m: any) => {
    // Deleting frees gigabytes and cannot be undone, so it is confirmed and the
    // size is named — "are you sure?" alone does not convey what is at stake.
    const ok = await confirm({
      title: `Delete ${m.modelName}?`,
      message:
        `This frees ${formatSize(m.sizeBytes)} and removes any adapters installed for it. ` +
        `You would need to download it again to use it.`,
      confirmLabel: 'Delete',
      tone: 'danger',
    });
    if (!ok) return;

    setBusy(m.modelId);
    try {
      await deleteInstalledModel(m.providerId, m.modelId, m.quantization);
      addToast('success', `${m.modelName} deleted`);
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

  /**
   * Points a capability at an adapter, or unassigns it.
   *
   * The whole list for this model is refetched rather than patched locally: only
   * one adapter can hold a capability, so assigning one displaces another, and
   * guessing which from here would eventually disagree with the manifest.
   */
  const handleCapabilityChange = async (m: any, a: InstalledAdapter, next: string) => {
    const capability = next === '' ? null : (next as Capability);
    setBusy(a.id);
    try {
      await setAdapterCapability(m.providerId, m.modelId, a.id, capability);
      const found = await listInstalledAdapters(m.providerId, m.modelId);
      setAdapters((prev) => ({ ...prev, [m.modelId]: found.adapters }));
      addToast(
        'success',
        capability
          ? `${a.name} will be used for ${CAPABILITY_LABELS[capability]}`
          : `${a.name} is no longer used`
      );
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

  const handleRemoveAdapter = async (m: any, a: InstalledAdapter) => {
    setBusy(a.id);
    try {
      await removeAdapter(m.providerId, m.modelId, a.id);
      setAdapters((prev) => ({
        ...prev,
        [m.modelId]: (prev[m.modelId] ?? []).filter((x) => x.id !== a.id),
      }));
      addToast('success', 'Adapter removed');
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

  // Loading reads gigabytes into VRAM and takes a while, so the button reports
  // progress and every other model's button locks — two concurrent loads would
  // fight over the same VRAM budget.
  const handleLoad = async (m: any) => {
    setBusy(m.modelId);
    try {
      await loadInstalledModel(m.providerId, m.modelId, m.quantization);
      addToast('success', `${m.modelName} loaded — the gateway can serve requests`);
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

  const handleUnload = async () => {
    try {
      await unloadActiveModel();
      addToast('info', 'Model unloaded — the gateway has nothing to serve until you load one');
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    }
  };

  if (loading) {
    return (
      <div className={styles.centered}>
        <Spinner />
        <p>Reading what is on disk…</p>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Storage</h1>
          <p className={styles.subtitle}>
            Models and adapters on this computer. To find new ones, use Discover.
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={refresh}>
          <RefreshCw size={14} /> Refresh
        </Button>
      </header>

      {error && (
        <div className={styles.error} role="alert">
          <AlertTriangle size={15} /> {error}
        </div>
      )}

      {summary && (
        <div className={styles.summary}>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Models</span>
            <span className={styles.statValue}>{models.length}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Used by models</span>
            <span className={styles.statValue}>{formatSize(summary.totalModelsBytes ?? 0)}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Free on disk</span>
            <span className={styles.statValue}>{formatSize(summary.availableDiskSpaceBytes ?? 0)}</span>
          </div>
        </div>
      )}

      {downloads.length > 0 && (
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>
            <Download size={15} /> Downloading
            <span className={styles.count}>{downloads.length}</span>
          </h2>
          <div className={styles.downloads}>
            {downloads.map((d) => (
              <DownloadBar
                key={d.taskId}
                download={d}
                onPause={(id) => void pause(id)}
                onResume={(id) => void resume(id)}
                onCancel={(id) => void cancel(id)}
                onDismiss={dismiss}
              />
            ))}
          </div>
        </section>
      )}

      {models.length === 0 && (
        <p className={styles.empty}>
          Nothing installed yet. Open <strong>Discover</strong> to find a model that fits your
          hardware.
        </p>
      )}

      {/* One section per kind, in a fixed order, and only for kinds that are
        * actually present. The grouping comes from each file's GGUF header, so
        * a model downloaded tomorrow lands on the right shelf without anything
        * here knowing its name. */}
      {grouped.map(([group, inGroup]) => (
        <section key={group} className={styles.section}>
          <div className={styles.sectionBar}>
            <h2 className={styles.sectionTitle}>
              {GROUP_LABELS[group]}
              <span className={styles.count}>{inGroup.length}</span>
            </h2>
            <p className={styles.sectionNote}>{GROUP_DESCRIPTIONS[group]}</p>
          </div>

          {inGroup.map((m: any) => {
          const isLoaded = loadedModelId === m.modelId;
          const isOpen = expanded.has(m.modelId);
          const mine = adapters[m.modelId];
          const cls: Classification | null = m.classification ?? null;
          const loadable = cls ? isLoadableGroup(cls.group) : true;

          return (
            <article
              key={`${m.modelId}-${m.quantization}`}
              className={`${styles.model} ${isLoaded ? styles.modelLoaded : ''} ${
                loadable ? '' : styles.modelInert
              }`}
            >
              <div className={styles.modelHead}>
                <div className={styles.modelInfo}>
                  <h3 className={styles.modelName}>{m.modelName}</h3>

                  {/* The facts that decide whether this is the right model to
                    * load, in one row: what it is, how it was compressed, what
                    * it costs on disk, and where it came from. Everything here
                    * is read from the file rather than from the manifest. */}
                  <div className={styles.specs}>
                    {cls?.architecture && (
                      <span className={styles.spec} title="Architecture, from the file's own header">
                        {cls.architecture}
                      </span>
                    )}
                    {cls?.isMoe && cls.expertCount > 0 && (
                      <span
                        className={styles.specAccent}
                        title={`${cls.expertCount} experts, ${cls.expertUsedCount} consulted per token`}
                      >
                        {cls.expertCount} experts · {cls.expertUsedCount} active
                      </span>
                    )}
                    {cls?.parameterCount ? (
                      <span className={styles.spec}>{formatParams(cls.parameterCount)}</span>
                    ) : null}
                    <span className={styles.spec}>{m.quantization}</span>
                    <span className={styles.spec}>{formatSize(m.sizeBytes)}</span>
                    <span className={styles.specMuted}>{m.providerId}</span>
                  </div>

                  {/* Why a file that is present cannot be used. Without this the
                    * card would simply have no Load button and no explanation,
                    * which is how a helper file came to look like a model that
                    * was merely broken. */}
                  {cls?.notLoadableReason && (
                    <p className={styles.inertNote}>
                      <AlertTriangle size={12} /> {cls.notLoadableReason}
                    </p>
                  )}
                </div>

                <div className={styles.modelActions}>
                  {isLoaded && (
                    <span className={styles.serving} title="This model is answering gateway requests">
                      <span className={styles.servingDot} aria-hidden /> Serving
                    </span>
                  )}

                  {/* Load and Unload occupy the same slot: they are one control
                    * in two states, and showing both at once invites clicking
                    * the wrong one. */}
                  {loadable &&
                    (isLoaded ? (
                      <Button variant="ghost" size="sm" onClick={handleUnload}>
                        <Power size={13} /> Unload
                      </Button>
                    ) : (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleLoad(m)}
                        disabled={busy !== null}
                        title="Load this model so the gateway can serve it"
                      >
                        <Play size={13} /> {busy === m.modelId ? 'Loading…' : 'Load'}
                      </Button>
                    ))}

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteModel(m)}
                    disabled={busy === m.modelId}
                    title="Delete this model and its adapters"
                    aria-label={`Delete ${m.modelName}`}
                  >
                    <Trash2 size={13} />
                  </Button>
                </div>
              </div>

              {/* Adapters belong to a model, so they are only offered on one —
                * a helper file has nothing to patch. */}
              {loadable && (
                <button className={styles.adapterToggle} onClick={() => toggleAdapters(m)}>
                  {isOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                  <Layers size={13} /> LoRA adapters
                  {mine !== undefined && <span className={styles.count}>{mine.length}</span>}
                </button>
              )}

              {isOpen && (
                <div className={styles.adapterList}>
                  {mine === undefined && <span className={styles.muted}>Checking…</span>}

                  {mine?.length === 0 && (
                    <span className={styles.muted}>
                      None installed. Find adapters for this model in Discover.
                    </span>
                  )}

                  {mine?.map((a) => (
                    <div key={a.id} className={styles.adapterRow}>
                      <span className={styles.adapterName}>{a.name}</span>

                      {/*
                        An adapter only ever runs through the capability it is
                        assigned to. Where that assignment came from is spelled
                        out beneath, because most of them are inferred from the
                        adapter's name and a guess shown as a fact is worse than
                        no guess at all.
                      */}
                      <label className={styles.adapterUse}>
                        <span className={styles.srOnly}>Use {a.name} for</span>
                        <select
                          className={styles.capabilitySelect}
                          value={a.capability ?? ''}
                          disabled={busy === a.id}
                          onChange={(e) => handleCapabilityChange(m, a, e.target.value)}
                        >
                          <option value="">Not used</option>
                          {CAPABILITIES.map((c) => (
                            <option key={c} value={c}>
                              {CAPABILITY_LABELS[c]}
                            </option>
                          ))}
                        </select>
                        {a.capability && a.assignmentConfidence && (
                          <span className={styles.muted}>
                            {ASSIGNMENT_SOURCE[a.assignmentConfidence]}
                          </span>
                        )}
                      </label>

                      <span className={styles.muted}>{formatSize(a.sizeBytes)}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRemoveAdapter(m, a)}
                        disabled={busy === a.id}
                        title="Delete this adapter"
                      >
                        <Trash2 size={12} />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </article>
          );
          })}
        </section>
      ))}
    </div>
  );
};
