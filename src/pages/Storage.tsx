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
  setCapabilityDefault,
  findAdaptersForInstalledModel,
  downloadAdapter,
  ASSIGNMENT_SOURCE,
  CAPABILITIES,
  CAPABILITY_LABELS,
  type Capability,
  type InstalledAdapter,
} from '../services/adapters.service';
import {
  formatSize,
  useCaseText,
  useCaseTooltip,
  type AdapterPage,
} from '../services/catalog.service';
import {
  GROUP_DESCRIPTIONS,
  GROUP_LABELS,
  MODEL_GROUPS,
  isLoadableGroup,
  type Classification,
  type ModelGroup,
} from '../types/ai';
import styles from './Storage.module.css';

/**
 * Groups adapters by the domain their own metadata claims.
 *
 * The key is whatever the adapter says it is — the author's tags first, its name
 * only as a fallback — so this agrees with the slot it lands in once installed.
 * Anything that claims nothing is collected under `Unsorted` rather than being
 * filed under a guess, which would read as a fact.
 */
function byDomain<T extends { capability?: string }>(items: T[]): [string, T[]][] {
  const buckets = new Map<string, T[]>();
  for (const item of items) {
    const key = item.capability ?? '';
    const list = buckets.get(key);
    if (list) list.push(item);
    else buckets.set(key, [item]);
  }
  const named = CAPABILITIES.filter((c) => buckets.has(c)) as string[];
  const extra = [...buckets.keys()].filter((k) => k !== '' && !named.includes(k));
  const order = [...named, ...extra, ...(buckets.has('') ? [''] : [])];
  return order.map((k) => [
    k === '' ? 'Unsorted' : (CAPABILITY_LABELS[k as Capability] ?? k),
    buckets.get(k) as T[],
  ]);
}

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
  /** Compatible adapters found for a model, keyed by model id. */
  const [available, setAvailable] = useState<Record<string, AdapterPage>>({});
  /** Which half of the adapter section a model is showing. */
  const [tab, setTab] = useState<Record<string, 'installed' | 'available'>>({});
  /** Adapters ticked for a batch install, keyed by model id. */
  const [picked, setPicked] = useState<Record<string, Set<string>>>({});
  /**
   * Where each adapter has got to, keyed by its repository id.
   *
   * A single `busy` flag could only say *something* was happening. Installing
   * several adapters at once needs each row to report itself, and a failure has
   * to stay on screen — a toast that has already faded cannot explain why an
   * adapter is missing from Installed.
   */
  const [status, setStatus] = useState<
    Record<string, { state: 'queued' | 'working' | 'done' | 'failed'; error?: string }>
  >({});
  /** Models whose unsupported-adapter list is expanded. */
  const [showBlocked, setShowBlocked] = useState<Set<string>>(new Set());
  /** Model whose adapter search is in flight. */
  const [finding, setFinding] = useState<string | null>(null);

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

  /**
   * Finds adapters for a model the user already has.
   *
   * The point of doing this here rather than sending them to Discover is that
   * Sarathi already knows which model this is. Making the user search a library
   * of thousands for the model sitting in front of them risks them picking a
   * near-identical repository, and adapter compatibility depends on getting
   * that exactly right.
   */
  const handleFindAdapters = async (m: any, refresh = false) => {
    setFinding(m.modelId);
    try {
      const page = await findAdaptersForInstalledModel(m.providerId, m.modelId, refresh);
      setAvailable((prev) => ({ ...prev, [m.modelId]: page }));
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setFinding(null);
    }
  };

  /**
   * Shows a half of the adapter section, fetching what it needs on first view.
   *
   * Available is searched lazily rather than with the installed list: opening a
   * model to check what it already has should not cost a network request for a
   * question the user has not asked.
   */
  const showTab = (m: any, which: 'installed' | 'available') => {
    setTab((prev) => ({ ...prev, [m.modelId]: which }));
    if (which === 'available' && !available[m.modelId]) void handleFindAdapters(m);
  };

  /**
   * Installs one adapter.
   *
   * Per-adapter rather than a batch: each is its own decision, each can fail on
   * its own — a conversion that refuses says so about that adapter — and a row
   * that reports its own progress needs no separate confirmation step.
   */
  const handleGetAdapter = async (m: any, repoId: string) => {
    await runInstalls(m, [repoId]);
  };

  /**
   * Installs adapters one at a time, reporting each as it goes.
   *
   * Sequential and individually guarded: one adapter failing to convert must not
   * abandon the others, and its reason stays on its own row rather than in a
   * toast that outlives the explanation. The installed list is refetched once at
   * the end, so an adapter appears under Installed without a manual refresh.
   */
  const runInstalls = async (m: any, repoIds: string[]) => {
    if (repoIds.length === 0) return;

    setStatus((prev) => {
      const next = { ...prev };
      for (const id of repoIds) next[id] = { state: 'queued' as const };
      return next;
    });

    let installed = 0;
    for (const repoId of repoIds) {
      setStatus((prev) => ({ ...prev, [repoId]: { state: 'working' as const } }));
      try {
        await downloadAdapter(m.providerId, m.modelId, repoId);
        setStatus((prev) => ({ ...prev, [repoId]: { state: 'done' as const } }));
        installed += 1;
      } catch (err) {
        // Kept on the row. The backend explains refusals in plain language, and
        // that explanation is the whole value of the failure.
        setStatus((prev) => ({
          ...prev,
          [repoId]: { state: 'failed' as const, error: String(err) },
        }));
      }
    }

    try {
      const found = await listInstalledAdapters(m.providerId, m.modelId);
      setAdapters((prev) => ({ ...prev, [m.modelId]: found.adapters }));
    } catch {
      // The installs already reported themselves; a failed refresh only means
      // the list is stale, which reopening fixes.
    }

    if (installed > 0) {
      addToast('success', `${installed} adapter${installed === 1 ? '' : 's'} installed`);
    }
  };

  const togglePicked = (modelId: string, repoId: string) => {
    setPicked((prev) => {
      const next = new Set(prev[modelId] ?? []);
      if (next.has(repoId)) next.delete(repoId);
      else next.add(repoId);
      return { ...prev, [modelId]: next };
    });
  };

  const clearPicked = (modelId: string) =>
    setPicked((prev) => ({ ...prev, [modelId]: new Set() }));

  /**
   * Installs every ticked adapter, one after another.
   *
   * Sequential and individually guarded: one adapter failing to convert must not
   * abandon the others the user asked for, and each failure names itself rather
   * than collapsing into "some did not install".
   */
  const handleInstallPicked = async (m: any) => {
    const chosen = Array.from(picked[m.modelId] ?? []);
    clearPicked(m.modelId);
    await runInstalls(m, chosen);
  };

  /**
   * Chooses which of several same-capability adapters the model actually uses.
   *
   * Installing a second coding adapter no longer displaces the first, so with
   * more than one there has to be a way to say which is bound.
   */
  const handleMakeDefault = async (m: any, a: InstalledAdapter) => {
    if (!a.capability) return;
    setBusy(a.id);
    try {
      await setCapabilityDefault(m.providerId, m.modelId, a.capability, a.id);
      const found = await listInstalledAdapters(m.providerId, m.modelId);
      setAdapters((prev) => ({ ...prev, [m.modelId]: found.adapters }));
      addToast('success', `${a.name} will be used for ${CAPABILITY_LABELS[a.capability]}`);
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

              {isOpen &&
                (() => {
                  const which = tab[m.modelId] ?? 'installed';
                  const page = available[m.modelId];
                  const searching = finding === m.modelId;

                  return (
                    <div className={styles.adapterList}>
                      {/*
                        Two questions, asked separately: what this model has, and
                        what it could have. Both are answered here — nothing sends
                        the user back to Discover to find their own model again.
                      */}
                      <div className={styles.adapterTabs} role="tablist">
                        <button
                          role="tab"
                          aria-selected={which === 'installed'}
                          className={styles.adapterTab}
                          onClick={() => showTab(m, 'installed')}
                        >
                          Installed
                          {mine !== undefined && <span className={styles.count}>{mine.length}</span>}
                        </button>
                        <button
                          role="tab"
                          aria-selected={which === 'available'}
                          className={styles.adapterTab}
                          onClick={() => showTab(m, 'available')}
                        >
                          Available
                          {page && <span className={styles.count}>{page.adapters.length}</span>}
                        </button>

                        {which === 'available' && page && (
                          <button
                            className={`${styles.adapterTab} ${styles.adapterSpacer}`}
                            onClick={() => handleFindAdapters(m, true)}
                            disabled={searching}
                            title="Ask HuggingFace again rather than using the saved result"
                          >
                            <RefreshCw size={11} /> {searching ? 'Searching…' : 'Find more'}
                          </button>
                        )}
                      </div>

                      {which === 'installed' && (
                        <>
                          {mine === undefined && <span className={styles.muted}>Checking…</span>}

                          {mine?.length === 0 && (
                            <span className={styles.muted}>
                              None yet — see Available to add a skill to this model.
                            </span>
                          )}

                          {byDomain(mine ?? []).map(([domain, list]) => (
                            <React.Fragment key={domain}>
                              <span className={styles.adapterDomain}>{domain}</span>
                              {list.map((a) => (
                            <div key={a.id} className={styles.adapterRow}>
                              <span className={styles.adapterName}>{a.name}</span>

                              {/*
                                An adapter only ever runs through the capability it
                                is assigned to. Where that assignment came from is
                                spelled out beside it, because most are inferred
                                from the adapter's name, and a guess shown as a
                                fact is worse than no guess at all.
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

                              {/*
                                Several adapters can share a capability, but only
                                one is bound. Saying which — and offering to switch
                                — is the difference between a list of four coding
                                adapters and knowing which one the model runs.
                              */}
                              {a.capability &&
                                (a.isDefault ? (
                                  <span
                                    className={styles.muted}
                                    title="This is the adapter this capability uses"
                                  >
                                    in use
                                  </span>
                                ) : (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleMakeDefault(m, a)}
                                    disabled={busy === a.id}
                                    title={`Use ${a.name} for ${CAPABILITY_LABELS[a.capability]} instead`}
                                  >
                                    Use this
                                  </Button>
                                ))}

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
                            </React.Fragment>
                          ))}
                        </>
                      )}

                      {which === 'available' && (
                        <>
                          {searching && !page && <span className={styles.muted}>Searching…</span>}

                          {/*
                            A notice explains an empty result — "none published"
                            and "I could not tell which model this was built from"
                            are different facts and must not both read as blank.
                          */}
                          {page?.notice && <span className={styles.muted}>{page.notice}</span>}

                          {byDomain(
                            (page?.adapters ?? []).filter((a) => a.installable)
                          ).map(([domain, list]) => (
                            <React.Fragment key={domain}>
                              <span className={styles.adapterDomain}>{domain}</span>
                              {list.map((cand) => {
                                const installed = (mine ?? []).some(
                                  (x) => x.repoId === cand.repoId
                                );
                                const st = status[cand.repoId];
                                const working =
                                  st?.state === 'queued' || st?.state === 'working';
                                return (
                                  <React.Fragment key={cand.repoId}>
                                    <div className={styles.adapterRow}>
                                      <input
                                        type="checkbox"
                                        checked={picked[m.modelId]?.has(cand.repoId) ?? false}
                                        disabled={installed || working}
                                        onChange={() => togglePicked(m.modelId, cand.repoId)}
                                        aria-label={`Select ${cand.name}`}
                                      />
                                      <span className={styles.adapterName}>{cand.name}</span>
                                      {cand.useCase && (
                                        <span
                                          className={
                                            cand.useCase.source === 'none'
                                              ? styles.useCaseUnknown
                                              : styles.useCase
                                          }
                                          title={useCaseTooltip(cand.useCase)}
                                        >
                                          {useCaseText(cand.useCase)}
                                        </span>
                                      )}
                                      <span className={styles.muted}>
                                        {cand.ggufReady ? 'ready' : 'converts on install'}
                                      </span>
                                      <span className={styles.muted}>
                                        {cand.downloads.toLocaleString()} downloads
                                      </span>
                                      <span className={`${styles.muted} ${styles.adapterSpacer}`}>
                                        {cand.author}
                                      </span>

                                      {/*
                                        Each row says where it has got to. There is
                                        no percentage because the backend fetches
                                        the file in one call and reports no bytes as
                                        it goes — a percentage here would be invented.
                                      */}
                                      {installed || st?.state === 'done' ? (
                                        <span className={styles.muted}>installed</span>
                                      ) : working ? (
                                        <span className={styles.muted}>
                                          <Spinner size="sm" />{' '}
                                          {st?.state === 'queued'
                                            ? 'queued'
                                            : cand.ggufReady
                                              ? 'downloading…'
                                              : 'downloading & converting…'}
                                        </span>
                                      ) : (
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          onClick={() => handleGetAdapter(m, cand.repoId)}
                                          title={`Install ${cand.name} for this model`}
                                        >
                                          <Download size={12} />{' '}
                                          {st?.state === 'failed' ? 'Retry' : 'Get'}
                                        </Button>
                                      )}
                                    </div>

                                    {/* Stays until the row is retried — a toast
                                      * cannot explain an absence later. */}
                                    {st?.state === 'failed' && (
                                      <div className={styles.adapterRow}>
                                        <span className={styles.adapterError}>{st.error}</span>
                                      </div>
                                    )}
                                  </React.Fragment>
                                );
                              })}
                            </React.Fragment>
                          ))}

                          {/*
                            The cost of the whole selection, before committing to
                            it. Adapters are a couple of hundred megabytes each,
                            so the number that matters is the total rather than
                            any single row.
                          */}
                          {(picked[m.modelId]?.size ?? 0) > 0 && (
                            <div className={styles.adapterRow}>
                              <span className={styles.muted}>
                                {picked[m.modelId].size} selected
                                {(() => {
                                  const total = (page?.adapters ?? [])
                                    .filter(
                                      (c) => c.installable && picked[m.modelId].has(c.repoId)
                                    )
                                    .reduce((sum, c) => sum + c.sizeBytes, 0);
                                  return total > 0 ? ` · ${formatSize(total)}` : '';
                                })()}
                              </span>
                              <Button
                                variant="primary"
                                size="sm"
                                onClick={() => handleInstallPicked(m)}
                                disabled={busy === m.modelId}
                              >
                                {busy === m.modelId ? 'Installing…' : 'Install selected'}
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => clearPicked(m.modelId)}
                                disabled={busy === m.modelId}
                              >
                                Clear
                              </Button>
                            </div>
                          )}

                          {/*
                            Kept, but out of the way and inert. Hiding them entirely
                            would leave a user who came looking for a specific
                            adapter with no explanation of where it went; listing
                            them beside installable ones implied they were a choice.
                          */}
                          {(() => {
                            const blocked = (page?.adapters ?? []).filter((a) => !a.installable);
                            if (blocked.length === 0) return null;
                            const open = showBlocked.has(m.modelId);
                            return (
                              <>
                                <button
                                  className={styles.adapterToggle}
                                  onClick={() =>
                                    setShowBlocked((prev) => {
                                      const next = new Set(prev);
                                      if (next.has(m.modelId)) next.delete(m.modelId);
                                      else next.add(m.modelId);
                                      return next;
                                    })
                                  }
                                >
                                  {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                                  Not supported for this model
                                  <span className={styles.count}>{blocked.length}</span>
                                </button>
                                {open &&
                                  blocked.map((cand) => (
                                    <div key={cand.repoId} className={styles.adapterRow}>
                                      <span className={styles.adapterName}>{cand.name}</span>
                                      <span
                                        className={`${styles.muted} ${styles.adapterSpacer}`}
                                        title={cand.blockedReason ?? undefined}
                                      >
                                        {cand.blockedReason}
                                      </span>
                                    </div>
                                  ))}
                              </>
                            );
                          })()}

                          {page?.ageHours != null && (
                            <span className={styles.muted}>checked {page.ageHours}h ago</span>
                          )}
                        </>
                      )}
                    </div>
                  );
                })()}

            </article>
          );
          })}
        </section>
      ))}
    </div>
  );
};
