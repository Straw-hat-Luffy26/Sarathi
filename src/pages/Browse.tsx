import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Search,
  AlertTriangle,
  RefreshCw,
  Download,
  Heart,
  Clock,
  Layers,
  ChevronDown,
  ChevronRight,
  X,
  Sparkles,
  ArrowDownWideNarrow,
  Cpu,
  Database,
} from 'lucide-react';
import { Button } from '../components/ui';
import {
  browseModelCards,
  findModelAdapters,
  formatSize,
  onCatalogProgress,
  onCatalogUpdated,
  refreshModelLibrary,
  type AdapterPage,
  type CatalogPage,
  type CatalogProgress,
} from '../services/catalog.service';
import {
  downloadAdapter,
  getAdapterDetails,
  type AdapterDetails,
} from '../services/adapters.service';
import { startModelDownload } from '../services/download.service';
import { useDownloads } from '../hooks/useDownloads';
import { useToast } from '../hooks/useToast';
import {
  CATEGORY_LABELS,
  KIND_LABELS,
  MODEL_CATEGORIES,
  runsHere,
  type ModelCard,
  type ModelCategory,
  type ModelKind,
  type Placement,
} from '../types/ai';
import styles from './Browse.module.css';

/** A downloadable size, as the card reports it. */
type Quant = ModelCard['quantizations'][number];

/**
 * The one filter in effect. Exactly one, never a combination.
 *
 * The sidebar reads as tabs, and tabs are alternatives rather than a stack of
 * constraints. Letting a category and a type both apply meant "Coding" showed
 * *coding models of the type already selected* — one card, under a heading that
 * had said 26 a moment earlier — and which models a tab held depended on the
 * order the user had clicked things. Each tab now answers the same question of
 * the whole catalog, so it shows the same models however it was reached.
 */
type Selection =
  | { tab: 'all' }
  | { tab: 'recommended' }
  | { tab: 'category'; value: ModelCategory }
  | { tab: 'type'; value: ModelKind };

/** How the grid is ordered. */
type SortKey = 'best' | 'small' | 'latest';

const SORT_LABELS: Record<SortKey, string> = {
  best: 'Best first',
  small: 'Small and fast first',
  latest: 'Latest models first',
};

/**
 * How good a bet this model is, as a single number.
 *
 * Ordering by downloads alone puts whatever went viral on top, including
 * 1-bit experiments and models too large to run here. This ranks on what makes
 * a model useful *to this user*: Sarathi vouches for it, it runs on their
 * hardware, it will not spray reasoning tokens into their editor — and only
 * then how many people have used it, as a tie-break between otherwise equal
 * candidates.
 *
 * Popularity is compressed with a log so a model with ten times the downloads
 * counts as somewhat better rather than ten times better; raw counts span six
 * orders of magnitude and would otherwise swamp every other term.
 */
function quality(card: ModelCard): number {
  const fitsVram = card.quantizations.some((q) => q.fits);
  // A model that only runs by holding its experts in system memory does run —
  // it is just slower than one that sits on the card. Ranking it above models
  // that cannot run at all, and below those that fit, is what puts an
  // offloadable MoE somewhere a user would actually find it.
  const offloads = !fitsVram && card.quantizations.some((q) => q.offload);
  return (
    (card.recommended ? 1000 : 0) +
    (fitsVram ? 500 : 0) +
    (offloads ? 250 : 0) +
    (card.emitsReasoning ? -250 : 0) +
    // A fine-tune or merge is somebody's variation on a model, not the model.
    // Searching for a family surfaced things like "heretic abliterated" and
    // "coder fable5 composer2.5" above the release they were built from,
    // because a remix that went briefly viral out-downloads the plain build.
    // The penalty is smaller than the fits-in-VRAM term on purpose: a
    // fine-tune that runs on this machine is still a better answer than a base
    // model that does not.
    (card.isFinetune ? -200 : 0) +
    Math.log10(card.downloads + 1) * 20 +
    Math.log10(card.likes + 1) * 5
  );
}

/**
 * A name with its separators removed, for comparing what people typed against
 * what publishers named things.
 *
 * The same model is written `gemma 4`, `gemma-4`, and `gemma_4` depending on
 * who uploaded it, so none of those may be the difference between a match and
 * a miss. Dots are deliberately *kept*: a dot is part of a version number, and
 * dropping it makes `Qwen3-8B` indistinguishable from `Qwen 3.8`.
 */
function squash(s: string): string {
  return s.toLowerCase().replace(/[\s_\-/]+/g, '');
}

/**
 * How well a card answers the term that was actually searched for.
 *
 * HuggingFace's `search=` is a loose substring match ordered by download count,
 * so "gemma 4" comes back led by `gemma-3-4b-it` — it contains "gemma", it
 * contains a "4", and it has been downloaded 187,000 times. Taking the Hub's
 * order as the answer therefore responds to a question about one model with
 * the most popular models that merely resemble its name.
 */
function relevance(card: ModelCard, query: string, brand?: string | null): number {
  const q = squash(query);
  if (!q) return 0;

  // Whose model this is — the one piece of evidence the text cannot supply.
  //
  // Two separate questions, and both matter, because for most labs they have
  // different answers. NVIDIA published exactly one GGUF chat model; the other
  // ninety NVIDIA models a person can run were converted by `bartowski`,
  // `unsloth`, and `lmstudio-community`. So promoting only what NVIDIA
  // *uploaded* moves one card and leaves the rest of an "nvidia" search sorted
  // by download count among unrelated repositories — which is exactly how it
  // looked before this existed.
  //
  //   6  NVIDIA uploaded it        — the official release
  //   3  it is an NVIDIA model     — somebody else's conversion of one
  //   0  it merely says "NVIDIA"   — e.g. a Qwen model quantized to NVFP4
  //
  // Additive with the text score below rather than replacing it, so the tiers
  // stay ordered internally: on "qwen coder", Qwen's coder models come before
  // Qwen's other models, and both come before everyone else's coder builds.
  const official = brand != null && card.publisherBrand === brand;
  const sameBrand = brand != null && !official && card.sourceBrand === brand;
  const provenance = official ? 6 : sameBrand ? 3 : 0;

  // The whole term as one run: `gemma4` inside `gemma-4-12b-it`. This is what
  // separates the model asked for from the model with a similar-looking name —
  // `gemma-3-4b-it` squashes to `gemma34bit`, which does not contain `gemma4`.
  if (squash(card.name).includes(q) || squash(card.repoId).includes(q)) return provenance + 2;

  // Every word present, but not adjacently. Weaker evidence, and ranked below
  // an exact run, but still a real match worth showing above unrelated models.
  const words = query.split(/\s+/).map(squash).filter(Boolean);
  if (words.length > 1 && words.every((w) => squash(card.repoId).includes(w))) return provenance + 1;

  return provenance;
}

/** The size actually downloaded if the user presses the button on this card. */
function offerSize(card: ModelCard): number {
  return pickOffer(card).offer?.sizeBytes ?? Number.MAX_SAFE_INTEGER;
}

function sortCards(
  cards: ModelCard[],
  key: SortKey,
  query = '',
  brand?: string | null
): ModelCard[] {
  const sorted = [...cards];

  // While a search is active, relevance outranks every ordering below it. The
  // user named a model; a listing led by a different model has answered a
  // question they did not ask, whichever sort happens to be selected. Within
  // one relevance tier the chosen sort decides, and `quality` puts what runs on
  // this machine above what does not — so the compatible builds of the model
  // asked for come first, which is the whole point of searching for it.
  const byRelevance = query
    ? (a: ModelCard, b: ModelCard) => relevance(b, query, brand) - relevance(a, query, brand)
    : () => 0;

  switch (key) {
    case 'best':
      return sorted.sort((a, b) => byRelevance(a, b) || quality(b) - quality(a));
    case 'small':
      // Smaller downloads are the ones that load fastest and generate fastest,
      // so size is the ordering. Every model with no runnable offer lands on
      // the same sentinel size, and quality breaks the tie so that block at the
      // bottom is still ordered usefully rather than arbitrarily.
      return sorted.sort(
        (a, b) =>
          byRelevance(a, b) || offerSize(a) - offerSize(b) || quality(b) - quality(a)
      );
    case 'latest':
      // `lastModified` is ISO 8601, which sorts correctly as text.
      return sorted.sort(
        (a, b) =>
          byRelevance(a, b) ||
          (b.lastModified ?? '').localeCompare(a.lastModified ?? '') ||
          quality(b) - quality(a)
      );
  }
}

/** Whether a card belongs in the given tab. */
function inSelection(card: ModelCard, selection: Selection): boolean {
  switch (selection.tab) {
    case 'all':
      return true;
    case 'recommended':
      return card.recommended;
    case 'category':
      return card.categories.includes(selection.value);
    case 'type':
      return card.kind === selection.value;
  }
}

/**
 * Model browser: category sidebar, search, and cards.
 *
 * Filtering happens on already-fetched cards, but *searching* goes back to
 * HuggingFace — the loaded page is only the popular sweep, so filtering it for
 * a specific fine-tune would wrongly report that nothing matches.
 *
 * Details open in a drawer rather than inline. Cards sit in a grid, and grid
 * rows share a height, so expanding one card stretched its neighbours into tall
 * empty boxes; the drawer also gives the size and adapter tables room to be
 * read, which a third of a card's width did not.
 */
export const Browse: React.FC = () => {
  const [page, setPage] = useState<CatalogPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>({ tab: 'all' });
  const [sort, setSort] = useState<SortKey>('best');
  const [searchTerm, setSearchTerm] = useState('');
  /** The term the current results reflect, so the header can say so. */
  const [activeSearch, setActiveSearch] = useState('');
  /** The card whose details are open in the drawer. */
  const [detailCard, setDetailCard] = useState<ModelCard | null>(null);
  /** Latest sweep progress, so a multi-minute fetch is visibly working. */
  const [progress, setProgress] = useState<CatalogProgress | null>(null);

  /**
   * Fetches a page of results.
   *
   * `quiet` swaps the results in without showing the loading state. It is what
   * a background refresh uses: the user is reading a working listing, and
   * replacing it with a spinner to deliver an update they did not ask for would
   * undo the whole point of serving the saved library first.
   */
  const load = useCallback(
    async (query?: string, opts?: { force?: boolean; quiet?: boolean }) => {
      const quiet = opts?.quiet ?? false;
      if (!quiet) {
        setLoading(true);
        setProgress(null);
      }
      setError(null);
      try {
        setPage(opts?.force ? await refreshModelLibrary() : await browseModelCards(query));
        setActiveSearch(opts?.force ? '' : query ?? '');
      } catch (err) {
        // These messages are written for people — rate limiting says to add a
        // token — so they are shown as-is rather than replaced.
        //
        // A quiet reload is the exception: it was not requested, so a failure
        // is not news. The listing already on screen still works.
        if (!quiet) setError(String(err));
      } finally {
        if (!quiet) {
          setLoading(false);
          setProgress(null);
        }
      }
    },
    []
  );

  // Progress arrives from the sweep itself, which is the only thing that knows
  // how many repositories there are to read. Subscribed for the life of the
  // page rather than per request, so a background refresh started by an earlier
  // visit still reports into this one.
  //
  // The subscription is *awaited* before the load starts. Registering it in an
  // earlier `useEffect` than the one calling `load()` was not enough: `listen`
  // returns a promise, and the sweep began while that promise was still
  // pending, so the earliest events — the whole searching phase — were emitted
  // before anything was listening. That is what left the screen showing a bare
  // spinner and no numbers during the part of the load that takes longest.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;

    void (async () => {
      unlisten = await onCatalogProgress(setProgress);
      if (!cancelled) void load();
    })();

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [load]);

  // A background refresh replaces the stored library. Re-reading it here is
  // what turns "we will check for updates" into the updates actually appearing,
  // rather than the user having to know to reload.
  useEffect(() => {
    const pending = onCatalogUpdated(() => {
      // Only for the default listing: pulling the swept library out from under
      // someone reading search results would discard what they asked for.
      if (!activeSearch) void load(undefined, { quiet: true });
    });
    return () => {
      void pending.then((off) => off());
    };
  }, [activeSearch, load]);

  /**
   * The full catalog for this search, and the single source every tab and
   * every count is derived from.
   *
   * Nothing narrows this. Selecting a tab changes which cards are *shown*, not
   * what the next tab is computed from — otherwise opening one tab silently
   * shrinks the others, and a tab's contents depend on the route taken to it.
   */
  const all = useMemo(() => page?.cards ?? [], [page]);

  /**
   * The publisher this search turned out to be about, or null.
   *
   * Read off the response rather than worked out here: the backend owns the
   * alias table — it is what knows "kimi" means `moonshotai` — and a second
   * copy on this side would drift from it the first time a brand is added.
   */
  const matchedBrand = page?.matchedBrand ?? null;

  const visible = useMemo(
    () => sortCards(all.filter((c) => inSelection(c, selection)), sort, activeSearch, matchedBrand),
    [all, selection, sort, activeSearch, matchedBrand]
  );

  /**
   * Counts, all taken from the whole catalog.
   *
   * Because exactly one tab applies at a time, each of these is by construction
   * the length of what clicking it shows — the count and the grid cannot drift
   * apart.
   */
  const categoryCounts = useMemo(() => {
    const counts = new Map<ModelCategory, number>();
    for (const card of all) {
      for (const c of card.categories) counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    return counts;
  }, [all]);

  const kindCounts = useMemo(() => {
    const counts = new Map<ModelKind, number>();
    for (const card of all) counts.set(card.kind, (counts.get(card.kind) ?? 0) + 1);
    return counts;
  }, [all]);

  const recommendedCount = useMemo(() => all.filter((c) => c.recommended).length, [all]);

  /**
   * Progress from a refresh running behind the results.
   *
   * Filtered on the flag rather than on `loading`, because the two are not the
   * same thing: a foreground sweep started from another window reports here
   * too, and putting its counts in the banner beside a listing that is already
   * drawn would describe the wrong work.
   */
  const backgroundProgress = progress?.background ? progress : null;
  /** The refresh's completed share, or null while it has no denominator yet. */
  const backgroundPct =
    backgroundProgress?.fraction != null
      ? Math.round(backgroundProgress.fraction * 100)
      : null;

  /**
   * Recommended cards shown as their own section, ahead of everything else.
   *
   * Only on the "All" tab: on a narrower tab, splitting the grid in two
   * obscures how many results that tab actually holds.
   */
  const showRecommendedSection =
    selection.tab === 'all' && !activeSearch && recommendedCount > 0;

  const [featured, remainder] = useMemo(() => {
    if (!showRecommendedSection) return [[], visible] as const;
    return [
      visible.filter((c) => c.recommended),
      visible.filter((c) => !c.recommended),
    ] as const;
  }, [showRecommendedSection, visible]);

  const submitSearch = (e: React.FormEvent) => {
    e.preventDefault();
    load(searchTerm.trim() || undefined);
    setSelection({ tab: 'all' });
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Models</h1>
          {/* Says what the listing *is*, which is not "HuggingFace" but "the
            * part of HuggingFace that runs here". Naming the models left out
            * keeps a short list from reading as a failed fetch. */}
          <p className={styles.subtitle}>
            {/* When the search named a publisher, say so. It reports that the
              * word was understood as a company and that their own shelf was
              * fetched — otherwise a listing led by first-party releases looks
              * like the same download-ordered results as always. */}
            {activeSearch && matchedBrand
              ? `Results for “${activeSearch}” — ${matchedBrand} releases first`
              : activeSearch
                ? `Results for “${activeSearch}”`
                : 'Models that run on this computer, from HuggingFace.'}
            {!activeSearch && (page?.hiddenIncompatible ?? 0) > 0 && (
              <span className={styles.filterNote}>
                {' · '}
                {page?.hiddenIncompatible.toLocaleString()} hidden — too large for this
                hardware
              </span>
            )}
          </p>
        </div>

        <form className={styles.search} onSubmit={submitSearch}>
          <Search size={15} className={styles.searchIcon} />
          <input
            className={styles.searchInput}
            /* Names a company as well as a model, because nothing else on the
             * screen says that typing one works. The examples are the whole
             * hint — "search by publisher" describes the feature without
             * telling anyone what to type. */
            placeholder="Search HuggingFace — a model, or a company like Qwen or NVIDIA…"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            aria-label="Search models"
          />
          <Button type="submit" size="sm" disabled={loading}>
            Search
          </Button>
        </form>

        <label className={styles.sort}>
          <ArrowDownWideNarrow size={14} aria-hidden />
          <span className={styles.sortLabel}>Sort</span>
          <select
            className={styles.sortSelect}
            value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}
            aria-label="Sort models"
          >
            {(Object.keys(SORT_LABELS) as SortKey[]).map((k) => (
              <option key={k} value={k}>
                {SORT_LABELS[k]}
              </option>
            ))}
          </select>
        </label>
      </header>

      {/* Where these results came from. Shown whenever they are a saved copy,
          because "this list is from yesterday and we are checking" and "this
          list is current" are different claims and the user is owed the right
          one. */}
      {!loading && page && page.source === 'stored' && (
        <div className={styles.cacheBar} role="status">
          <Database size={14} aria-hidden />
          <span>
            Showing your saved library{ageLabel(page.ageSeconds) && ` from ${ageLabel(page.ageSeconds)}`}.
            {page.refreshing
              ? // The same counts the loading screen would show, in one line.
                // "Checking in the background" on its own is indistinguishable
                // from a check that silently died.
                ` ${backgroundProgress?.message ?? 'Updating model library'}…`
              : ' It is up to date.'}
          </span>

          {/* The refresh's own progress, kept to a thin bar inside this row.
            * The results are already on screen and readable, so this reports the
            * update without taking the page over the way the first-load panel
            * does. Rendered only while a refresh is running, so the row does not
            * reserve empty space once the library is up to date. */}
          {page.refreshing && (
            <div
              className={styles.miniTrack}
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              // Absent during the search phase for the same reason the full
              // panel omits it: the denominator is not known yet.
              aria-valuenow={backgroundPct ?? undefined}
              aria-label="Updating the model library"
            >
              <div
                className={backgroundPct === null ? styles.miniIndeterminate : styles.miniFill}
                style={backgroundPct === null ? undefined : { width: `${backgroundPct}%` }}
              />
            </div>
          )}
          <Button
            variant="ghost"
            size="sm"
            disabled={loading || page.refreshing}
            onClick={() => void load(undefined, { force: true })}
          >
            <RefreshCw size={13} /> Check now
          </Button>
        </div>
      )}

      {page?.notice && (
        <div className={styles.notice} role="status">
          <AlertTriangle size={15} />
          <span>{page.notice}</span>
        </div>
      )}

      {error && (
        <div className={styles.error} role="alert">
          <AlertTriangle size={15} />
          <span>{error}</span>
          <Button variant="ghost" size="sm" onClick={() => load(activeSearch || undefined)}>
            <RefreshCw size={13} /> Try again
          </Button>
        </div>
      )}

      <div className={styles.body}>
        <nav className={styles.sidebar} aria-label="Browse models by">
          {/* Every entry is an alternative view of the same catalog, so each one
              replaces the selection rather than adding to it, and each count is
              taken from the whole catalog. A tab therefore holds the same models
              no matter which tab preceded it. */}
          {recommendedCount > 0 && (
            <button
              className={`${styles.catBtn} ${styles.recommendBtn} ${
                selection.tab === 'recommended' ? styles.catActive : ''
              }`}
              onClick={() => setSelection({ tab: 'recommended' })}
              aria-current={selection.tab === 'recommended'}
              title="Models Sarathi vouches for: dependable publishers, proven conversions, no reasoning tokens, and they run on this machine."
            >
              <span>
                <Sparkles size={13} className={styles.recommendIcon} /> Sarathi Recommended
              </span>
              <span className={styles.count}>{recommendedCount}</span>
            </button>
          )}

          <button
            className={`${styles.catBtn} ${selection.tab === 'all' ? styles.catActive : ''}`}
            onClick={() => setSelection({ tab: 'all' })}
            aria-current={selection.tab === 'all'}
          >
            <span>All</span>
            <span className={styles.count}>{all.length}</span>
          </button>

          {MODEL_CATEGORIES.filter((c) => (categoryCounts.get(c) ?? 0) > 0).map((c) => (
            <button
              key={c}
              className={`${styles.catBtn} ${
                selection.tab === 'category' && selection.value === c ? styles.catActive : ''
              }`}
              onClick={() => setSelection({ tab: 'category', value: c })}
              aria-current={selection.tab === 'category' && selection.value === c}
            >
              <span>{CATEGORY_LABELS[c]}</span>
              <span className={styles.count}>{categoryCounts.get(c)}</span>
            </button>
          ))}

          {/* Type asks a different question from category — "what is this entry"
              rather than "what is it good at" — but it is still one of the
              alternatives, not a second constraint layered on top. */}
          {kindCounts.size > 1 && (
            <>
              <h2 className={styles.sidebarHeading}>Type</h2>
              {(Object.keys(KIND_LABELS) as ModelKind[])
                .filter((k) => (kindCounts.get(k) ?? 0) > 0)
                .map((k) => (
                  <button
                    key={k}
                    className={`${styles.catBtn} ${
                      selection.tab === 'type' && selection.value === k ? styles.catActive : ''
                    }`}
                    onClick={() => setSelection({ tab: 'type', value: k })}
                    aria-current={selection.tab === 'type' && selection.value === k}
                  >
                    <span>{KIND_LABELS[k]}</span>
                    <span className={styles.count}>{kindCounts.get(k)}</span>
                  </button>
                ))}
            </>
          )}
        </nav>

        <section
          className={loading ? `${styles.results} ${styles.resultsCentered}` : styles.results}
        >
          {loading && <LoadingLibrary progress={progress} />}

          {/* An empty listing has two very different causes now, and saying
            * "no models found" for both would be misleading. If models were
            * found and then filtered out, the honest message is that this
            * hardware cannot run them — not that the Hub had nothing. */}
          {!loading && visible.length === 0 && !error && (
            <div className={styles.centered}>
              <p>
                {selection.tab !== 'all'
                  ? 'Nothing in this category runs on this computer.'
                  : (page?.hiddenIncompatible ?? 0) > 0
                    ? `None of the ${page?.hiddenIncompatible.toLocaleString()} models found ` +
                      `will run on this computer’s memory.`
                    : 'No models found. Try a different search.'}
              </p>
            </div>
          )}

          {!loading && featured.length > 0 && (
            <>
              <div className={styles.sectionBar}>
                <h2 className={styles.sectionTitle}>
                  <Sparkles size={14} className={styles.recommendIcon} /> Sarathi Recommended
                </h2>
                <p className={styles.sectionNote}>
                  Dependable publishers, conversions that have been proven in use, no reasoning
                  tokens in the output — and they run on this machine.
                </p>
              </div>
              {featured.map((card) => (
                <Card
                  key={card.repoId}
                  card={card}
                  open={detailCard?.repoId === card.repoId}
                  onOpen={() => setDetailCard(card)}
                  matchedBrand={matchedBrand}
                />
              ))}
            </>
          )}

          {!loading && featured.length > 0 && remainder.length > 0 && (
            <div className={styles.sectionBar}>
              <h2 className={styles.sectionTitle}>Everything else</h2>
              <p className={styles.sectionNote}>
                The rest of the library, newest and most downloaded first. Worth reading the size
                and quality notes before picking one.
              </p>
            </div>
          )}

          {!loading &&
            remainder.map((card) => (
              <Card
                key={card.repoId}
                card={card}
                open={detailCard?.repoId === card.repoId}
                onOpen={() => setDetailCard(card)}
                matchedBrand={matchedBrand}
              />
            ))}
        </section>
      </div>

      {detailCard && <DetailDrawer card={detailCard} onClose={() => setDetailCard(null)} />}
    </div>
  );
};

/** Compact age of a saved library: `2 hours ago`, `3 days ago`. */
function ageLabel(seconds?: number | null): string | null {
  if (seconds === null || seconds === undefined || seconds < 60) return null;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? '' : 's'} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? '' : 's'} ago`;
}

/**
 * The loading state, which has to prove the application is working.
 *
 * A full authenticated sweep is around two thousand requests and takes minutes.
 * The previous state — a spinner over the words "Reading the model library…" —
 * said the same thing at second one and at minute three, so the honest reading
 * of it was that the app had hung.
 *
 * Three things fix that, and all three are needed. The phase says which of the
 * two very different jobs is running. The counts move, which no frozen process
 * does. And the bar fills, but only once there is a real denominator to fill it
 * against: during the search phase the number of repositories is not yet known,
 * so an indeterminate shimmer is shown rather than a percentage that would have
 * to jump backwards when the truth arrives.
 */
/** `95` -> `1:35`. Tabular digits keep it from jittering as it counts. */
function clock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

/** What each phase is actually doing, in the order they run. */
const PHASES: { key: CatalogProgress['phase']; label: string }[] = [
  { key: 'searching', label: 'Searching' },
  { key: 'fetching', label: 'Reading details' },
  { key: 'caching', label: 'Saving' },
];

function LoadingLibrary({ progress }: { progress: CatalogProgress | null }) {
  const pct = progress?.fraction != null ? Math.round(progress.fraction * 100) : null;

  // A clock that ticks regardless of what the backend is doing.
  //
  // Every other signal here depends on an event arriving, and the gaps between
  // events are exactly when the screen used to read as hung — a batch of eight
  // repositories can take many seconds, and during that time nothing moved. The
  // elapsed counter is the one thing that cannot stall while the app is alive,
  // which makes it the difference between "working" and "frozen" at a glance.
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const id = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - started) / 1000)),
      1000
    );
    return () => window.clearInterval(id);
  }, []);

  const phaseIndex = PHASES.findIndex((p) => p.key === progress?.phase);
  const current = PHASES[phaseIndex]?.label ?? 'Starting';

  return (
    <div className={styles.centered}>
      <div className={styles.loadingPanel}>
        <div className={styles.loadingHead}>
          <span className={styles.loadingPhase}>{current}</span>
          {/* Right-aligned against the phase so the two never reflow each
            * other as the numbers change width. */}
          <span className={styles.loadingClock}>{clock(elapsed)}</span>
        </div>

        <div
          className={styles.progressTrack}
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          // Omitted while indeterminate, which is what tells a screen reader the
          // difference between "no progress yet" and "0% done".
          aria-valuenow={pct ?? undefined}
          aria-label="Loading the model library"
        >
          <div
            className={pct === null ? styles.progressIndeterminate : styles.progressFill}
            style={pct === null ? undefined : { width: `${pct}%` }}
          />
        </div>

        {/* The counts, which are the proof of work: they move even when the
          * percentage cannot be computed yet. */}
        <div className={styles.loadingStats}>
          <span>{progress?.message ?? 'Contacting HuggingFace…'}</span>
          {progress != null && progress.total > 0 && (
            <span className={styles.loadingCount}>
              {progress.done.toLocaleString()} / {progress.total.toLocaleString()}
              {pct !== null && ` · ${pct}%`}
            </span>
          )}
        </div>

        {/* Three dots showing which of the three jobs is running. A single bar
          * cannot say that the search phase has no denominator; this can. */}
        <ol className={styles.phaseList}>
          {PHASES.map((p, i) => (
            <li
              key={p.key}
              className={
                i < phaseIndex
                  ? styles.phaseDone
                  : i === phaseIndex
                    ? styles.phaseActive
                    : styles.phasePending
              }
            >
              {p.label}
            </li>
          ))}
        </ol>

        <p className={styles.loadingNote}>
          The first load reads the whole library from HuggingFace and takes a few
          minutes. After this it is saved, and opening this page is instant.
        </p>
      </div>
    </div>
  );
}

/** How the offered download would run. There is no "does not run" case here. */
type Runs = Placement;

/**
 * The build this card offers, as chosen by the Rust planner.
 *
 * This used to rank the quantizations itself, with a third tier that fell back
 * to the smallest build and offered it behind a "Download anyway" warning. That
 * tier is why models this computer cannot run were reaching the browser at all:
 * the backend listed every repository, and the UI always found *something* to
 * offer. Both halves are gone. Discover now only receives models with a
 * placement, and the build is read from `bestQuantization` rather than picked
 * again here — deciding it in two places would give two answers to one
 * question, which is exactly the duplication the planner is meant to prevent.
 *
 * Returns no offer when the card carries no placement. That should not happen
 * in a normal listing; it is handled rather than asserted because a card can
 * also come from hardware Sarathi could not read.
 */
function pickOffer(card: ModelCard): { offer: Quant | null; runs: Runs | null } {
  if (!card.runsHere || !card.bestQuantization) return { offer: null, runs: null };

  const offer = card.quantizations.find((q) => q.label === card.bestQuantization) ?? null;
  return { offer, runs: offer ? card.runsHere : null };
}

/**
 * Starts a download of a build that runs here.
 *
 * Shared by the card and the drawer so both behave identically.
 *
 * It used to take a confirmation callback and warn before downloading a build
 * that could not load. Nothing reaches it in that state any more: Discover
 * lists only models with a placement, and both callers offer only the builds
 * the planner found room for. Asking the user to accept a download that would
 * "most likely fail to load" was delegating back the judgement Sarathi exists
 * to make.
 */
async function beginDownload(
  card: ModelCard,
  quant: Quant,
  addToast: (kind: 'success' | 'error', msg: string) => void
): Promise<void> {
  try {
    await startModelDownload({
      modelId: card.repoId,
      modelName: card.name,
      providerId: 'huggingface',
      quantization: quant.label,
      format: 'GGUF',
      backend: 'llama.cpp (GGUF)',
    });
    addToast('success', `Downloading ${card.name} (${quant.label}) — see progress in Storage`);
  } catch (err) {
    addToast('error', `Could not start the download: ${String(err)}`);
  }
}

interface CardProps {
  card: ModelCard;
  open: boolean;
  onOpen: () => void;
  /**
   * The publisher the current search named, when it named one. A card whose
   * own publisher matches is one of that company's releases rather than a
   * third party's conversion of it, and says so.
   */
  matchedBrand?: string | null;
}

function Card({ card, open, onOpen, matchedBrand }: CardProps) {
  // Two different claims, and saying which one applies is the whole value.
  //
  // "Official" means the lab uploaded this. The weaker tag means the lab *made*
  // the model and somebody else converted it to GGUF — true of almost every
  // runnable NVIDIA, DeepSeek, Meta, and Moonshot model, because those labs
  // publish weights rather than GGUF. Collapsing the two into one badge would
  // put NVIDIA's name on a file NVIDIA never uploaded.
  const official = matchedBrand != null && card.publisherBrand === matchedBrand;
  const sameBrand = matchedBrand != null && !official && card.sourceBrand === matchedBrand;
  const [starting, setStarting] = useState(false);
  const { addToast } = useToast();
  const { isDownloading } = useDownloads();
  const fitting = card.quantizations.filter((q) => q.fits);
  const offloadable = card.quantizations.filter((q) => !q.fits && q.offload);
  const { offer, runs } = pickOffer(card);

  const download = async () => {
    if (!offer) return;
    setStarting(true);
    try {
      await beginDownload(card, offer, addToast);
    } finally {
      setStarting(false);
    }
  };

  return (
    <article className={`${styles.card} ${open ? styles.cardOpen : ''}`}>
      <div className={styles.cardTop}>
        <span className={styles.publisher}>
          {card.publisher}
          {/* Only during a search that named this publisher. Outside that, a
            * badge on every first-party repository would be decoration — the
            * point is to answer "which of these is the real one" at the moment
            * that question is being asked. */}
          {official && (
            <span
              className={styles.officialTag}
              title={`Published by ${card.publisherBrand} themselves, not converted by a third party.`}
            >
              Official
            </span>
          )}
          {/* Quieter than "Official" on purpose: it is a claim about the model,
            * not about who uploaded this copy of it. */}
          {sameBrand && (
            <span
              className={styles.sourceTag}
              title={`A ${card.sourceBrand} model, converted to GGUF by ${card.publisher}. ${card.sourceBrand} publish weights rather than GGUF, so this is how it becomes runnable.`}
            >
              {card.sourceBrand} model
            </span>
          )}
        </span>
        <div className={styles.badges}>
          {/* What this *is* comes first: whether it can run on its own is the
            * thing a non-specialist most needs to know before downloading. */}
          <span
            className={card.kind === 'lora-adapter' ? styles.badgeStrong : styles.badgeKind}
            title={card.kindExplanation}
          >
            {KIND_LABELS[card.kind]}
          </span>
          {card.license && <span className={styles.badge}>{card.license}</span>}
        </div>
      </div>

      <h2 className={styles.cardName}>{card.name}</h2>
      <p className={styles.summary}>{card.summary}</p>

      {/* Spelled out, because "LoRA adapter" alone does not tell someone that
        * downloading it on its own will not give them a working model. */}
      <p className={styles.kindNote}>{card.kindExplanation}</p>

      {card.datasets && card.datasets.length > 0 && (
        <p className={styles.datasets}>Trained on {card.datasets.join(', ')}</p>
      )}

      <div className={styles.cats}>
        {card.categories.map((c) => (
          <span key={c} className={styles.cat}>
            {c.replace(/-/g, ' ')}
          </span>
        ))}
      </div>

      <div className={styles.stats}>
        <span title="Downloads">
          <Download size={12} /> {card.downloadsLabel}
        </span>
        <span title="Likes">
          <Heart size={12} /> {card.likes}
        </span>
        {card.ageLabel && (
          <span title="Last updated">
            <Clock size={12} /> {card.ageLabel}
          </span>
        )}
        <span title="Quantizations available">
          <Layers size={12} /> {card.quantizations.length}
        </span>
      </div>

      {card.baseModel && (
        <p className={styles.base}>
          Based on <code>{card.baseModel}</code>
        </p>
      )}

      {/* The one line that changes what this card means on this machine: a
        * model far larger than the card's VRAM, which nonetheless runs here.
        * Without it the sizes table reads "too large" all the way down and the
        * download button looks like a mistake. */}
      {offloadable.length > 0 && offloadable[0].offload && (
        <p className={styles.offloadNote}>
          <Cpu size={12} aria-hidden /> {offloadable[0].offload.note}
        </p>
      )}

      <div className={styles.cardFoot}>
        <button className={styles.expand} onClick={onOpen} aria-expanded={open}>
          {`Sizes & LoRA — ${fitting.length} of ${card.quantizations.length} fit`}
          {offloadable.length > 0 && `, ${offloadable.length} offloadable`}
        </button>

        {/* An adapter is not a runnable model, so it gets no download button —
          * it is installed from the base model's card instead. */}
        {card.kind !== 'lora-adapter' && offer && (
          <button
            className={styles.downloadBtn}
            disabled={starting || isDownloading(card.repoId)}
            onClick={() => void download()}
            // Both remaining cases describe a model that runs. The third
            // wording this replaced — "still needs more memory than this
            // computer has free" — has no card left to appear on.
            title={
              runs === 'offload'
                ? `Download ${offer.label} (${formatSize(offer.sizeBytes)}). Larger than this ` +
                  `computer's video memory, but it runs by keeping its experts in system memory.`
                : `Download the largest size that fits: ${offer.label}, ${formatSize(
                    offer.sizeBytes
                  )}`
            }
          >
            <Download size={13} />
            {isDownloading(card.repoId)
              ? 'Downloading…'
              : starting
              ? 'Starting…'
              : `Download ${offer.label}`}
          </button>
        )}
      </div>
    </article>
  );
}

interface DrawerProps {
  card: ModelCard;
  onClose: () => void;
}

/**
 * Full detail for one model, docked beside the list.
 *
 * Nothing here changes the grid's layout, so opening details cannot disturb
 * any other card.
 */
function DetailDrawer({ card, onClose }: DrawerProps) {
  const [adapters, setAdapters] = useState<AdapterPage | null>(null);
  const [loadingAdapters, setLoadingAdapters] = useState(true);
  const [installing, setInstalling] = useState<string | null>(null);
  const [installed, setInstalled] = useState<Set<string>>(new Set());
  /** Adapters ticked for a batch install. */
  const [picked, setPicked] = useState<Set<string>>(new Set());
  /** True once the user has touched the selection, so defaults stop applying. */
  const [touched, setTouched] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);
  const [starting, setStarting] = useState<string | null>(null);
  const { addToast } = useToast();
  const { isDownloading } = useDownloads();

  // Escape closes it, as with any dialog.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Looked up per model, not with the listing: one request per card would mean
  // a hundred extra calls per sweep and would hit the rate limit immediately.
  const lookupAdapters = useCallback(
    async (refresh: boolean) => {
      const baseModelId = card.baseModel || card.repoId;
      setLoadingAdapters(true);
      try {
        // Architecture is unknown for a model that is not installed, so
        // compatibility cannot be judged here and nothing is hidden on a guess.
        setAdapters(await findModelAdapters(baseModelId, null, refresh));
      } catch (err) {
        setAdapters({ baseModelId, adapters: [], readyCount: 0, notice: String(err) });
      } finally {
        setLoadingAdapters(false);
      }
    },
    [card.repoId, card.baseModel]
  );

  useEffect(() => {
    let cancelled = false;
    setAdapters(null);
    // Served from the shared cache when it is fresh, so reopening a card costs
    // nothing. "Find more" is how the user forces a new request.
    void lookupAdapters(false).then(() => {
      if (cancelled) setAdapters(null);
    });

    return () => {
      cancelled = true;
    };
  }, [lookupAdapters]);

  /**
   * A small starting selection: the adapters that install without conversion.
   *
   * Deliberately not framed as *best*. There are no per-adapter benchmarks, and
   * download counts favour whatever went viral — so the only claim made here is
   * one that can be verified, that these load as they stand. It is a starting
   * point the user edits, not a verdict, and it is capped so the default never
   * proposes a multi-gigabyte download.
   */
  const recommended = useMemo(() => {
    const ready = (adapters?.adapters ?? []).filter((a) => a.ggufReady && a.installable);
    return new Set(ready.slice(0, 3).map((a) => a.repoId));
  }, [adapters]);

  // Applied once per adapter list. After the user ticks anything the default
  // stops re-asserting itself, so unticking a suggestion makes it stay unticked.
  useEffect(() => {
    setPicked(new Set(recommended));
    setTouched(false);
  }, [recommended]);

  const togglePicked = (repoId: string) => {
    setTouched(true);
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(repoId)) next.delete(repoId);
      else next.add(repoId);
      return next;
    });
  };

  const selected = useMemo(
    () =>
      (adapters?.adapters ?? []).filter(
        (a) => a.installable && picked.has(a.repoId) && !installed.has(a.repoId)
      ),
    [adapters, picked, installed]
  );
  const selectedBytes = selected.reduce((sum, a) => sum + a.sizeBytes, 0);

  /**
   * Installs the whole selection, one adapter at a time.
   *
   * Each is guarded on its own: an adapter that refuses to convert reports that
   * about itself and the rest still install.
   */
  const installSelected = async () => {
    setInstallError(null);
    for (const a of selected) {
      setInstalling(a.repoId);
      try {
        await downloadAdapter('huggingface', card.baseModel || card.repoId, a.repoId);
        setInstalled((prev) => new Set(prev).add(a.repoId));
      } catch (err) {
        setInstallError(String(err));
      }
    }
    setInstalling(null);
  };

  const install = async (adapterRepoId: string) => {
    setInstalling(adapterRepoId);
    setInstallError(null);
    try {
      // Adapters install beside the base model, so the base model's id is what
      // identifies where they belong.
      await downloadAdapter('huggingface', card.baseModel || card.repoId, adapterRepoId);
      setInstalled((prev) => new Set(prev).add(adapterRepoId));
    } catch (err) {
      // The backend explains refusals in plain language — pass it through.
      setInstallError(String(err));
    } finally {
      setInstalling(null);
    }
  };

  const download = async (q: Quant) => {
    setStarting(q.label);
    try {
      await beginDownload(card, q, addToast);
    } finally {
      setStarting(null);
    }
  };

  return (
    <>
      <div className={styles.scrim} onClick={onClose} aria-hidden="true" />

      <aside
        className={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-label={`${card.name} details`}
      >
        <header className={styles.drawerHead}>
          <div>
            <span className={styles.publisher}>{card.publisher}</span>
            <h2 className={styles.drawerTitle}>{card.name}</h2>
          </div>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close details">
            <X size={16} />
          </button>
        </header>

        <div className={styles.drawerBody}>
          <p className={styles.summary}>{card.summary}</p>

          <section>
            <h3 className={styles.drawerSection}>Sizes</h3>
            {/* Headed, because "5.2 GB" on its own does not say whether that is
              * the download, the memory needed, or both. */}
            <table className={styles.quants}>
              <thead>
                <tr className={styles.quantHead}>
                  <th scope="col">Version</th>
                  <th scope="col">Download</th>
                  <th scope="col">Quality</th>
                  <th scope="col" className={styles.qFit}>
                    Runs here
                  </th>
                  <th scope="col" aria-label="Get" />
                </tr>
              </thead>
              <tbody>
                {card.quantizations.map((q) => {
                  const runs = runsHere(q);
                  return (
                  <tr
                    key={q.label}
                    className={
                      runs === 'vram'
                        ? styles.fits
                        : runs === 'offload'
                        ? styles.offloads
                        : styles.tooBig
                    }
                  >
                    <td className={styles.qLabel}>{q.label}</td>
                    <td>{formatSize(q.sizeBytes)}</td>
                    <td className={styles.qNote}>
                      {q.lowQuality && (
                        <AlertTriangle
                          size={11}
                          className={styles.qWarnIcon}
                          aria-hidden="true"
                        />
                      )}
                      {q.qualityNote}
                    </td>
                    {/* Three answers, not two. "Offloaded" is a real way to run
                      * and reducing it to "too large" hid every model this
                      * machine could run but not hold. The reason a MoE model
                      * cannot run is worth carrying too — short of system RAM
                      * and short of VRAM have opposite remedies. */}
                    <td
                      className={styles.qFit}
                      title={q.offload?.note ?? q.offloadBlockedReason ?? undefined}
                    >
                      {runs === 'vram' ? 'fits' : runs === 'offload' ? 'offloaded' : 'too large'}
                    </td>
                    {/* The table lists every published build so the sizes can
                      * be compared honestly, but only the ones that run here
                      * are downloadable. Offering the rest would reintroduce,
                      * one row down, the "download something that cannot load"
                      * path the card no longer has. */}
                    <td className={styles.qAction}>
                      {card.kind !== 'lora-adapter' && runs !== 'no' && (
                        <button
                          className={styles.rowBtn}
                          disabled={
                            starting !== null || isDownloading(card.repoId, q.label)
                          }
                          onClick={() => void download(q)}
                          aria-label={`Download ${card.name} ${q.label}`}
                        >
                          {isDownloading(card.repoId, q.label) ? '…' : <Download size={12} />}
                        </button>
                      )}
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>

            <p className={styles.sizeNote}>
              “Download” is the file size — roughly what the weights take in memory too.
              Running also needs spare memory for the conversation, which “Runs here”
              already allows for.
            </p>

            {/* Spelled out once, in the place someone is comparing sizes. The
              * word "offloaded" in a table cell does not explain why a 12 GB
              * download is being offered on an 8 GB card. */}
            {card.quantizations.some((q) => q.offload) && (
              <p className={styles.sizeNote}>
                <strong>Offloaded</strong> means this is a mixture-of-experts model: only a
                fraction of it runs for any one word, so Sarathi keeps the unused experts in
                system memory and the rest on the graphics card. It runs on this computer
                despite being larger than its video memory.
              </p>
            )}
          </section>

          <section>
            <h3 className={styles.drawerSection}>LoRA adapters</h3>

            {loadingAdapters && <p className={styles.adapterNotice}>Looking for adapters…</p>}
            {adapters?.notice && <p className={styles.adapterNotice}>{adapters.notice}</p>}

            {/* The saved result is reused until it expires, so this is the only
              * way to ask HuggingFace again — and the age says why one might. */}
            {adapters && !loadingAdapters && (
              <p className={styles.adapterNotice}>
                <button type="button" onClick={() => void lookupAdapters(true)}>
                  Find more
                </button>
                {adapters.ageHours != null && ` · checked ${adapters.ageHours}h ago`}
              </p>
            )}
            {installError && <p className={styles.adapterError}>{installError}</p>}

            {adapters && adapters.adapters.length > 0 && (
              <>
                {/*
                  A starting selection, not a ranking. The claim is only that
                  these install without conversion — everything else the Hub
                  reports about an adapter is popularity, which is not quality.
                */}
                {recommended.size > 0 && !touched && (
                  <p className={styles.sizeNote}>
                    Recommended for this model: {recommended.size} that install as they
                    stand. Untick any you do not want.
                  </p>
                )}

                <table className={styles.adapterTable}>
                  <thead>
                    <tr className={styles.quantHead}>
                      <th scope="col" aria-label="Select" />
                      <th scope="col">Adapter</th>
                      <th scope="col">By</th>
                      <th scope="col" className={styles.qFit}>
                        Usable
                      </th>
                      <th scope="col" aria-label="Get" />
                    </tr>
                  </thead>
                  <tbody>
                    {adapters.adapters
                      .filter((a) => a.installable)
                      .map((a) => (
                      <AdapterRow
                        key={a.repoId}
                        adapter={a}
                        installing={installing === a.repoId}
                        installed={installed.has(a.repoId)}
                        picked={picked.has(a.repoId)}
                        onPick={() => togglePicked(a.repoId)}
                        onInstall={() => void install(a.repoId)}
                      />
                    ))}
                  </tbody>
                </table>
                {selected.length > 0 && (
                  <p className={styles.sizeNote}>
                    {selected.length} selected
                    {selectedBytes > 0 && ` · ${formatSize(selectedBytes)}`}{' '}
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => void installSelected()}
                      disabled={installing !== null}
                    >
                      {installing !== null ? 'Installing…' : 'Install selected'}
                    </Button>{' '}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setTouched(true);
                        setPicked(new Set());
                      }}
                      disabled={installing !== null}
                    >
                      Clear
                    </Button>
                  </p>
                )}

                {/* Explained rather than hidden: a user who came looking for a
                  * specific adapter should learn why it is not offered, without
                  * it sitting among the ones that are. */}
                {adapters.adapters.some((a) => !a.installable) && (
                  <p className={styles.sizeNote}>
                    {adapters.adapters.filter((a) => !a.installable).length} adapter(s) are
                    published for this model but not supported here:{' '}
                    {adapters.adapters.find((a) => !a.installable)?.blockedReason}
                  </p>
                )}

                <p className={styles.sizeNote}>
                  Select an adapter's name to see what it changes about the model.
                </p>
              </>
            )}
          </section>
        </div>
      </aside>
    </>
  );
}

interface AdapterRowProps {
  adapter: {
    repoId: string;
    name: string;
    author: string;
    focus: string;
    ggufReady: boolean;
    installable: boolean;
    blockedReason?: string | null;
  };
  installing: boolean;
  installed: boolean;
  picked: boolean;
  onPick: () => void;
  onInstall: () => void;
}

/**
 * One adapter, with its name acting as a disclosure for what it changes.
 *
 * The detail is fetched on selection rather than with the list: one lookup per
 * adapter would hit HuggingFace's rate limit immediately.
 */
function AdapterRow({
  adapter: a,
  installing,
  installed,
  picked,
  onPick,
  onInstall,
}: AdapterRowProps) {
  const [open, setOpen] = useState(false);
  const [details, setDetails] = useState<AdapterDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  const toggle = async () => {
    const next = !open;
    setOpen(next);
    if (!next || details || loading) return;

    setLoading(true);
    setFailed(null);
    try {
      setDetails(await getAdapterDetails(a.repoId));
    } catch (err) {
      setFailed(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <tr className={a.ggufReady ? styles.fits : styles.tooBig}>
        <td>
          {/* Already installed means there is nothing left to select. */}
          <input
            type="checkbox"
            checked={picked && !installed && a.installable}
            disabled={installed || installing || !a.installable}
            onChange={onPick}
            aria-label={`Select ${a.name}`}
          />
        </td>
        <td>
          <button
            className={styles.adapterNameBtn}
            onClick={() => void toggle()}
            aria-expanded={open}
            title="What does this change?"
          >
            {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
            <span className={styles.adapterFocus}>{a.focus}</span>
          </button>
        </td>
        <td className={styles.adapterAuthor}>{a.author}</td>
        <td className={styles.qFit}>
          {/* An adapter the installer would refuse says so here, with the reason
            * — the alternative is a Get button whose failure only arrives after
            * the download. */}
          {!a.installable ? (
            <span className={styles.needsConversion} title={a.blockedReason ?? undefined}>
              Incompatible
            </span>
          ) : a.ggufReady ? (
            'ready'
          ) : (
            <span
              className={styles.needsConversion}
              title="PEFT safetensors. Sarathi converts it to GGUF after downloading, which needs the base model installed and a supported model family."
            >
              converts on install
            </span>
          )}
        </td>
        <td className={styles.qAction}>
          {/* A PEFT adapter is downloaded and converted in one step; if the
            * conversion cannot be done the whole install is rolled back, so the
            * button never leaves a file behind that llama.cpp would refuse.
            *
            * No button at all when the conversion is known in advance to be
            * impossible — offering one would only move the refusal to after the
            * download. */}
          {a.installable && (
          <button
            className={styles.getBtn}
            disabled={installing || installed}
            onClick={onInstall}
            title={
              a.ggufReady
                ? 'Download this adapter'
                : 'Download and convert this adapter to GGUF'
            }
          >
            {installed
              ? 'installed'
              : installing
                ? a.ggufReady
                  ? 'getting…'
                  : 'converting…'
                : 'Get'}
          </button>
          )}
        </td>
      </tr>

      {open && (
        <tr>
          <td colSpan={5} className={styles.adapterDetailCell}>
            {loading && <p className={styles.adapterNotice}>Reading its page…</p>}
            {failed && <p className={styles.adapterError}>{failed}</p>}

            {details && (
              <div className={styles.adapterDetail}>
                {details.effects.length > 0 && (
                  <ul className={styles.effects}>
                    {details.effects.map((e) => (
                      <li key={e.skill} className={styles.effect}>
                        <span className={styles.effectSkill}>{e.skill}</span>
                        {/* A guess read off the repository name must never be
                          * presented with the same weight as the author's own
                          * tags, so each line says which it is.
                          *
                          * The label says "author tagged", not "author says":
                          * the author supplied a tag, and the sentence beside it
                          * is Sarathi explaining what that tag means. Wording it
                          * as speech put our words in their mouth, which also
                          * made every adapter in a category look identical. */}
                        <span
                          className={
                            e.confidence === 'stated' ? styles.stated : styles.suggested
                          }
                          title={
                            e.confidence === 'stated'
                              ? 'The author tagged this adapter with this skill. The description is Sarathi explaining what that tag means.'
                              : 'Guessed from the repository name — the author did not tag it'
                          }
                        >
                          {e.confidence === 'stated' ? 'author tagged' : 'from its name'}
                        </span>
                        <span className={styles.effectText}>{e.description}</span>
                      </li>
                    ))}
                  </ul>
                )}

                {details.datasets.length > 0 && (
                  <p className={styles.detailLine}>
                    <strong>Trained on:</strong> {details.datasets.join(', ')}
                  </p>
                )}

                <p className={styles.detailLine}>
                  {details.sizeBytes > 0 && <>{formatSize(details.sizeBytes)} · </>}
                  {details.downloads.toLocaleString()} downloads
                  {details.license && <> · {details.license}</>}
                </p>

                {details.notice && <p className={styles.adapterNotice}>{details.notice}</p>}
                {details.blockedReason && (
                  <p className={styles.adapterNotice}>{details.blockedReason}</p>
                )}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
