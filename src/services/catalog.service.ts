// Model browsing IPC.
//
// Mirrors src-tauri/src/commands/catalog.rs.

import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type { ModelCard, ModelCategory } from '../types/ai';

export interface CategoryCount {
  category: ModelCategory;
  label: string;
  count: number;
}

/** Which of the three sources answered a request. */
export type CatalogSource =
  /** Swept from HuggingFace just now. */
  | 'live'
  /** The in-process cache from earlier in this session. */
  | 'session'
  /** The library stored on disk by a previous run. */
  | 'stored';

export interface CatalogPage {
  cards: ModelCard[];
  /** Only categories that match at least one result, so no filter is a dead end. */
  categories: CategoryCount[];
  /** Memory available for weights; 0 when hardware could not be read. */
  weightBudgetBytes: number;
  /** System RAM available to hold a MoE model's offloaded experts; 0 if unknown. */
  hostRamBytes: number;
  /** Where these results came from, so the UI need not present stale data as current. */
  source: CatalogSource;
  /** Age of the served library in seconds. Absent for a live sweep. */
  ageSeconds?: number | null;
  /** True when a refresh is running behind this response; wait for `catalog:updated`. */
  refreshing: boolean;
  /** Explains a partial result, e.g. rate limiting or browsing without a token. */
  notice?: string | null;
  /**
   * The publisher this search turned out to be about, when the typed words
   * named one — "NVIDIA" for "nvidia nemotron", "DeepSeek" for "deepseek".
   *
   * Resolved in Rust against the same table the sweep used, so the browser can
   * rank that publisher's own releases first without keeping a second copy of
   * the alias list here and letting the two drift.
   *
   * Absent for ordinary searches and for every non-search listing.
   */
  matchedBrand?: string | null;
  /**
   * How many swept models were dropped for having no placement on this machine.
   *
   * Discover lists only models that run here, so this is the difference between
   * what the Hub offered and what is shown. Reported so the count can be stated
   * plainly rather than leaving the listing looking mysteriously short. Zero
   * when hardware could not be read, since nothing is filtered then.
   */
  hiddenIncompatible: number;
}

/**
 * Browses the catalog, optionally filtered by a search term.
 *
 * A search queries HuggingFace directly rather than filtering what is cached —
 * the cache holds only the popular sweep, so filtering it would report that a
 * specific fine-tune does not exist.
 *
 * Without a search this prefers a cached library and returns immediately, which
 * is why the response carries `source` and `refreshing`: the results may be a
 * saved copy with a sweep running behind them.
 */
export function browseModelCards(query?: string): Promise<CatalogPage> {
  return invoke<CatalogPage>('browse_model_cards', { query: query ?? null });
}

/**
 * Discards every cached library and sweeps HuggingFace again.
 *
 * The escape hatch for "there must be something newer than this" — everything
 * else prefers a cached answer.
 */
export function refreshModelLibrary(): Promise<CatalogPage> {
  return invoke<CatalogPage>('refresh_model_library');
}

/** How far along a sweep is. */
export interface CatalogProgress {
  phase: 'searching' | 'fetching' | 'caching' | 'done';
  /** One line describing what is happening, written for a person. */
  message: string;
  /**
   * Completed share of the current phase, 0–1.
   *
   * Absent while searching: how many repositories a sweep will find is not known
   * until the search pages are in, and a bar that guesses then jumps backwards
   * is worse than one that waits.
   */
  fraction?: number | null;
  done: number;
  total: number;
  /** True for a refresh behind an already-drawn listing, which stays quiet. */
  background: boolean;
}

/** Subscribes to sweep progress. Resolves to the unsubscribe function. */
export function onCatalogProgress(
  handler: (progress: CatalogProgress) => void
): Promise<UnlistenFn> {
  return listen<CatalogProgress>('catalog:progress', (e) => handler(e.payload));
}

/**
 * Fires when a background refresh has replaced the cached library, carrying the
 * number of repositories found. An open browser reloads on this rather than
 * making the user know to.
 */
export function onCatalogUpdated(handler: (count: number) => void): Promise<UnlistenFn> {
  return listen<number>('catalog:updated', (e) => handler(e.payload));
}

/** Every known category, for a sidebar that stays stable while results load. */
export function listModelCategories(): Promise<CategoryCount[]> {
  return invoke<CategoryCount[]>('list_model_categories');
}

/** A LoRA adapter published for some base model. */
export interface AdapterListing {
  repoId: string;
  name: string;
  author: string;
  downloads: number;
  likes: number;
  /**
   * True when the repo ships `.gguf` files. Most published adapters are PEFT
   * safetensors and cannot be loaded until converted — the card says so rather
   * than offering a download that will not work.
   */
  ggufReady: boolean;
  /** What the adapter is for, e.g. `text to sql`. */
  focus: string;
  /**
   * Bytes installing this adapter downloads — its weight file alone.
   *
   * `0` means the Hub did not report a size. Render that as unknown rather than
   * as nothing, so a running total never quietly understates itself.
   */
  sizeBytes: number;
  /**
   * Capability slot this adapter fills, from its own metadata.
   *
   * The author's tags decide it where they exist; the repository name is only a
   * fallback. `undefined` means neither settled it — shown as unsorted rather
   * than filed under a guess.
   */
  capability?: string;
  /** `stated` (author's tags) or `suggested` (read from the name). */
  capabilityConfidence?: string;
  /**
   * Whether Sarathi can install this against the model being viewed.
   *
   * `false` means the installer would refuse it — the reason is in
   * `blockedReason`, and no Get button should be offered.
   */
  installable: boolean;
  /** Why it cannot be installed, in plain language. */
  blockedReason?: string | null;
  /** What the adapter is for. Present on every listing the backend sends. */
  useCase?: AdapterUseCase;
}

/** What an adapter is for, and how Sarathi knows. */
export interface AdapterUseCase {
  /** One or two short labels, e.g. `Finance` or `Tax · Law`. */
  label: string;
  /** The language it targets, when that is not English. */
  language?: string;
  /** Where the label came from. `none` means nothing said. */
  source: 'tags' | 'name' | 'card' | 'laya' | 'none';
  /** The first sentence of its model card, when it has one. */
  summary?: string;
}

/** The chip text: `Finance` or `Lyrics & music · Nepali`. */
export function useCaseText(u: AdapterUseCase): string {
  return u.language ? `${u.label} · ${u.language}` : u.label;
}

/**
 * The chip's tooltip: where the label came from, then the card's own words.
 *
 * A label read from the author's tags and one guessed by Laya must not look
 * equally certain, so the source is always stated.
 */
export function useCaseTooltip(u: AdapterUseCase): string {
  const source = {
    tags: "From the author's tags.",
    name: "From the adapter's name.",
    card: 'From its model card.',
    laya: 'Picked by Laya from its description.',
    none: 'Its author does not say what it is for.',
  }[u.source];
  return u.summary ? `${source}\n\n${u.summary}` : source;
}

export interface AdapterPage {
  /**
   * The model these were found for.
   *
   * Shown so the list can say *which* model it searched: an answer is only
   * meaningful alongside its question, and for an installed model the id
   * searched is the original rather than the quantization on disk. Empty when
   * the base model could not be determined.
   */
  baseModelId: string;
  adapters: AdapterListing[];
  /** How many are loadable as-is. */
  readyCount: number;
  /** Whole hours since this was fetched, when served from cache and not fresh. */
  ageHours?: number | null;
  /** Explains an empty or unusable result. */
  notice?: string | null;
}

/**
 * LoRA adapters published for a base model.
 *
 * `baseModelId` must be the **original** model an adapter author declares, not
 * a quantization repository. For a model already installed, use
 * `findAdaptersForInstalledModel` instead — it resolves that id from the
 * package rather than making the caller know it.
 *
 * `refresh` bypasses the cache. That is what "Find more" does, and the only way
 * to make Sarathi ask HuggingFace again before the cache expires.
 */
export function findModelAdapters(
  baseModelId: string,
  architecture?: string | null,
  refresh = false
): Promise<AdapterPage> {
  return invoke<AdapterPage>('find_model_adapters', {
    baseModelId,
    architecture: architecture ?? null,
    refresh,
  });
}

/** Bytes as GB with one decimal, e.g. `4.7 GB`. */
export function formatSize(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}
