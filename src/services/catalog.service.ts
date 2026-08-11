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
}

export interface AdapterPage {
  adapters: AdapterListing[];
  /** How many are loadable as-is. */
  readyCount: number;
  /** Explains an empty or unusable result. */
  notice?: string | null;
}

/**
 * LoRA adapters published for a base model.
 *
 * Uses HuggingFace's `base_model:adapter:` tag, which adapter authors set to
 * declare their parent — a real relationship, not a name-similarity guess.
 */
export function findModelAdapters(baseModelId: string): Promise<AdapterPage> {
  return invoke<AdapterPage>('find_model_adapters', { baseModelId });
}

/** Bytes as GB with one decimal, e.g. `4.7 GB`. */
export function formatSize(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}
