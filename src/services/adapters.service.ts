// Adapter download and management IPC.
//
// Mirrors src-tauri/src/commands/adapters.rs.

import { invoke } from '@tauri-apps/api/core';

import type { AdapterPage } from './catalog.service';

/**
 * Capability slots the runtime can route a turn to.
 *
 * Mirrors CapabilitySpec::all_keys() in capability/profile.rs. An adapter
 * assigned to none of these is installed but never bound.
 */
export const CAPABILITIES = [
  'coding',
  'reasoning',
  'tool-calling',
  'mathematics',
  'research',
] as const;

export type Capability = (typeof CAPABILITIES)[number];

export const CAPABILITY_LABELS: Record<Capability, string> = {
  coding: 'Coding',
  reasoning: 'Reasoning',
  'tool-calling': 'Tool calling',
  mathematics: 'Mathematics',
  research: 'Research',
};

/** How an adapter's capability slot was arrived at — see capability/assign.rs. */
export type AssignmentConfidence = 'stated' | 'suggested' | 'manual';

/** Plain-language provenance, so a guess is never shown as a fact. */
export const ASSIGNMENT_SOURCE: Record<AssignmentConfidence, string> = {
  stated: "from the author's tags",
  suggested: "guessed from the adapter's name",
  manual: 'your choice',
};

export interface InstalledAdapter {
  id: string;
  repoId: string;
  name: string;
  baseModelId: string;
  filePath: string;
  sizeBytes: number;
  /** Slot this adapter is bound to. Absent means installed but unused. */
  capability?: Capability;
  assignmentConfidence?: AssignmentConfidence;
  /**
   * Whether this is the adapter its capability actually binds.
   *
   * Several adapters can serve one capability — a Python specialist and a
   * general coding adapter are both `coding` — but only one is used. Without
   * this the list would show four coding adapters with no way to tell which one
   * the model is running.
   */
  isDefault: boolean;
}

export interface InstalledAdapters {
  adapters: InstalledAdapter[];
  totalBytes: number;
}

export function listInstalledAdapters(
  providerId: string,
  modelId: string
): Promise<InstalledAdapters> {
  return invoke<InstalledAdapters>('list_installed_adapters', { providerId, modelId });
}

/**
 * Downloads an adapter for a model.
 *
 * Rejects PEFT safetensors adapters before fetching anything — llama.cpp cannot
 * load them, so the error explains that rather than leaving a file that fails
 * only when a request needs it.
 */
export function downloadAdapter(
  providerId: string,
  modelId: string,
  adapterRepoId: string
): Promise<InstalledAdapter> {
  return invoke<InstalledAdapter>('download_adapter', { providerId, modelId, adapterRepoId });
}

export function removeAdapter(
  providerId: string,
  modelId: string,
  adapterId: string
): Promise<void> {
  return invoke<void>('remove_adapter', { providerId, modelId, adapterId });
}

/**
 * Points a capability at this adapter, or unassigns it with `null`.
 *
 * Assigning is also a choice to use it, so this adapter becomes the
 * capability's default. Any adapter previously bound to that capability stays
 * installed and keeps its own capability — it is simply no longer the one the
 * model runs with. Use `setCapabilityDefault` to switch between them.
 */
export function setAdapterCapability(
  providerId: string,
  modelId: string,
  adapterId: string,
  capability: Capability | null
): Promise<void> {
  return invoke<void>('set_adapter_capability', {
    providerId,
    modelId,
    adapterId,
    capability,
  });
}

/** How sure a claim about an adapter is — see adapter_details.rs. */
export type Confidence = 'stated' | 'suggested';

export interface AdapterEffect {
  skill: string;
  description: string;
  confidence: Confidence;
}

export interface AdapterDetails {
  repoId: string;
  name: string;
  author: string;
  license: string | null;
  downloads: number;
  likes: number;
  sizeBytes: number;
  ggufReady: boolean;
  blockedReason: string | null;
  datasets: string[];
  effects: AdapterEffect[];
  notice: string | null;
}

/**
 * What an adapter changes about the model it is applied to.
 *
 * Every effect is labelled with where it came from — the author's own tags
 * (`stated`) or the repository name (`suggested`) — so a hint is never shown
 * as a fact.
 */
export function getAdapterDetails(repoId: string): Promise<AdapterDetails> {
  return invoke<AdapterDetails>('get_adapter_details', { repoId });
}

/**
 * Chooses which installed adapter a capability binds.
 *
 * Several adapters can serve one capability, but only one can be bound: a turn
 * resolves to a capability and the runtime needs a single answer. This is how
 * the user says which.
 *
 * Fails if the adapter does not serve that capability — binding a maths adapter
 * to `coding` would apply weights trained for one task to another.
 */
export function setCapabilityDefault(
  providerId: string,
  modelId: string,
  capability: Capability,
  adapterId: string
): Promise<void> {
  return invoke<void>('set_capability_default', {
    providerId,
    modelId,
    capability,
    adapterId,
  });
}

/**
 * Finds adapters compatible with a model that is already installed.
 *
 * The point of this over `findModelAdapters` is that the caller does not need
 * to know the base model id. Sarathi resolves it from the installed package, so
 * the user never has to go back to the library and search for their own model
 * again — and never risks picking the wrong repository while doing it.
 *
 * Returns a page with an explanatory `notice` and no adapters when the base
 * model cannot be determined. That is deliberately distinct from an empty list:
 * "no adapters exist" and "I do not know where to look" are different facts.
 */
export function findAdaptersForInstalledModel(
  providerId: string,
  modelId: string,
  refresh = false
): Promise<AdapterPage> {
  return invoke<AdapterPage>('find_adapters_for_installed_model', {
    providerId,
    modelId,
    refresh,
  });
}
