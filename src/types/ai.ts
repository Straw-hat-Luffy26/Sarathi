export type RuntimeStatus =
  | 'NotLoaded'
  | 'Loading'
  | 'Ready'
  | 'Generating'
  | 'Unloading'
  | 'Error';

export interface LoadedModelInfo {
  modelId: string;
  modelName: string;
  quantization: string;
  filePath: string;
  contextLength: number;
  gpuLayers: number;
  threads: number;
  backendUsed: string;
  chatTemplate?: string;
  stopTokens?: string[];
  modelFamily?: string;
  activeAdapter?: string | null;
}

export interface InferenceStatusPayload {
  status: string;
  step?: string | null;
  model?: LoadedModelInfo | null;
  error?: string | null;
}

export interface StreamChunkPayload {
  text: string;
  isFinal: boolean;
  tokensGenerated?: number | null;
  finishReason?: string | null;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface GenerationParams {
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  repeatPenalty?: number;
}

/**
 * What a model is for. A model carries several — a small coding model is both
 * `coding` and `small-and-fast`, so either filter surfaces it.
 */
export type ModelCategory =
  | 'lora-adapter'
  | 'coding'
  | 'reasoning'
  | 'math'
  | 'agentic'
  | 'vision'
  | 'mixture-of-experts'
  | 'moe-offloadable'
  | 'long-context'
  | 'multilingual'
  | 'small-and-fast'
  | 'general';

/**
 * Display order for the category sidebar.
 *
 * `lora-adapter` leads because it changes what an entry *is* — a patch applied
 * on top of a base model, not something that runs on its own — rather than what
 * it is good at.
 */
export const MODEL_CATEGORIES: readonly ModelCategory[] = [
  'lora-adapter',
  'coding',
  'reasoning',
  'math',
  'agentic',
  'vision',
  'mixture-of-experts',
  'moe-offloadable',
  'long-context',
  'multilingual',
  'small-and-fast',
  'general',
] as const;

export const CATEGORY_LABELS: Record<ModelCategory, string> = {
  'lora-adapter': 'LoRA adapter',
  coding: 'Coding',
  reasoning: 'Reasoning',
  math: 'Math',
  agentic: 'Agentic',
  vision: 'Vision',
  'mixture-of-experts': 'MoE',
  'moe-offloadable': 'Runs here (MoE offload)',
  'long-context': 'Long context',
  multilingual: 'Multilingual',
  'small-and-fast': 'Small & fast',
  general: 'General',
};

/**
 * What an entry is, in terms someone can act on.
 *
 * The distinction that matters: a base model, fine-tune, or quantization can be
 * downloaded and run. A LoRA adapter cannot — it needs a base model underneath.
 */
export type ModelKind = 'base-model' | 'fine-tuned' | 'quantized' | 'lora-adapter';

export const KIND_LABELS: Record<ModelKind, string> = {
  'base-model': 'Base model',
  'fine-tuned': 'Fine-tuned',
  quantized: 'Quantized',
  'lora-adapter': 'LoRA adapter',
};

/**
 * A MoE placement this machine can execute — routed experts held in system RAM
 * while attention and the KV cache stay on the GPU.
 *
 * Computed from detected VRAM and RAM on every sweep, never stored: the same
 * model is offloadable on one computer and out of reach on another.
 */
export interface ExpertOffload {
  /** Routed experts of this many leading layers move to system RAM. */
  cpuMoeLayers: number;
  totalLayers: number;
  /** System RAM the offloaded experts occupy. */
  hostBytes: number;
  /** Parameters used per token — why a model this size stays responsive. */
  activeParameters: number;
  /** One line for the UI, in the terms someone choosing a download needs. */
  note: string;
}

/** One quantization choice, for the size/quality comparison. */
export interface QuantizationOption {
  label: string;
  sizeBytes: number;
  /** Plain-English quality note, e.g. "Balanced — recommended". */
  qualityNote: string;
  /**
   * Whether the weights fit in VRAM as they are.
   *
   * A statement about VRAM alone. A MoE model that runs only by moving experts
   * to system RAM reports `false` here and describes itself in `offload` —
   * loading onto the card and loading across the PCIe bus are a real difference
   * in speed, and folding them together would hide it.
   */
  fits: boolean;
  /** True for 1–2 bit quantizations, where output can degrade to nonsense. */
  lowQuality?: boolean;
  /** Set when this build runs here by offloading experts to system RAM. */
  offload?: ExpertOffload | null;
  /**
   * Why there is no placement, when a MoE model cannot run here. Names whether
   * RAM or VRAM was short, which have opposite remedies.
   */
  offloadBlockedReason?: string | null;
}

/**
 * How a build runs on this machine.
 *
 * Mirrors `card::Placement` in Rust, which has no "cannot run" variant on
 * purpose: a model with no placement is not listed at all.
 */
export type Placement = 'vram' | 'offload';

/** Whether a build runs here, and by what route. */
export function runsHere(q: QuantizationOption): Placement | 'no' {
  if (q.fits) return 'vram';
  if (q.offload) return 'offload';
  return 'no';
}

/**
 * The shelf an installed model sits on in Storage.
 *
 * Decided in Rust from the file's GGUF header, never from its name — see
 * `model_manager::classify`. The categories below it are the same
 * `ModelCategory` values Discover files models under, so one model is described
 * the same way on both screens.
 */
export type ModelGroup =
  | 'mixture-of-experts'
  | 'dense'
  | 'vision'
  | 'embedding'
  /** A LoRA, a projector, a speculative-decoding draft. Not loadable. */
  | 'auxiliary'
  /** The header could not be read. */
  | 'unknown';

/** Display order: what you can chat with first, then everything else. */
export const MODEL_GROUPS: readonly ModelGroup[] = [
  'mixture-of-experts',
  'dense',
  'vision',
  'embedding',
  'auxiliary',
  'unknown',
] as const;

export const GROUP_LABELS: Record<ModelGroup, string> = {
  'mixture-of-experts': 'Mixture of experts',
  dense: 'Dense',
  vision: 'Vision',
  embedding: 'Embedding',
  auxiliary: 'Helper files',
  unknown: 'Unrecognised',
};

export const GROUP_DESCRIPTIONS: Record<ModelGroup, string> = {
  'mixture-of-experts':
    'Only some of the weights run for each word, so these can be far larger than your graphics card and still be quick.',
  dense: 'Every weight runs for every word. The ordinary kind.',
  vision: 'These can be shown images as well as text.',
  embedding:
    'These turn text into vectors for search and comparison. They do not hold conversations.',
  auxiliary:
    'Files that support a model rather than being one. They cannot be loaded on their own.',
  unknown: 'Sarathi could not read these files well enough to say.',
};

/** Groups whose models can actually be loaded and talked to. */
export function isLoadableGroup(group: ModelGroup): boolean {
  return group === 'mixture-of-experts' || group === 'dense' || group === 'vision';
}

/** What an installed file actually is, read from its GGUF header. */
export interface Classification {
  group: ModelGroup;
  /** Shared with Discover's filters. */
  categories: ModelCategory[];
  /** GGUF `general.architecture`, e.g. `qwen3moe`, `gpt-oss`, `llama`. */
  architecture: string;
  isMoe: boolean;
  expertCount: number;
  /** Experts consulted per token — why a large MoE stays responsive. */
  expertUsedCount: number;
  blockCount: number;
  parameterCount?: number | null;
  contextLength?: number | null;
  /** As the file declares it, which is not always what the filename says. */
  quantization?: string | null;
  /** Why this cannot be loaded, when it cannot. */
  notLoadableReason?: string | null;
}

/**
 * Presentation data for a model listing card.
 *
 * Optional fields are genuinely unknown rather than zero — the card omits them
 * instead of showing a placeholder that looks like a measurement.
 */
export interface ModelCard {
  repoId: string;
  /** HuggingFace org or user. */
  publisher: string;
  name: string;
  /** Factual one-liner assembled from metadata, not marketing copy. */
  summary: string;
  categories: ModelCategory[];
  /** Base model, fine-tune, quantization, or adapter. */
  kind: ModelKind;
  /** Plain-language note on whether this can run on its own. */
  kindExplanation: string;
  /** Datasets it was trained on, when named and informative. */
  datasets?: string[];
  license?: string | null;
  downloads: number;
  /** Compact form, e.g. "52M". */
  downloadsLabel: string;
  likes: number;
  /** Compact relative age, e.g. "2mo". Absent when the date is unknown. */
  ageLabel?: string | null;
  isFinetune: boolean;
  /**
   * True when this is a LoRA adapter — a patch for a base model, not something
   * that can be loaded and run on its own.
   */
  isLoraAdapter: boolean;
  /** Parent model, when this is a fine-tune, adapter, or re-quantization. */
  baseModel?: string | null;
  totalParameters?: number | null;
  contextLength?: number | null;
  quantizations: QuantizationOption[];
  /**
   * How this model runs on this machine, decided by the Rust planner.
   *
   * Discover only lists models where this is set, so in a normal listing it is
   * always present. It is optional because a card built on hardware Sarathi
   * could not read carries no placement.
   */
  runsHere?: Placement | null;
  /**
   * The quantization Discover offers, chosen by the planner.
   *
   * Read rather than re-derived: picking the build in the UI as well would give
   * two answers to one question, and they would drift.
   */
  bestQuantization?: string | null;
  /**
   * True when Sarathi vouches for this one: a publisher whose conversions are
   * dependable, enough real use to have surfaced problems, no reasoning tokens
   * in its output, and a size that runs on this machine.
   */
  recommended: boolean;
  /**
   * True when the model narrates its reasoning (`<think>` and similar) into its
   * replies. Read from the GGUF's own chat template where one was published.
   */
  emitsReasoning: boolean;
  /** ISO-8601 last-modified stamp, for sorting. `ageLabel` is the readable form. */
  lastModified: string;
}

/** How a capability is actually being applied to the model. */
export type CapabilityBackend = 'base' | 'prompt-profile' | 'lora';

/**
 * Emitted by the backend on `capability:changed` *after* the capability has
 * been applied to the prompt, sampler, and adapter binding.
 *
 * This replaces the previous client-side `routePromptCapability` badge, which
 * displayed a routing decision that never reached inference.
 */
export interface CapabilityPayload {
  capability: string;
  displayName: string;
  /** Pre-formatted label, e.g. `Code · lora`. */
  badge: string;
  backend: CapabilityBackend;
  confidence: number;
  switched: boolean;
  /** Why this capability is active (classification / hysteresis / override). */
  reason: string;
  /** Why this backend was chosen, including any degradation from LoRA. */
  backendReason: string;
  adapterPath?: string | null;
  /** Sampling values generation actually used, after capability overrides. */
  effectiveTemperature: number;
  effectiveTopP: number;
  effectiveRepeatPenalty: number;
}