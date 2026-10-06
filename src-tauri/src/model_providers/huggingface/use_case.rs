//! What a LoRA adapter is *for*, in a word or two, for every adapter listed.
//!
//! The capability slots (`coding`, `mathematics`, …) only cover what the router
//! can switch to, so an adapter for tax law, Nepali song lyrics or crop advice
//! used to be listed with nothing but its repository name. This answers the
//! question a person actually has when scanning the list — "what would I use
//! this for?" — for all of them.
//!
//! Evidence is read in the order it can be trusted:
//!
//! 1. **The author's tags** — something they chose to declare.
//! 2. **The repository name** — usually descriptive, sometimes a brand.
//! 3. **The model card** — the first prose of the README, fetched once and
//!    cached with the listing. This is what explains a name like `rinlekha`.
//! 4. **Laya** — when it is installed and none of the above matched, it reads
//!    the name and card and picks a use case, kept only when confident.
//!
//! Task and library tags (`image-text-to-text`, `peft`, `gguf`, …) are not
//! evidence: every Gemma 3 adapter carries `image-text-to-text` because the base
//! model does, text-only or not. When nothing settles it, the answer says so
//! rather than guessing.

use serde::{Deserialize, Serialize};

/// The use case shown beside an adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UseCase {
    /// One or two short labels, e.g. `Finance` or `Tax · Law`.
    pub label: String,
    /// The language it targets, when that is not English.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Where the label came from: `tags`, `name`, `card`, `laya`, or `none`.
    pub source: String,
    /// The model card's own first sentence, when one was found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Shown when no evidence names a use case.
pub const NOT_STATED: &str = "Not stated";

/// Use cases, most specific first — the order breaks ties.
///
/// Labels are kept short because they sit in a narrow table column. Keywords
/// are matched as whole words; entries with a space are matched as phrases.
pub const DOMAINS: &[(&str, &[&str])] = &[
    ("SQL", &["sql", "text2sql", "nl2sql", "text to sql", "sqlite", "postgres", "postgresql", "mysql"]),
    ("Coding", &["code", "coder", "coding", "programming", "python", "javascript", "typescript", "rust",
        "golang", "java", "codegen", "software", "developer", "debugging", "leetcode", "humaneval",
        "verilog", "solidity", "kotlin", "swift"]),
    ("Math", &["math", "maths", "mathematics", "mathematical", "gsm8k", "algebra", "arithmetic",
        "calculus", "geometry", "equation", "equations", "calc", "olympiad", "aime"]),
    ("Reasoning", &["reasoning", "cot", "chain of thought", "logic", "logical", "thinking", "reasoner",
        // Reinforcement learning on verifiable rewards — reasoning training.
        "rlvr", "grpo"]),
    ("Tool use", &["function calling", "function call", "tool use", "tool calling", "tool calls",
        "agent", "agents", "agentic", "web search", "structured output", "json mode"]),
    ("Summarization", &["summarization", "summarisation", "summarize", "summarise", "summary",
        "summaries", "tldr"]),
    ("Research & Q&A", &["research", "rag", "retrieval", "question answering", "qa", "knowledge base",
        "information extraction", "extraction", "extractor", "relation extraction"]),
    ("Finance", &["finance", "financial", "fintech", "credit", "loan", "loans", "banking", "bank", "nbfc",
        "stock", "stocks", "trading", "investment", "investing", "crypto", "economics", "insurance"]),
    ("Tax", &["tax", "taxes", "taxation", "gst", "vat", "accounting", "accountant", "audit", "bookkeeping"]),
    ("Law", &["law", "laws", "legal", "lawyer", "court", "contract", "contracts", "statute", "judicial",
        "compliance", "regulatory", "jurisprudence"]),
    ("Medicine", &["medical", "medicine", "clinical", "health", "healthcare", "doctor", "patient",
        "patients", "biomedical", "pubmed", "radiology", "pharma", "pharmacy", "diagnosis",
        "mental health", "therapy", "nutrition", "drug", "drugs", "adverse", "pharmacovigilance"]),
    ("Science", &["science", "scientific", "chemistry", "physics", "biology", "molecule", "molecular",
        "protein"]),
    ("Agriculture", &["agriculture", "agricultural", "agronomy", "farming", "farmer", "farmers", "crop",
        "crops", "soil", "agri"]),
    ("Education", &["education", "educational", "tutor", "tutoring", "teaching", "teacher", "student",
        "students", "exam", "exams", "quiz", "homework", "curriculum"]),
    ("Lyrics & music", &["lyrics", "lyric", "song", "songs", "music", "rap", "songwriting"]),
    ("Creative writing", &["story", "stories", "storytelling", "fiction", "novel", "poem", "poems",
        "poetry", "creative", "writer", "screenplay"]),
    ("Roleplay", &["roleplay", "role play", "rp", "character", "characters", "persona", "companion", "npc"]),
    ("Translation", &["translation", "translate", "translator", "machine translation", "bilingual"]),
    ("Safety", &["harm", "harmful", "safety", "moderation", "toxicity", "toxic", "guard", "guardrail",
        "guardrails", "jailbreak", "hate", "nsfw"]),
    ("Classification", &["classifier", "classification", "classify", "sentiment", "intent", "ner",
        "labeling", "labelling", "tagging"]),
    ("Captioning", &["caption", "captioning", "captions", "captioner", "alt text"]),
    ("OCR & documents", &["ocr", "document", "documents", "receipt", "receipts", "invoice", "invoices",
        "pdf", "scanned", "handwriting", "docvqa"]),
    ("Vision", &["vlm", "vision", "visual", "vqa", "multimodal", "photo", "photos", "video"]),
    ("Speech", &["speech", "asr", "tts", "audio", "voice", "transcription"]),
    ("Cybersecurity", &["cybersecurity", "cyber", "malware", "pentest", "pentesting", "vulnerability",
        "exploit", "infosec"]),
    ("Customer support", &["customer support", "customer service", "helpdesk", "support tickets"]),
    ("Marketing", &["marketing", "sales", "seo", "ecommerce", "advertising", "copywriting"]),
    ("Data analysis", &["analytics", "tabular", "csv", "spreadsheet", "data analysis"]),
    ("General chat", &["chat", "chatbot", "assistant", "general purpose", "instruction following",
        "dialogue", "conversation"]),
];

/// Tags that describe the file, the library or the base model's task rather
/// than what the adapter is for.
const NON_EVIDENCE_TAGS: &[&str] = &[
    "image-text-to-text", "text-generation", "text2text-generation", "conversational", "transformers",
    "safetensors", "gguf", "peft", "lora", "qlora", "rs-lora", "dora", "adapter", "merged", "finetune",
    "fine-tuned", "sft", "trl", "sfttrainer", "unsloth", "llama-factory", "mlx", "vllm", "axolotl",
    "text-generation-inference", "endpoints_compatible", "experimental", "distillation",
];

/// Language names and the ISO 639-1 codes HuggingFace uses as tags.
const LANGUAGES: &[(&str, &str)] = &[
    ("ar", "Arabic"), ("as", "Assamese"), ("bn", "Bengali"), ("cs", "Czech"), ("de", "German"),
    ("el", "Greek"), ("es", "Spanish"), ("fa", "Persian"), ("fil", "Filipino"), ("fr", "French"),
    ("gu", "Gujarati"), ("ha", "Hausa"), ("he", "Hebrew"), ("hi", "Hindi"), ("id", "Indonesian"),
    ("it", "Italian"), ("ja", "Japanese"), ("kn", "Kannada"), ("ko", "Korean"), ("ml", "Malayalam"),
    ("mr", "Marathi"), ("ms", "Malay"), ("ne", "Nepali"), ("nl", "Dutch"), ("oc", "Occitan"),
    ("or", "Odia"), ("pa", "Punjabi"), ("pl", "Polish"), ("pt", "Portuguese"), ("ru", "Russian"),
    ("si", "Sinhala"), ("sv", "Swedish"), ("sw", "Swahili"), ("ta", "Tamil"), ("te", "Telugu"),
    ("th", "Thai"), ("tl", "Tagalog"), ("tr", "Turkish"), ("uk", "Ukrainian"), ("ur", "Urdu"),
    ("vi", "Vietnamese"), ("yo", "Yoruba"), ("zh", "Chinese"), ("zu", "Zulu"),
];

/// Lowercased words separated by single spaces, padded so ` phrase ` matches
/// whole words only.
fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push(' ');
    let mut last_space = true;
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            out.push(c);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    if !last_space {
        out.push(' ');
    }
    out
}

fn evidence_tags(tags: &[String]) -> Vec<&str> {
    tags.iter()
        .map(String::as_str)
        .filter(|t| !t.contains(':') && !NON_EVIDENCE_TAGS.contains(&t.to_ascii_lowercase().as_str()))
        .collect()
}

/// Which source a match came from, strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Source {
    Tags,
    Name,
    Card,
}

impl Source {
    fn weight(self) -> u32 {
        match self {
            Self::Tags => 3,
            Self::Name => 2,
            Self::Card => 1,
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::Tags => "tags",
            Self::Name => "name",
            Self::Card => "card",
        }
    }
}

/// The repository part of `org/repo`, with model-family boilerplate left in —
/// it never matches a use-case keyword, so there is nothing to gain by guessing
/// which words are boilerplate.
fn repo_name(repo_id: &str) -> &str {
    repo_id.rsplit('/').next().unwrap_or(repo_id)
}

/// Scores every domain against the evidence. Returns (domain index, score,
/// strongest source) for each domain that matched, best first.
fn score(repo_id: &str, tags: &[String], card: Option<&str>) -> Vec<(usize, u32, Source)> {
    let sources = [
        (Source::Tags, normalise(&evidence_tags(tags).join(" "))),
        (Source::Name, normalise(repo_name(repo_id))),
        (Source::Card, normalise(card.unwrap_or(""))),
    ];

    let mut scored: Vec<(usize, u32, Source)> = DOMAINS
        .iter()
        .enumerate()
        .filter_map(|(i, (_, keywords))| {
            let mut total = 0;
            let mut best: Option<Source> = None;
            for (source, text) in &sources {
                if keywords.iter().any(|k| text.contains(&format!(" {k} "))) {
                    total += source.weight();
                    best = Some(best.map_or(*source, |b| b.min(*source)));
                }
            }
            best.map(|b| (i, total, b))
        })
        .collect();

    // Highest score first; the list's own order breaks ties.
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    scored
}

/// The non-English language an adapter targets, when its metadata says.
pub fn language(repo_id: &str, tags: &[String], card: Option<&str>) -> Option<String> {
    // Codes only from tags: in a name, `it` means "instruct" and `id` is a word.
    for tag in tags {
        let t = tag.to_ascii_lowercase();
        if let Some((_, name)) = LANGUAGES.iter().find(|(code, _)| *code == t) {
            return Some((*name).to_string());
        }
    }
    let haystacks = [normalise(&tags.join(" ")), normalise(repo_name(repo_id)), normalise(card.unwrap_or(""))];
    for text in &haystacks {
        for (_, name) in LANGUAGES {
            if text.contains(&format!(" {} ", name.to_lowercase())) {
                return Some((*name).to_string());
            }
        }
    }
    None
}

/// The use case from what is already known: name, tags, and a card when one has
/// been fetched. Never touches the network.
pub fn from_metadata(repo_id: &str, tags: &[String], card: Option<&str>) -> UseCase {
    let scored = score(repo_id, tags, card);
    let language = language(repo_id, tags, card);
    let summary = card.and_then(first_sentence);

    let Some(&(best, best_score, source)) = scored.first() else {
        // A language-only adapter is still a clear use case: it teaches the base
        // model a language.
        let (label, source) = match &language {
            Some(lang) => (format!("{lang} language"), "tags"),
            None => (NOT_STATED.to_string(), "none"),
        };
        return UseCase { label, language: None, source: source.to_string(), summary };
    };

    // A second label only when it is nearly as well supported — "Tax · Law" for
    // a tax-law adapter, not every word that happened to match.
    let mut label = DOMAINS[best].0.to_string();
    if let Some(&(second, second_score, _)) = scored.get(1) {
        if second_score * 2 >= best_score && second_score >= 2 && DOMAINS[second].0 != "General chat" {
            let (first, then) = if second < best { (second, best) } else { (best, second) };
            label = format!("{} · {}", DOMAINS[first].0, DOMAINS[then].0);
        }
    }

    UseCase { label, language, source: source.key().to_string(), summary }
}

/// The routing slot a use case belongs in, when it belongs in one.
///
/// The router sends each chat turn to one of five slots, so this is the
/// question "which kind of turn should this adapter answer?". Domain-knowledge
/// and extraction adapters — finance, law, medicine, documents — answer the
/// "find and synthesise from sources" turns the research slot exists for.
///
/// Creative writing, lyrics, roleplay, translation and general chat have no
/// slot: the router files those turns as general conversation and answers them
/// with the base model, so an adapter filed under research would only ever
/// fire on the wrong prompts. Those return `None` rather than a misfiling.
pub fn slot_for(use_case: &UseCase) -> Option<&'static str> {
    let slot = |label: &str| -> Option<&'static str> {
        Some(match label {
            "SQL" | "Coding" => "coding",
            "Math" => "mathematics",
            "Reasoning" => "reasoning",
            "Tool use" => "tool-calling",
            "Summarization" | "Research & Q&A" | "OCR & documents" | "Classification" | "Medicine"
            | "Law" | "Tax" | "Finance" | "Science" | "Agriculture" | "Education" | "Data analysis"
            | "Cybersecurity" | "Safety" => "research",
            _ => return None,
        })
    };
    // `Tax · Law` and `Safety · Classification` are read label by label, the
    // stronger first.
    use_case.label.split(" · ").find_map(slot)
}

/// Whether a card is worth fetching to settle this one.
pub fn needs_card(use_case: &UseCase) -> bool {
    use_case.summary.is_none()
}

/// Folds a fetched model card into a verdict made from tags and the name.
///
/// A label the author's tags or the name already settled is kept — the card is
/// prose and its keywords are the weakest evidence. The card contributes its
/// summary always, and the label only when nothing else had one.
pub fn with_card(existing: &UseCase, repo_id: &str, card: &str) -> UseCase {
    let from_card = from_metadata(repo_id, &[], Some(card));
    let mut out = if existing.source == "none" { from_card.clone() } else { existing.clone() };
    out.summary = out.summary.or(from_card.summary);
    if out.language.is_none() && !out.label.ends_with(" language") {
        out.language = from_card.language;
    }
    out
}

/// The first sentence of a model card's prose, without markdown.
///
/// Skips the YAML header, HTML comments (where auto-generated cards hide their
/// "fill this in" notes), headings, tables, code, badges and link-only lines.
pub fn first_sentence(readme: &str) -> Option<String> {
    let mut text = readme;
    if let Some(rest) = text.strip_prefix("---") {
        text = rest.split_once("\n---").map(|(_, after)| after).unwrap_or(rest);
    }

    // Drop HTML comments wholesale.
    let mut clean = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<!--") {
        clean.push_str(&rest[..start]);
        rest = rest[start..].split_once("-->").map(|(_, after)| after).unwrap_or("");
    }
    clean.push_str(rest);

    let mut in_code = false;
    for line in clean.lines() {
        let line = line.trim();
        if line.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code
            || line.is_empty()
            || line.starts_with('#')
            || line.starts_with('|')
            || line.starts_with('<')
            || line.starts_with("![")
            || line.starts_with("[!")
            || line.starts_with('>')
        {
            continue;
        }

        let plain = strip_markdown(line);
        let plain = plain.trim().trim_start_matches(['-', '*', ' ']).trim();
        // Link-only and label-only lines say nothing about the purpose.
        if plain.chars().filter(|c| c.is_alphabetic()).count() < 25 {
            continue;
        }
        if is_template_line(line, plain) {
            continue;
        }

        let sentence = match plain.find(". ") {
            Some(end) if end >= 30 => &plain[..=end],
            _ => plain,
        };
        return Some(truncate(sentence.trim(), 200));
    }
    None
}

/// Lines from HuggingFace's default model-card template, which an author who
/// never edited the card leaves in place. They describe the template, not the
/// adapter: `- **Developed by:** [More Information Needed]`, or "This model card
/// has been automatically generated."
fn is_template_line(raw: &str, plain: &str) -> bool {
    /// Stock sentences of HuggingFace's model-card and PEFT templates.
    const TEMPLATE_PHRASES: &[&str] = &[
        "more information needed",
        "automatically generated",
        "this is the model card of",
        "use the code below to get started",
        "this section is meant to convey",
        "users (both direct and downstream) should be made aware",
        "carbon emissions can be estimated",
        "proofread and complete it",
        "provide a quick summary",
        "provide a longer summary",
    ];
    let lower = plain.to_lowercase();
    if TEMPLATE_PHRASES.iter().any(|p| lower.contains(p)) {
        return true;
    }
    // A bulleted `Label: value` line is a field in a form, not a description.
    let bulleted = raw.starts_with('-') || raw.starts_with('*') && !raw.starts_with("**");
    bulleted && plain.find(':').is_some_and(|i| i <= 40 && plain[..i].split_whitespace().count() <= 5)
}

/// Removes emphasis and keeps link text: `**[GLM](url)**` -> `GLM`.
fn strip_markdown(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' | '_' | '`' => {}
            ']' if chars.peek() == Some(&'(') => {
                // Skip `(url)`.
                for d in chars.by_ref() {
                    if d == ')' {
                        break;
                    }
                }
            }
            '[' | ']' => {}
            _ => out.push(c),
        }
    }
    out
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let cut: String = text.chars().take(max_chars).collect();
    let cut = cut.rsplit_once(' ').map(|(head, _)| head).unwrap_or(&cut);
    format!("{cut}…")
}

/// The head of an adapter's README, fetched on the calling thread.
///
/// For background threads only — reqwest's blocking client must not run on the
/// async runtime. Reads at most 8 KB: the purpose is in the first paragraph,
/// and some cards run to megabytes of benchmark tables.
pub fn fetch_card_blocking(repo_id: &str, token: Option<&str>) -> Option<String> {
    use std::io::Read;
    const CARD_BYTES: u64 = 8 * 1024;

    let client = reqwest::blocking::Client::builder()
        .user_agent("Sarathi/0.1.0")
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .ok()?;
    let mut req = client
        .get(format!("https://huggingface.co/{repo_id}/raw/main/README.md"))
        .header(reqwest::header::RANGE, format!("bytes=0-{}", CARD_BYTES - 1));
    if let Some(t) = token.map(str::trim).filter(|t| !t.is_empty()) {
        req = req.bearer_auth(t);
    }
    let resp = req.send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let mut body = Vec::new();
    resp.take(CARD_BYTES).read_to_end(&mut body).ok()?;
    Some(String::from_utf8_lossy(&body).into_owned())
}

/// Asks Laya to pick a use case from the card when no keyword settled it.
///
/// Returns `None` when Laya is not running or not confident — a calibrated
/// "not sure" is kept as "Not stated", never promoted to a label.
pub fn from_laya(repo_id: &str, summary: Option<&str>) -> Option<String> {
    /// Laya's calibrated probability below which its pick is not shown.
    const MIN_CONFIDENCE: f32 = 0.5;

    let router = crate::capability::laya::global()?;
    let text = match summary {
        Some(s) => format!("{} — {s}", repo_name(repo_id)),
        None => repo_name(repo_id).to_string(),
    };
    let labels: Vec<&str> = DOMAINS.iter().map(|(label, _)| *label).collect();
    let (choice, confidence) = router.choose(
        &text,
        "What is the LoRA adapter described in `request` fine-tuned to do?",
        &labels,
    )?;
    (confidence >= MIN_CONFIDENCE).then_some(choice)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// The adapters in the screenshot that prompted this, with their real tags.
    #[test]
    fn every_adapter_in_the_gemma_3_list_gets_a_use_case() {
        let cases = [
            ("pradipbasnet68/gemma-3-4b-nepali-lyrics-lora",
                tags(&["peft", "lora", "nepali", "devanagari", "lyrics-generation", "text-generation"]),
                "Lyrics & music", Some("Nepali")),
            ("a-anurag1024/rinlekha-gemma3-4b-finetuned",
                tags(&["safetensors", "finance", "credit-analysis", "nbfc", "qlora", "en"]),
                "Finance", None),
            ("Toxotes/gemma3-finance-v1-gguf", tags(&["gguf", "lora"]), "Finance", None),
            ("crystalseok/gemma3-4b-shorts-harm-classifier", tags(&["peft"]), "Safety · Classification", None),
            ("DreamingBumblebee/gemma-3-4b-vlm-qlora-sft-ko-1.5k", tags(&["peft", "ko"]), "Vision", Some("Korean")),
            ("manifesta/adaption-agronomy-calc-problems", tags(&["peft"]), "Math · Agriculture", None),
            ("itcen-entec/gemma-3-4b-tax-law-lora-gguf", tags(&["gguf"]), "Tax · Law", None),
            ("younaice/gemma3-4b-bean-captioning", tags(&["peft"]), "Captioning", None),
        ];
        for (id, t, label, lang) in cases {
            let uc = from_metadata(id, &t, None);
            assert_eq!(uc.label, label, "{id}");
            assert_eq!(uc.language.as_deref(), lang, "{id}");
            assert_ne!(uc.source, "none", "{id}");
        }
    }

    #[test]
    fn the_base_models_task_tag_is_not_evidence() {
        // Every Gemma 3 adapter is tagged image-text-to-text by inheritance.
        let uc = from_metadata("someone/gemma-3-4b-roleplay", &tags(&["image-text-to-text", "roleplay"]), None);
        assert_eq!(uc.label, "Roleplay");
        let none = from_metadata("someone/my-gemma-tune", &tags(&["image-text-to-text", "lora"]), None);
        assert_eq!(none.label, NOT_STATED);
        assert_eq!(none.source, "none");
    }

    #[test]
    fn tags_outrank_the_name() {
        let uc = from_metadata("someone/coder-v2", &tags(&["medical"]), None);
        assert_eq!(uc.label, "Coding · Medicine");
        assert_eq!(uc.source, "tags");
    }

    #[test]
    fn the_card_settles_a_name_that_says_nothing() {
        let card = "---\nlicense: mit\n---\n# Rinlekha\n\nFine-tuned **Gemma 3 4B IT** on 640 synthetic \
                    credit memo examples for Indian NBFCs. Given a profile, it writes a memo.";
        let uc = from_metadata("a-anurag1024/rinlekha", &[], Some(card));
        assert_eq!(uc.label, "Finance");
        assert_eq!(uc.source, "card");
        assert!(uc.summary.as_deref().unwrap().starts_with("Fine-tuned Gemma 3 4B IT on 640"));
    }

    #[test]
    fn a_card_adds_its_summary_but_never_overrides_the_tags() {
        let tagged = from_metadata("x/finance-tune", &tags(&["finance"]), None);
        let card = "This adapter writes short poems about money and markets for children.";
        let merged = with_card(&tagged, "x/finance-tune", card);
        assert_eq!(merged.label, "Finance", "tags outrank card prose");
        assert_eq!(merged.summary.as_deref(), Some(card));

        let unknown = from_metadata("x/rinlekha", &[], None);
        let settled = with_card(&unknown, "x/rinlekha", "Writes credit memos for loan officers at banks.");
        assert_eq!(settled.label, "Finance");
        assert_eq!(settled.source, "card");
    }

    /// The two adapters installed on this machine that sat in "Unsorted".
    #[test]
    fn installed_adapters_from_the_storage_screen_find_a_slot() {
        let drug = from_metadata("someone/qwen2.5-3b-instruct-qlora-drug-ade-relation-extractor", &[], None);
        assert_eq!(slot_for(&drug), Some("research"), "{drug:?}");

        let rlvr = from_metadata("someone/elmo-rlvr-lora", &[], None);
        assert_eq!(slot_for(&rlvr), Some("reasoning"), "{rlvr:?}");
    }

    #[test]
    fn use_cases_without_a_routing_slot_are_not_misfiled() {
        for label in ["Lyrics & music", "Roleplay", "Creative writing", "Translation", "General chat", NOT_STATED] {
            let uc = UseCase { label: label.into(), language: None, source: "name".into(), summary: None };
            assert_eq!(slot_for(&uc), None, "{label}");
        }
        let combined = UseCase { label: "Roleplay · Coding".into(), language: None, source: "tags".into(), summary: None };
        assert_eq!(slot_for(&combined), Some("coding"), "the second label is read when the first has no slot");
    }

    #[test]
    fn a_language_adapter_says_so() {
        let uc = from_metadata("julienp79/occitan-gemma-3-4b-it-lora", &tags(&["lora", "oc"]), None);
        assert_eq!(uc.label, "Occitan language");
    }

    #[test]
    fn instruct_in_a_name_is_not_italian() {
        assert_eq!(language("someone/gemma-3-4b-it-sql", &[], None), None);
        assert_eq!(language("someone/x", &tags(&["it"]), None).as_deref(), Some("Italian"));
    }

    #[test]
    fn keywords_match_whole_words_only() {
        // "rap" must not fire inside "therapy" or "graph".
        let uc = from_metadata("someone/graph-therapy-bot", &[], None);
        assert_eq!(uc.label, "Medicine");
    }

    #[test]
    fn auto_generated_card_boilerplate_is_skipped() {
        let card = "<!-- This model card has been generated automatically according to the information \
                    the Trainer had access to. You should probably proofread and complete it, then remove \
                    this comment. -->\n\n# ocr\n\nThis model is a fine-tuned version of \
                    [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) on the ocr_finetune \
                    dataset. It achieves the following results.";
        let s = first_sentence(card).unwrap();
        assert!(s.starts_with("This model is a fine-tuned version of google/gemma-3-4b-it on the"), "{s}");
        assert!(!s.contains("http"), "links keep their text, not their URL: {s}");
    }

    /// HuggingFace's default card, as left by an author who never filled it in.
    #[test]
    fn the_unfilled_card_template_is_not_a_summary() {
        let card = "# Model Card for Model ID\n\n<!-- Provide a quick summary of what the model is/does. -->\n\n\
                    ## Model Details\n\n### Model Description\n\n\
                    This is the model card of a 🤗 transformers model that has been pushed on the Hub. \
                    This model card has been automatically generated.\n\n\
                    - **Developed by:** [More Information Needed]\n\
                    - **Model type:** [More Information Needed]\n\
                    - **Finetuned from model [optional]:** google/gemma-3-4b-it with a long tail\n\n\
                    ## How to Get Started with the Model\n\n\
                    Use the code below to get started with the model.\n";
        assert_eq!(first_sentence(card), None, "nothing in it describes the adapter");
    }

    #[test]
    fn a_real_sentence_after_the_template_is_found() {
        let card = "- **Developed by:** [More Information Needed]\n\n\
                    Korean tax-law Q/A specialised LoRA fine-tuned on Gemma 3 4B, converted to GGUF.";
        assert!(first_sentence(card).unwrap().starts_with("Korean tax-law Q/A"));
    }

    #[test]
    fn long_summaries_are_cut_at_a_word() {
        let card = format!("This adapter {}.", "does many useful things ".repeat(20));
        let s = first_sentence(&card).unwrap();
        assert!(s.chars().count() <= 201);
        assert!(s.ends_with('…'));
    }

    #[test]
    fn every_domain_label_is_short_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for (label, keywords) in DOMAINS {
            assert!(label.len() <= 16, "{label} is too long for the column");
            assert!(seen.insert(*label), "{label} listed twice");
            assert!(!keywords.is_empty());
        }
    }
}
